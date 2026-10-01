"""ANVIL (for projection matrices) + Adam (for everything else).

T4 port: torch.distributed is gone (world of one, so every reduce is a copy and every gather a
no-op), the CUDA-graph capture hooks are gone (fully eager), and the bf16 mantissa trick is
replaced by an fp32 master copy per ANVIL bank (fp16 bit tricks do not compose with fp32 the way
the bf16 ones did -- see _sign_aligned_decay_update).
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from track_1_short.perf.bank_scalars import BankScalarStaging
from track_1_short.perf.kernels.polar_express import XTX, XXT, ba_plus_cAA
from track_1_short.perf.kernels.transpose import transpose_add, transpose_copy
from track_1_short.perf.replicated_adam import ReplicatedAdam

# -----------------------------------------------------------------------------
# ANVIL: twin-rail momentum + whitening cascade (record #360)

# Six quintic spectral maps applied to the velocity Gram, re-derived for this run rather than
# taken from the Polar Express reference (https://arxiv.org/pdf/2505.16932).
ANVIL_MAPS = [
    (3.923798038567, -6.095026865488, 3.905234618423),
    (3.278126713798, -3.328923386476, 0.989127286973),
    (3.505298394150, -5.137358782410, 1.968325560615),
    (2.815058591845, -3.685181239622, 1.417196497642),
    (2.245503932403, -2.443826979899, 0.963091710461),
    (2.256537145403, -2.166840097229, 0.929501253245),
]

# Twin-rail velocity. Rail 0 is fast (beta scheduled by get_rail_beta until RAIL_ENGAGE_STEP, then
# RAIL_FAST_BETA); rail 1 is slow (RAIL_SLOW_BETA) and accumulates from step 0. Until the engage
# step the update reads the fast rail only; after it, RAIL_FAST_WEIGHT * fast + (1 - w) * slow.
RAIL_FAST_BETA, RAIL_SLOW_BETA, RAIL_FAST_WEIGHT, RAIL_ENGAGE_STEP = 0.85, 0.98, 0.4385, 514

# The cascade's input is divided by FROBENIUS_MARGIN * ||X||_F + FROBENIUS_EPS, which puts every
# singular value safely below 1 where the maps converge (the margin and eps of record #360).
FROBENIUS_MARGIN, FROBENIUS_EPS = 1.05, 1e-6
# Banks whose matrices have more rows than this run the a*X + X@B step as two kernels instead of one
# baddbmm (see anvil_cascade): true for mlp_bank (2816 rows), false for qk_bank (256) and vo_bank (768).
SPLIT_BADDBMM_MIN_ROWS = 1024


def anvil_cascade(grad_chunk: torch.Tensor, velocity: torch.Tensor, momentum_t: torch.Tensor,
                  split_baddbmm: bool, fast_beta_t: torch.Tensor, fast_weight_t: torch.Tensor):
    """Twin-rail Nesterov momentum, then the ANVIL cascade that drives every singular value to ~1.

    velocity is one fp32 [2, *chunk] tensor: rail 0 fast, rail 1 slow. Their blend gets a Nesterov
    lookahead (momentum_t), is cast to fp16 (record #360 used bf16; Turing has no bf16 and fp16 is
    strictly sharper), normalized by its Frobenius norm, and whitened by ANVIL_MAPS. momentum_t,
    fast_beta_t and fast_weight_t are 0-D device tensors (AnvilBank).
    """
    grad_chunk = grad_chunk.float()
    momentum = momentum_t.to(grad_chunk.dtype)
    velocity[0].lerp_(grad_chunk, 1 - fast_beta_t.to(grad_chunk.dtype))
    velocity[1].lerp_(grad_chunk, 1 - RAIL_SLOW_BETA)
    w = fast_weight_t.to(grad_chunk.dtype)
    blend = w * velocity[0] + (1 - w) * velocity[1]
    g = grad_chunk.lerp_(blend, momentum)

    X = g.half().contiguous()
    is_tall = g.size(-2) > g.size(-1)

    # The first Gram is taken on the unnormalized X: its trace is ||X||_F^2, which gives the
    # normalization for free; X and the Gram are then rescaled instead of recomputed.
    if is_tall:
        # Tall: use Triton kernels with X^T @ X (small) and right multiplication
        A = torch.empty((*X.shape[:-2], X.size(-1), X.size(-1)), device=X.device, dtype=X.dtype)
        XTX(X, out=A)  # A = X.T @ X
        tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
        d = tr.sqrt() * FROBENIUS_MARGIN + FROBENIUS_EPS
        X = (X.float() / d).half()
        A = (A.float() / d.square()).half()
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            XB_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_XB = torch.baddbmm if X.ndim > 2 else torch.addmm

        for k, (a, b, c) in enumerate(ANVIL_MAPS):
            if k > 0:
                XTX(X, out=A)  # A = X.T @ X
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b*A + c*(A@A)

            # Referencing X twice causes pytorch to make a defensive copy,
            # resulting in a cudaMemcpyAsync in baddbmm.
            # For large matrices (i.e., the mlp weights), it's faster to split
            # the operation into two kernels to avoid this.
            if split_baddbmm:
                XB_matmul(X, B, out=C)  # C = X @ B
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_XB(X, X, B, beta=a, out=C)  # C = a * X + X @ B

            X, C = C, X  # Swap references to avoid unnecessary copies
    else:
        # Wide: use Triton kernels with X @ X^T (small) and left multiplication
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        XXT(X, out=A)  # A = X @ X.mT
        tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
        d = tr.sqrt() * FROBENIUS_MARGIN + FROBENIUS_EPS
        X = (X.float() / d).half()
        A = (A.float() / d.square()).half()
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        # Select batched vs unbatched
        if split_baddbmm:
            BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

        for k, (a, b, c) in enumerate(ANVIL_MAPS):
            if k > 0:
                XXT(X, out=A)  # A = X @ X.mT
            ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A

            if split_baddbmm:
                BX_matmul(B, X, out=C)  # C = B @ X
                C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
            else:
                aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X

            X, C = C, X  # Swap references to avoid unnecessary copies

    return X

# -----------------------------------------------------------------------------
# Combined ANVIL + Adam Optimizer

@dataclass(slots=True)
class ParamConfig:
    """Per-parameter configuration for AnvilAndAdam."""
    label: str
    optim: str  # "adam" or "anvil"
    comms: str  # "replicated" or "sharded"
    adam_betas: tuple[float, float] | None
    lr_mul: float
    wd_mul: float
    lr: float
    initial_lr: float
    weight_decay: float
    # Adam-specific
    eps: float | None = None
    # ANVIL-specific
    reshape: tuple | None = None
    chunk_size: int | None = None
    momentum: float | None = None
    beta2: float | None = None
    per_matrix_lr_mul: list[float] | None = None


@dataclass(slots=True)
class AnvilBank:
    """One ANVIL bank's update: every tensor anvil_bank_update reads or writes.

    Everything is allocated once and only ever written in place: the reduce-scatter equivalent lands
    in `grad`, and the per-step scalars are device tensors refreshed by one H2D at the top of every
    step() (perf/bank_scalars.py).
    """
    label: str
    grad: Tensor          # [chunk, rows, cols] fp16: this bank's (copied) gradient
    velocity: Tensor      # [2, chunk, rows, cols] fp32: fast and slow rail
    lane_energy: Tensor   # the equalizer's per-lane EMA
    master: Tensor        # [chunk, rows, cols] fp32: the master weight the update writes
    p_slice: Tensor       # the parameter itself (a [chunk, rows, cols] fp16 view)
    momentum: Tensor      # 0-D fp32: the Nesterov lookahead (the scheduled rail beta)
    eff_wd: Tensor        # 0-D fp32: wd_mul * weight_decay * lr
    fast_beta: Tensor     # 0-D fp32: the fast rail's beta
    fast_weight: Tensor   # 0-D fp32: the fast rail's weight in the blend
    eff_lr: Tensor        # [chunk, 1, 1] fp32: lr_mul * per-matrix multiplier * lr
    beta2: float          # the equalizer's EMA decay
    split_baddbmm: bool
    red_dim: int

    def mutated(self) -> dict[str, Tensor]:
        """What the update writes."""
        return {"velocity": self.velocity, "lane_energy": self.lane_energy, "master": self.master,
                "p_slice": self.p_slice}


class AnvilAndAdam:
    """
    Combined optimizer that handles both ANVIL (for projection matrices) and
    Adam (for embeddings/scalars/gate weights).

    ANVIL (record #360), differences from standard Muon (https://kellerjordan.github.io/posts/muon/):
    - Twin-rail momentum: a fast and a slow velocity EMA, blended after RAIL_ENGAGE_STEP
    - Newton-Schulz is replaced with the ANVIL cascade: six re-derived quintic maps after a
      Frobenius normalization (successor of Polar Express)
    - Per-lane energy equalizer, the low-rank variance estimator from NorMuon
      (https://arxiv.org/pdf/2510.05491)
    - Cautious weight decay gated on the slow rail's sign
    - Mantissa tracking for precision

    Adam (for embeddings/scalars/gates):
    - Standard Adam with bias correction
    - Cautious weight decay

    Configuration:
    Unlike torch.optim.Optimizer, this class uses per-parameter configs from a `param_table` dict
    and does not include parameter "groups". All parameters require a .label attribute, and a
    corresponding entry in the param_table to specify their hyperparameters (lr_mul, wd_mul, adam_betas, etc.).

    Communication and ordering (T4 port: world of one):
    The multi-GPU gradient communication is gone: every parameter is "replicated" in effect, each
    sharded label's reduce is a copy into its bank/state buffer, and gathers are no-ops (chunk sizes
    equal the full shapes). scatter_order and work_order still enumerate every label once.

    # Contributors include @YouJiacheng, @KonstantinWilleke, @alexrgilbert, @adricarda,
    # @tuttyfrutyee, @vdlad, @ryanyang0, @vagrawal, @varunneal, @chrisjmccormick
    """
    def __init__(self, named_params, param_table: dict, scatter_order: list, work_order: list,
                 adam_defaults: dict, anvil_defaults: dict, bank_update: Callable[[AnvilBank], None]):
        """bank_update(bank) runs one bank's ANVIL update (eager: anvil_bank_update)."""
        self.world_size = 1
        self.rank = 0

        # Store defaults for each optimizer type
        self.adam_defaults = adam_defaults
        self.anvil_defaults = anvil_defaults
        self.param_table = param_table
        self.scatter_order = scatter_order
        self.work_order = work_order

        # Collect params by label and build config
        self.param_cfgs: dict[nn.Parameter, ParamConfig] = {}
        self.param_states: dict[nn.Parameter, dict] = {}
        self._param_by_label: dict[str, nn.Parameter] = {}
        for name, param in named_params:
            label = getattr(param, "label", None)
            assert label is not None and label in param_table  # all params must have valid label
            assert label not in self._param_by_label  # exactly one param per label
            self._param_by_label[label] = param
            self._build_param_cfg(param, label)

        # Assert scatter_order and work_order match present labels exactly
        present = self._param_by_label.keys()
        assert set(scatter_order) == present and set(work_order) == present

        # The replicated Adam params are reduced and updated as flat buffers (perf/replicated_adam.py).
        replicated = [p for p, c in self.param_cfgs.items() if c.optim == "adam" and c.comms == "replicated"]
        self.replicated_labels = {self.param_cfgs[p].label for p in replicated}

        # Initialize state for all params
        self._init_state()
        self.replicated = ReplicatedAdam(replicated, self.param_cfgs, self.param_states, device=replicated[0].device)

        # Adam's per-parameter scalars: 0-D CPU tensors to avoid recompilation
        self._step_size_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # ANVIL's twin-rail scalars for this step (set_rails)
        self.fast_beta, self.fast_weight = 0.0, 1.0
        self._init_banks()
        self.bank_update = bank_update

        # Track async operations
        self._reduce_futures: dict[nn.Parameter, tuple] = {}

        # Embed/lm_head tying state
        self.split_embed = False
        self._lm_head_param = self._param_by_label.get("lm_head")
        self._embed_param = self._param_by_label.get("embed")

    def _build_param_cfg(self, param: nn.Parameter, label: str):
        """Build config for a single parameter from param_table."""
        table_entry = self.param_table[label]
        optim = table_entry["optim"]
        comms = table_entry["comms"]
        adam_betas = table_entry.get("adam_betas")
        lr_mul = table_entry.get("lr_mul", 1.0)
        wd_mul = table_entry.get("wd_mul", 1.0)

        if optim == "adam":
            chunk_size = param.shape[0] // self.world_size if comms == "sharded" else None
            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.adam_defaults["lr"],
                initial_lr=self.adam_defaults["lr"],
                weight_decay=self.adam_defaults["weight_decay"],
                eps=self.adam_defaults["eps"],
                chunk_size=chunk_size,
            )
        elif optim == "anvil":
            reshape = getattr(param, "reshape", None)
            if reshape is None:
                raise ValueError(f"ANVIL param {label} must have .reshape attribute")
            if reshape[0] % self.world_size != 0:
                raise ValueError(f"reshape[0]={reshape[0]} must be divisible by world_size")

            chunk_size = reshape[0] // self.world_size
            chunk_shape = (chunk_size, *reshape[1:])
            # Shape-based LR multiplier for ANVIL
            shape_mult = max(1.0, chunk_shape[-2] / chunk_shape[-1]) ** 0.5 if len(chunk_shape) >= 2 else 1.0
            lr_mul = shape_mult * lr_mul

            # Per-matrix LR multipliers for MLP c_proj (2x LR on odd indices). Matrices the model
            # marks as frozen (`param.frozen_matrices`, e.g. a removed layer's MLP) get LR 0, which
            # also zeroes their decay term: they do not move at all.
            per_matrix_lr_mul = None
            if label == "mlp_bank":
                start_idx = self.rank * chunk_size
                frozen = getattr(param, "frozen_matrices", frozenset())
                per_matrix_lr_mul = []
                for i in range(chunk_size):
                    global_idx = start_idx + i
                    is_c_proj = (global_idx % 2 == 1)
                    per_matrix_lr_mul.append(0.0 if global_idx in frozen else 2.0 if is_c_proj else 1.0)

            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.anvil_defaults["lr"],
                initial_lr=self.anvil_defaults["lr"],
                weight_decay=self.anvil_defaults["weight_decay"],
                reshape=reshape,
                chunk_size=chunk_size,
                momentum=self.anvil_defaults["momentum"],
                beta2=self.anvil_defaults["beta2"],
                per_matrix_lr_mul=per_matrix_lr_mul,
            )
        else:
            raise ValueError(f"Unknown optim type: {optim}")

        self.param_cfgs[param] = p_cfg

    def _init_state(self):
        """Initialize optimizer state for all parameters."""
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam":
                # Sharded params use chunk state, replicated use full state
                if p_cfg.comms == "sharded":
                    chunk = param[:p_cfg.chunk_size]
                else:
                    chunk = param
                exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=param.device)
                self.param_states[param] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg))

            elif p_cfg.optim == "anvil":
                chunk_shape = (p_cfg.chunk_size, *p_cfg.reshape[1:])

                # Twin-rail velocity (FP32 for precision): [0] fast rail, [1] slow rail
                velocity = torch.zeros(
                    (2, *chunk_shape), dtype=torch.float32, device=param.device
                )

                # Per-lane update energy for the equalizer - reduced along the longer dimension
                if chunk_shape[-2] >= chunk_shape[-1]:
                    lane_shape = (*chunk_shape[:-1], 1)
                else:
                    lane_shape = (*chunk_shape[:-2], 1, chunk_shape[-1])
                lane_energy = torch.zeros(
                    lane_shape, dtype=torch.float32, device=param.device
                )

                # The fp32 master weight (replaces record #360's uint16 mantissa shadow: fp16's bit
                # layout is not the high half of fp32's, so the bf16-era trick cannot decode it).
                master = param.data.view(p_cfg.reshape).float()

                self.param_states[param] = dict(
                    velocity=velocity,
                    lane_energy=lane_energy,
                    master=master,
                )

    # -----------------------------------
    # ANVIL banks: fixed buffers and device scalars

    def _init_banks(self):
        """One AnvilBank per ANVIL parameter, with its persistent gradient buffer and its per-step
        scalars: views into one device buffer that one H2D per step refreshes (perf/bank_scalars.py)."""
        anvil = [(p, c) for p, c in self.param_cfgs.items() if c.optim == "anvil"]
        device = anvil[0][0].device
        self.bank_scalars = BankScalarStaging([c.chunk_size for _, c in anvil], device)
        self.banks: dict[str, AnvilBank] = {}
        for i, (param, cfg) in enumerate(anvil):
            # fp16, so anvil_cascade's grad.float() is a copy and its in-place lerp never writes `grad`.
            assert param.dtype == torch.float16, f"{cfg.label} must be fp16 before the optimizer is built"
            chunk_shape = (cfg.chunk_size, *cfg.reshape[1:])
            self.banks[cfg.label] = AnvilBank(
                label=cfg.label,
                grad=torch.empty(chunk_shape, dtype=param.dtype, device=device),
                **self.live_bank_state(cfg.label),
                **self.bank_scalars.scalars(i),
                eff_lr=self.bank_scalars.eff_lr(i),
                beta2=cfg.beta2,
                split_baddbmm=chunk_shape[-2] > SPLIT_BADDBMM_MIN_ROWS,
                red_dim=-1 if chunk_shape[-2] >= chunk_shape[-1] else -2,
            )

    def live_bank_state(self, label: str) -> dict[str, Tensor]:
        """The bank's state tensors as the live optimizer and parameter hold them now."""
        param = self._param_by_label[label]
        cfg, state = self.param_cfgs[param], self.param_states[param]
        lo = self.rank * cfg.chunk_size
        return dict(velocity=state["velocity"], lane_energy=state["lane_energy"], master=state["master"],
                    p_slice=param.data.view(cfg.reshape)[lo:lo + cfg.chunk_size])

    def stage_bank_scalars(self):
        """Upload this step's ANVIL scalars (from the current ParamConfigs and rails) in one H2D."""
        fields, eff_lrs = [], []
        for label in self.banks:
            cfg = self.param_cfgs[self._param_by_label[label]]
            # The products keep the eager code's order (lr_mul * matrix multiplier * lr).
            fields.append((cfg.momentum, cfg.wd_mul * cfg.weight_decay * cfg.lr, self.fast_beta, self.fast_weight))
            per_matrix = cfg.per_matrix_lr_mul or [1.0] * cfg.chunk_size
            eff_lrs.append([cfg.lr_mul * m * cfg.lr for m in per_matrix])
        self.bank_scalars.upload(fields, eff_lrs)

    # -----------------------------------
    # Reduce/Gather operations (world of one)

    def _launch_reduce(self, param: nn.Parameter, grad: Tensor):
        """Land the gradient in the destination buffer a world-1 reduce would fill (a plain copy)."""
        p_cfg = self.param_cfgs[param]
        if p_cfg.comms == "sharded":
            if p_cfg.optim == "anvil":
                # ANVIL: the shard is the whole bank; copy into its persistent gradient buffer.
                grad_chunk = self.banks[p_cfg.label].grad
                grad_chunk.copy_(grad.view(p_cfg.reshape))
                self._reduce_futures[param] = (None, grad_chunk)
            else:
                # Adam: with chunk == full shape, reduce_scatter is the identity.
                self._reduce_futures[param] = (None, grad)

    # -----------------------------------
    # State management

    def reset(self):
        """Reset ANVIL velocity/lane state and split_embed state (called on training reset)."""
        self.split_embed = False
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "anvil":
                p_state = self.param_states[param]
                p_state["velocity"].zero_()
                p_state["lane_energy"].zero_()
                # Re-derive the master from the parameter itself (zeroing it would destroy the weights).
                p_state["master"].copy_(param.data.view(p_cfg.reshape).float())

    def copy_lm_state_to_embed(self):
        """
        Copy the optimizer state from the lm_head to the embed at the untie point (world of one:
        lm_head's "shard" is the full (768, 50304) matrix, so the reshard is one transpose-copy).
        """
        lm_head = self._lm_head_param
        embed = self._embed_param
        lm_state = self.param_states[lm_head]
        embed_state = self.param_states[embed]

        embed_state['step'] = lm_state['step'] # Preserve step count for bias correction

        for key in ["exp_avg", "exp_avg_sq"]:
            embed_state[key].copy_(lm_state[key].T)

        # Mark as split
        self.split_embed = True

    def state_dict(self):
        """Return the optimizer state as a dict."""
        return {
            "param_states": {id(p): s for p, s in self.param_states.items()},
            "param_cfgs": {id(p): s for p, s in self.param_cfgs.items()},
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state from a dict. Tensors are copied in place: the replicated params' moments
        are views into ReplicatedAdam's flat buffers and must never be rebound."""
        # Build id->param mapping
        id_to_param = {id(p): p for p in self.param_cfgs}

        for param_id, saved_p_state in state_dict["param_states"].items():
            if param_id in id_to_param:
                param = id_to_param[param_id]
                p_state = self.param_states[param]
                for k, v in saved_p_state.items():
                    if isinstance(v, torch.Tensor) and k in p_state:
                        p_state[k].copy_(v)
                    else:
                        p_state[k] = v

    # -----------------------------------
    # Unified optimizer step with explicit ordering

    @torch.no_grad()
    def step(self, do_adam: bool) -> None:
        """
        Combined optimizer step, in one pass (world of one: no collectives to schedule).

        Args:
            do_adam: If True, update Adam params. ANVIL params always updated.

        Flow:
        0. Adam steps: the replicated params' copy + fused Adam launch (perf/replicated_adam.py)
        1. Scatter phase: land each sharded label's gradient (a copy; lm_head first aggregates
           embed.grad.T while the embeddings are tied)
        2. Work phase: wait for nothing, compute every update
        3. Finalize: while tied, copy lm_head.T --> embed.data

        While the embeddings are tied:
        - Update math is only done on lm_head (embed's moments stay untouched).
        - We add embed.grad.T into lm_head.grad before the update.
        - We copy lm_head.data.T --> embed.data after it.
        """
        lm_param, embed_param = self._lm_head_param, self._embed_param
        # The ANVIL scalars the bank updates read this step.
        self.stage_bank_scalars()

        # ===== Phase 0: one flat copy per dtype for the replicated params =====
        if do_adam:
            self.replicated.launch_reduce()

        # ===== Phase 1: land gradients in scatter_order =====
        for label in self.scatter_order:
            param = self._param_by_label[label]
            p_cfg = self.param_cfgs[param]

            if p_cfg.optim == "adam" and not do_adam:
                continue
            if param.grad is None or label in self.replicated_labels:
                continue

            # lm_head when tied: aggregate embed.grad.T (tiled Triton transpose-add)
            if label == "lm_head" and do_adam and not self.split_embed:
                if embed_param is not None and embed_param.grad is not None:
                    transpose_add(embed_param.grad, param.grad)

            # Skip embed when tied (copied from lm_head after the update)
            if label == "embed" and not self.split_embed:
                continue

            self._launch_reduce(param, param.grad)

        # ===== Phase 2: compute the updates in work_order =====
        replicated_done = False
        for label in self.work_order:
            if label in self.replicated_labels:
                # The replicated params are independent, so one fused pass at the first of them equals
                # updating each in turn.
                if do_adam and not replicated_done:
                    self.replicated.update()
                    replicated_done = True
                continue
            param = self._param_by_label[label]
            if param not in self._reduce_futures:
                continue

            p_cfg = self.param_cfgs[param]
            if p_cfg.optim == "adam" and not do_adam:
                continue
            future, grad_chunk = self._reduce_futures[param]
            if future is not None:
                future.wait()

            if p_cfg.optim == "adam":
                self._adam_update(param, grad_chunk, p_cfg)
            else:
                self._anvil_update(p_cfg, grad_chunk)
            # world 1: p_slice is already the full parameter; nothing to gather.

        # ===== Phase 3: sync embed if tied =====
        if do_adam and not self.split_embed and embed_param is not None and lm_param is not None:
            transpose_copy(lm_param.data, embed_param.data)

        self._reduce_futures.clear()

        # Clear grads for updated params
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam" and not do_adam:
                continue  # Don't clear Adam grads on even steps
            param.grad = None

    # -----------------------------------
    # Adam update

    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig) -> Tensor:
        """Apply Adam update to a parameter. Returns the updated p_slice."""
        beta1, beta2 = p_cfg.adam_betas
        lr = p_cfg.lr * p_cfg.lr_mul

        # Get parameter slice
        if p_cfg.comms == "sharded":
            p_slice = param[self.rank * p_cfg.chunk_size:(self.rank + 1) * p_cfg.chunk_size]
        else:
            p_slice = param

        p_state = self.param_states[param]
        p_state["step"] += 1
        t = p_state["step"]

        bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))
        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)

        AnvilAndAdam._adam_update_step(
            p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],
            beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
        )

        return p_slice

    @torch.no_grad()
    @staticmethod
    def _adam_update_step(p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
        """Eager Adam update step: moments stay fp32, the fp16 parameter is written through an fp32
        round-trip (fp16.add_(fp32) would raise)."""
        g = g_slice.float()
        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)
        # Cautious weight decay
        p_f = p_slice.float()
        mask = (update * p_f) > 0
        update.addcmul_(p_f, mask, value=eff_wd_t)
        p_slice.copy_(p_f.sub_(update))

    # -----------------------------------
    # ANVIL update

    def set_rails(self, fast_beta: float, fast_weight: float):
        """Set the fast rail's beta and its weight in the twin-rail blend for this step."""
        self.fast_beta, self.fast_weight = fast_beta, fast_weight

    def _anvil_update(self, p_cfg: ParamConfig, grad_chunk: Tensor) -> Tensor:
        """Apply the ANVIL update to this rank's slice of a bank. Returns the updated p_slice."""
        bank = self.banks[p_cfg.label]
        assert grad_chunk is bank.grad
        self.bank_update(bank)
        return bank.p_slice

    @staticmethod
    def _sign_aligned_decay_update(p_slice, master, update, wd_tensor, lr_tensor, gate_src):
        """
        Sign-aligned (cautious) weight decay + parameter update: decay applies only where the
        parameter and `gate_src` (the slow rail, a denoised gradient estimate) agree in sign.
        wd_tensor is a 0-D device tensor, lr_tensor a [matrices, 1, 1] device tensor (per-matrix lr).

        The math runs on `master`, an fp32 shadow of the whole parameter; the fp16 parameter is
        re-rounded from it every step. Record #360 instead kept the low 16 bits of an fp32 shadow in a
        uint16 buffer and re-decoded the high bits from the bf16 parameter's raw bits — that decode
        assumes bf16's layout (which IS fp32's top half); fp16's layout is not, so the trick cannot be
        ported, only the fp32-master effect it was approximating.
        """
        update = update.float()
        wd_factor = wd_tensor.to(torch.float32)
        lr_factor = lr_tensor.to(torch.float32)
        aligned = (gate_src.float() * master) >= 0
        master.sub_(master * aligned * wd_factor * lr_factor + update * lr_factor)
        p_slice.copy_(master.to(p_slice.dtype))

    @staticmethod
    def _rail_equalizer(v_chunk, lane_energy, beta2, red_dim):
        """Equalize lanes (reduced over red_dim, the longer matrix dimension): each lane's update is
        rescaled by the inverse root of an EMA of its mean squared update, then the whole matrix is
        rescaled back to its pre-equalization norm. Low-rank, Adafactor-like variance estimate from
        NorMuon (https://arxiv.org/pdf/2510.05491)."""
        lane_power = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
        lane_len = v_chunk.size(red_dim)
        pre_norm = lane_power.sum(dim=(-2, -1), keepdim=True).mul_(lane_len).sqrt_()
        lane_energy.lerp_(lane_power.to(dtype=lane_energy.dtype), 1 - beta2)
        lane_gain = lane_energy.clamp_min(1e-10).rsqrt_()
        post_power = (lane_power * lane_len) * lane_gain.float().square()
        post_norm = post_power.sum(dim=(-2, -1), keepdim=True).sqrt_()
        eq_scale = lane_gain * (pre_norm / post_norm.clamp_min_(1e-10))
        return v_chunk.mul_(eq_scale.type_as(v_chunk))


def anvil_bank_update(bank: AnvilBank):
    """One bank's ANVIL update, from its (copied) gradient."""
    # 1. Twin-rail Nesterov momentum + ANVIL whitening cascade
    v_chunk = anvil_cascade(
        bank.grad, bank.velocity, bank.momentum,
        split_baddbmm=bank.split_baddbmm,
        fast_beta_t=bank.fast_beta, fast_weight_t=bank.fast_weight,
    )
    # 2. Equalize per-lane update energy
    v_chunk = AnvilAndAdam._rail_equalizer(v_chunk, bank.lane_energy, bank.beta2, bank.red_dim)
    # 3. Update the parameter in place, with weight decay gated on the slow rail's sign. eff_lr is per
    #    matrix: MLP c_proj gets 2x, frozen matrices 0.
    AnvilAndAdam._sign_aligned_decay_update(
        bank.p_slice, bank.master, v_chunk, bank.eff_wd, bank.eff_lr, bank.velocity[1],
    )
