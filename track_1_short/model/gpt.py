"""The GPT model: embeddings, 11 transformer blocks with MUDD skip connections, and the loss."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from track_1_short.model.attention import AttnArgs, CausalSelfAttention, Yarn
from track_1_short.model.layers import CastedLinearT, next_multiple_of_n, norm
from track_1_short.losses import softcapped_ce
from track_1_short.perf.residual_fusion import rms_norm_with_head, scale, scale_add
from track_1_short.sampled_softmax import SampledLoss

# Layer topology (11 layers). Depth cut from record #360 (ANVIL2): layer 7 is removed whole -- its
# residual scaling and x0 injection stay, so cache[7] still exists -- and layers 4 and 9 run
# their MLP only. Layer 6 has had no attention since @YouJiacheng; it adds a gated skip from layer 3.
NUM_LAYERS = 11
NO_ATTN_LAYERS = (4, 6, 7, 9)
NO_MLP_LAYERS = (7,)
ATTN_LAYERS = tuple(i for i in range(NUM_LAYERS) if i not in NO_ATTN_LAYERS)  # (0, 1, 2, 3, 5, 8, 10)
# Long sliding window (with a partial key offset) on these layers, short on the other attention layers.
LONG_WINDOW_LAYERS = (3, 10)
PAIRED_HEAD_LAYERS = (0, 2, 5)

# Mixed head widths, from record #360. The long-window layers keep full-width d_qk = 128 query/key
# heads; every other attention layer runs d_qk = 64 with a fully rotating rotary. The HALF_V_LAYERS
# also halve the value/output head (d_v = 64); record #360 lists 1, 4, 7, 8, of which 4 and 7 have
# no attention. The banks keep full-width (128) heads: a narrow layer reads the leading 64 of each
# head's Q/K/V rows and of each head's O columns, and the rest never get a gradient.
WIDE_QK_LAYERS = LONG_WINDOW_LAYERS
HALF_V_LAYERS = (1, 8)
NARROW_QK_LAYERS = tuple(i for i in ATTN_LAYERS if i not in WIDE_QK_LAYERS + HALF_V_LAYERS)  # (0, 2, 5)
NARROW_HEAD_DIM = 64  # the narrow query/key heads' width, and the halved value/output heads'
# Softmax scales at the start of training (YaRN grows them with the window): the narrow layers' from
# record #360; the wide layers' is record #360's Yarn default.
NARROW_ATTN_SCALE = 0.13
WIDE_ATTN_SCALE = 0.085
# Attention layers grouped by packed QKV weight shape (kept for ATTN_BANK_ORDER's ordering; the fp8
# per-group weight caches are gone with the fp8 path).
ATTN_WIDTH_GROUPS = {"narrow": NARROW_QK_LAYERS, "half_v": HALF_V_LAYERS, "wide": WIDE_QK_LAYERS}
# Attention-bank slot order: slot j of qk_bank / vo_bank belongs to layer ATTN_BANK_ORDER[j], so each
# width group is a contiguous run of slots, and the d_qk = 64 groups (narrow, half_v) lead.
ATTN_BANK_ORDER = sum(ATTN_WIDTH_GROUPS.values(), ())  # (0, 2, 5, 1, 8, 3, 10)
NUM_QK64_SLOTS = len(NARROW_QK_LAYERS) + len(HALF_V_LAYERS)
assert sorted(ATTN_BANK_ORDER) == list(ATTN_LAYERS), "every attention layer needs exactly one width group"
assert ATTN_BANK_ORDER[NUM_QK64_SLOTS:] == WIDE_QK_LAYERS
# Per-head output gates (from the MUDD gates) on these attention layers.
ATTN_GATE_LAYERS = (3, 10)
# Gated XSA on these attention layers, its per-head strength from the pre MUDD gate.
XSA_LAYERS = (1, 3)
# The residual stream after these layers is kept for later skips: layer 6 re-adds cache[3], and the
# last layer and the post-loop MUDD mix read cache[7].
CACHE_LAYERS = (3, 7)
# Token value embeddings are added to V on these layers, one embedding plane each.
VALUE_EMBED_LAYERS = (1, 2, 8, 10)
# ...each gated by a learned per-head gate, except the last layer, whose gate comes from MUDD.
VALUE_EMBED_GATE_LAYERS = (1, 2, 8)
# The gate reads this many leading channels of both the normed attention input and the value embedding.
VALUE_EMBED_GATE_CHANNELS = 6
# Residual injection sites, from record #360. x0 (the normed input embedding) is added back into the
# residual stream on X0_INJECT_LAYERS; each site has its own per-token MUDD gate lane. The last layer
# injects it through its own MUDD coefficients (mu[10]) instead; layer 6 injects neither. The hashed
# n-gram embedding and its injection are gone with the n-gram table (T4 port).
X0_INJECT_LAYERS = (0, 1, 2, 4, 5, 7)
assert not {6, 10} & set(X0_INJECT_LAYERS)
# MUDD gates (init_mudd_gate): the pre gate, computed from x0, serves the layers before POST_GATE_LAYER;
# the post gate, computed at the start of POST_GATE_LAYER, serves it and the layers after.
POST_GATE_LAYER = 4
PRE_GATE_X0_LAYERS = tuple(i for i in X0_INJECT_LAYERS if i < POST_GATE_LAYER)             # (0, 1, 2)
POST_GATE_X0_LAYERS = tuple(i for i in X0_INJECT_LAYERS if i >= POST_GATE_LAYER)           # (4, 5, 7)
PRE_GATE_ATTN_GATE_LAYERS = tuple(i for i in ATTN_GATE_LAYERS if i < POST_GATE_LAYER)      # (3,)
POST_GATE_ATTN_GATE_LAYERS = tuple(i for i in ATTN_GATE_LAYERS if i >= POST_GATE_LAYER)    # (10,)
assert all(i < POST_GATE_LAYER for i in XSA_LAYERS), "the XSA strengths come from the pre gate"
# Layer 8 runs a second MLP in parallel, from MLP bank slot 11 (the slot that was sharding padding).
PARALLEL_MLP_LAYER, PARALLEL_MLP_SLOT = 8, 11
# The post-loop MUDD mix splits model_dim into this many channel groups, each with its own coefficient delta.
MUDD_GROUPS = 12
# MUDD coefficients at the start of the last layer (init_mudd lists them).
LAST_LAYER_MUDD_COEFS = 14
# MUDD gate lane widths (init_mudd_gate): one lane per head for the XSA strengths and the attention
# gates, one lane per injection site for x0 / the n-gram embedding, one for the layer-6 skip.
MUDD_GATE_HEAD_LANES = 6
MUDD_GATE_SCALE = 0.1  # the gates' output scale at init; biases are stored pre-divided by it
# MLP bank: 12 slots of (c_fc, c_proj), 24 matrices for even sharding over 8 GPUs. Slot i is layer i's
# MLP, slot 11 is PARALLEL_MLP_SLOT, slot 7 is dead (NO_MLP_LAYERS) but keeps the bank even.
NUM_MLP_SLOTS = 12
# MLP hidden size, cut from 4 * 768 = 3072 in record #360.
MLP_HIDDEN_DIM = 2816

# Validation computes the loss over slabs of this many rows (record #360 used 32768, whose 6.6 GB
# fp32 [rows, vocab] logits do not fit comfortably next to the rest of a 16 GB T4), so the logit
# block stays small.
EVAL_CE_SLAB_ROWS = 8192

# relu(x @ c_fc.T)^2 @ c_proj, the record #360 MLP, as the plain differentiable PyTorch expression
# (the fp8 fused triton kernel needs sm89+ FP8 conversions; the bf16 triton kernel is dropped with it).
# https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
def ReLUSqrdMLP(x_normed, c_fc, c_proj):
    # mlp_bank stores both matrices as (mlp_hdim, dim); c_fc coincides with F.linear's (out, in)
    # convention, c_proj must be transposed to (dim, mlp_hdim) for it.
    return F.linear(F.relu(F.linear(x_normed, c_fc)).square(), c_proj.T)

@dataclass(slots=True)
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    prefix_weight: torch.Tensor
    ws_short: int
    ws_long: int
    train_max_seq_len: int  # longest attention segment in a training batch
    # Training only: the candidate set when this step's loss is a sampled softmax; None = full softmax.
    sampled_loss: SampledLoss | None = None

class GPT(nn.Module):
    """The fp16 eager model: every projection runs in fp16 (Turing has no bf16), with a static loss
    scale and isfinite guard instead of a GradScaler (training.py / train_gpt.py)."""
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int,
                 *, world_size: int, device: torch.device):
        super().__init__()
        assert num_layers == NUM_LAYERS
        self.world_size = world_size
        self.device = device
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        # there are only 50257 unique GPT-2 tokens; extend to nearest multiple of 128 for efficiency.
        # suggested by @Grad62304977, originates from Karpathy's experiments.
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)

        # Prefix-token lookup table for prefix token prediction. Allocated here, but filled
        # after the clock starts (see "start the clock") so the build is charged to training
        # time. -1 means "no valid prefix" == term disabled, which is what warmup runs with.
        self.register_buffer("prefix_table", torch.full((self.vocab_size,), -1, dtype=torch.int64), persistent=False)

        # Canonical token mask for the validation softmax, one bit per (prev, cur) pair.
        # Allocated all-zero == no masking.
        self.register_buffer("canon_mask", torch.zeros(self.vocab_size, self.vocab_size // 8, dtype=torch.uint8), persistent=False)

        self.lm_head = CastedLinearT(model_dim, self.vocab_size, x_s=100/448, w_s=2.0/448, grad_s=(0.75 / 8) / 448)
        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)

        self.embed = nn.Embedding(self.vocab_size, model_dim)
        with torch.no_grad():
            # tie embed and lm_head at init
            self.embed.weight.copy_(self.lm_head.weight.T)

        self.init_attn(model_dim, head_dim, num_heads, max_seq_len)
        self.init_mlp(model_dim)
        self.init_misc(model_dim, num_layers)
        self.init_mudd(num_layers, model_dim)
        self.init_mudd_gate(model_dim)

        # Auto-label parameters
        for name, param in self.named_parameters():
            param.label = name.replace('.weight', '')

    def init_attn(self, model_dim, head_dim, num_heads, max_seq_len):
        # One attention module per attention layer, at the layer's head widths (no learned params --
        # weights come from qk_bank/vo_bank). The patched FA3 takes unequal query/key and value widths
        # only as (64, 128) / (128, 64).
        assert head_dim == 2 * NARROW_HEAD_DIM
        self.attn = nn.ModuleDict({
            str(layer): CausalSelfAttention(
                num_heads, head_dim, qk_dim=self.attn_qk_dim(layer), v_dim=self.attn_v_dim(layer),
                val_max_seq_len=max_seq_len, paired=layer in PAIRED_HEAD_LAYERS,
            )
            for layer in ATTN_LAYERS
        })
        # Rotary tables: one per (query/key width, pairing) in use.
        self.yarn = Yarn(NARROW_HEAD_DIM, max_seq_len, attn_scale=NARROW_ATTN_SCALE, device=self.device)
        self.yarn_paired_head = Yarn(
            NARROW_HEAD_DIM, max_seq_len, paired=True, attn_scale=NARROW_ATTN_SCALE, device=self.device,
        )
        self.yarn_wide = Yarn(head_dim, max_seq_len, attn_scale=WIDE_ATTN_SCALE, device=self.device)
        assert not set(PAIRED_HEAD_LAYERS) & set(WIDE_QK_LAYERS), "no paired rotary table at full width"

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        # spherical gaussian init by @photomz
        # One [vocab, model_dim] plane per VALUE_EMBED_LAYERS entry. The T4 port reads it with plain
        # indexing, so its gradient is a dense [num_ve * vocab, model_dim] fp16 tensor (the sharded
        # row-pull machinery is gone with multi-GPU support).
        num_ve = len(VALUE_EMBED_LAYERS)
        self.value_embeds = nn.Parameter(0.01 * torch.randn(num_ve * self.vocab_size, model_dim, dtype=torch.float16))

        # value embedding gate weights, one per VALUE_EMBED_GATE_LAYERS entry
        self.ve_gate_bank = nn.Parameter(torch.zeros(len(VALUE_EMBED_GATE_LAYERS), num_heads, 12))

        # Parameter banks for sharded optimization, by @chrisjmccormick. Only ATTN_LAYERS own
        # attention weights; bank slot j belongs to layer ATTN_BANK_ORDER[j]. Rows are full width
        # (head_dim per head) on every layer; the narrower layers read a leading part of each head.
        num_slots = len(ATTN_BANK_ORDER)
        hdim = num_heads * head_dim

        # QK bank: per-head-pair ANVIL groups for Q, K weights. Each pair of adjacent heads gets its
        # own independent whitening: a slot's 2 * num_heads heads (Q heads, then K heads) are
        # num_heads groups of two full-width heads.
        qk_groups_per_slot = num_heads
        num_qk_groups = num_slots * qk_groups_per_slot  # 42
        self._num_qk_groups = num_qk_groups
        num_qk_padded = next_multiple_of_n(num_qk_groups, n=self.world_size)  # 48
        self.qk_bank = nn.Parameter(torch.empty(num_qk_padded, 2 * head_dim, model_dim))
        self.qk_bank.reshape = (num_qk_padded, 2 * head_dim, model_dim)

        # VO bank: per-layer ANVIL groups for V and O weights; slot j's V is matrix 2j, its O 2j + 1.
        # V is stored [out = hdim, in = model_dim] and O [out = model_dim, in = hdim], both nn.Linear
        # layout (record #360); the two shapes coincide because hdim == model_dim.
        assert hdim == model_dim
        num_vo_real = 2 * num_slots  # 14
        num_vo_padded = next_multiple_of_n(num_vo_real, n=self.world_size)  # 16
        self.vo_bank = nn.Parameter(torch.empty(num_vo_padded, hdim, model_dim))
        self.vo_bank.reshape = (num_vo_padded, hdim, model_dim)

        # improved init scale by @YouJiacheng and @srashedll. Every live row is drawn, including the
        # rows the narrower layers never read (record #360 does the same).
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.qk_bank[:num_qk_groups].uniform_(-bound, bound)
            self.qk_bank[num_qk_groups:].zero_()
            self.vo_bank[:num_vo_real].uniform_(-bound, bound)
            self.vo_bank[num_vo_real:].zero_()

    def attn_qk_dim(self, layer: int) -> int:
        return self.head_dim if layer in WIDE_QK_LAYERS else NARROW_HEAD_DIM

    def attn_v_dim(self, layer: int) -> int:
        return NARROW_HEAD_DIM if layer in HALF_V_LAYERS else self.head_dim

    def init_mlp(self, model_dim):
        # MLP bank: stores c_fc and c_proj for all NUM_MLP_SLOTS slots.
        self.mlp_hdim = MLP_HIDDEN_DIM
        self.mlp_bank = nn.Parameter(torch.empty(NUM_MLP_SLOTS, 2, self.mlp_hdim, model_dim))  # (12, 2, 2816, 768)
        self.mlp_bank.reshape = (2 * NUM_MLP_SLOTS, self.mlp_hdim, model_dim)  # Shape for sharding: (24, 2816, 768)
        # The optimizer leaves these matrices untouched: c_fc and c_proj of each NO_MLP_LAYERS slot.
        self.mlp_bank.frozen_matrices = frozenset(2 * layer + j for layer in NO_MLP_LAYERS for j in (0, 1))

        # improved init scale by @YouJiacheng and @srashedll
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.mlp_bank[:, 0, :, :].uniform_(-bound, bound)  # c_fc
            self.mlp_bank[:, 1, :, :].zero_()  # c_proj - zero init suggested by @Grad62304977

    def init_misc(self, model_dim, num_layers):
        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)

        self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))

        # Per-sublayer residual scaling: [num_layers, 2] where [:,0]=attn, [:,1]=mlp
        # sqrt(1.1) per sublayer so cumulative per-layer scaling is 1.1
        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))

        pad = (-num_layers * 2 - 2) % self.world_size
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    -1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18
                    torch.ones(pad),
                ]
            )
        )

    def init_mudd(self, num_layers: int, model_dim: int):
        """
        Multiway Dynamic Dense Connections @lishengping. https://arxiv.org/abs/2502.12170
        Expressive and efficient mechanism for data dependent skip connections.
        Given current activation x, return n skip coefficients computed via ~mlp(x).
        Trimmed for speedrun: invoked at start of last layer and post-loop only.

        Start of last layer produces LAST_LAYER_MUDD_COEFS (14) coefficients:
          mu[0..2]  = v_mudd source coefs  (cache[0], cache[7], x)   -> added into V
          mu[3..5]  = residual source coefs (cache[0], cache[7], x)  -> residual recombination
          mu[6..7]  = per-pair ve_gate (2 channels, tiled to num_heads)
          mu[8..9]  = resid_attn / post_attn lambdas (dynamic)
          mu[10]    = x0 injection lambda (dynamic); mu[11] is vestigial (was the n-gram injection
                      lambda, dropped with the n-gram table; kept so the coefficient layout matches)
          mu[12..13]= resid_mlp / post_mlp lambdas (dynamic)

        Post-loop produces 5 residual coefs over
          {cache[0], cache[7], cache[9], ve_bank0, cache[3]}.
        """
        num_mudd_layers = 2
        self._mudd_scale = 0.1
        mudd_dim = 64
        max_num_coef = LAST_LAYER_MUDD_COEFS

        self.mudd_w1 = nn.Parameter(torch.empty(num_mudd_layers, mudd_dim, model_dim))
        for j in range(num_mudd_layers):
            nn.init.kaiming_uniform_(self.mudd_w1.data[j], a=math.sqrt(5))

        self.mudd_w2 = nn.Parameter(torch.zeros(num_mudd_layers, max_num_coef, mudd_dim))

        # Bias init in pre-scaled domain (effective = bias * _mudd_scale).
        bs_init = torch.zeros(num_mudd_layers, max_num_coef)
        # Per-pair ve_gate baseline (matches max of `2*sigmoid` used at other layers):
        bs_init[0, 6]  = 2.0 / self._mudd_scale       # ve_gate lane 0
        bs_init[0, 7]  = 2.0 / self._mudd_scale       # ve_gate lane 1
        # Layer-0 layer-10 dynamic lambdas (effective values match per-layer defaults):
        bs_init[0, 8]  = 1.1**0.5 / self._mudd_scale  # resid_attn[10]
        bs_init[0, 9]  = 1.0 / self._mudd_scale       # post_attn[10]
        bs_init[0, 10] = 0.0                          # x0_lambda[10] (init 0)
        bs_init[0, 11] = 0.05 / self._mudd_scale      # vestigial (was bigram_lambda[10])
        bs_init[0, 12] = 1.1**0.5 / self._mudd_scale  # resid_mlp[10]
        bs_init[0, 13] = 1.0 / self._mudd_scale       # post_mlp[10]
        # Layer-1 (post-loop): -0.5 backout absorbed into residual h7 coef.
        bs_init[1, 1]  = -0.5 / self._mudd_scale      # post-loop residual h7 coef
        self.mudd_b2 = nn.Parameter(bs_init)
        # Post-loop only: per-channel-group coefficient deltas, read off the same 64-dim hidden.
        # Zero-init, so step 0 is the plain per-token mix.
        self.mudd_w2g = nn.Parameter(torch.zeros(max_num_coef, MUDD_GROUPS, mudd_dim))

    def forward_mudd(self, x, id, num_coef):
        """Returns `num_coef` per-token MUDD coefficients from block `id` (0 or 1)."""
        x = F.gelu(F.linear(x, self.mudd_w1[id]))
        x = (F.linear(x, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        return x.split(1, dim=-1)

    def forward_mudd_grouped(self, x, id, num_coef):
        """forward_mudd's coefficients plus (B, T, num_coef, MUDD_GROUPS) per-channel-group deltas."""
        h = F.gelu(F.linear(x, self.mudd_w1[id]))
        y = (F.linear(h, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        deltas = torch.einsum("btd,kgd->btkg", h, self.mudd_w2g[:num_coef]) * self._mudd_scale
        return y.split(1, dim=-1), deltas

    def init_mudd_gate(self, model_dim: int):
        self._mudd_gate_scale = nn.Parameter(torch.tensor(MUDD_GATE_SCALE))
        mudd_gate_dim = 64
        H = MUDD_GATE_HEAD_LANES
        assert self.num_heads == H
        # Gate lane layouts, in the order unpack_pre_mudd_gate / unpack_post_mudd_gate read them:
        # pre:  xsa[1,3] 12 + attn[3] 6 + x0[0,1,2] 3 = 21
        # post: attn[10] 6 + x0[4,5,7] 3 + skip 1 = 10
        # (the bigram lanes went with the n-gram table).
        pre_attn_start = H * len(XSA_LAYERS)
        pre_x0_start = pre_attn_start + H * len(PRE_GATE_ATTN_GATE_LAYERS)
        self._mudd_gate_pre_num_coef = pre_x0_start + len(PRE_GATE_X0_LAYERS)
        post_x0_start = H * len(POST_GATE_ATTN_GATE_LAYERS)
        post_skip_lane = post_x0_start + len(POST_GATE_X0_LAYERS)
        self._mudd_gate_post_num_coef = post_skip_lane + 1
        max_num_coef = max(self._mudd_gate_pre_num_coef, self._mudd_gate_post_num_coef)
        self.mudd_gate_w1 = nn.Parameter(torch.empty(2, mudd_gate_dim, model_dim))
        self.mudd_gate_w2 = nn.Parameter(torch.zeros(2, max_num_coef, mudd_gate_dim))
        for j in range(2):
            nn.init.kaiming_uniform_(self.mudd_gate_w1.data[j], a=math.sqrt(5))

        # Bias init in the pre-scaled domain (effective = bias * MUDD_GATE_SCALE); XSA and x0 lanes start at 0.
        bs_init = torch.zeros(2, max_num_coef)
        attn_gate_bias = 0.25 / MUDD_GATE_SCALE
        skip_gate_bias = 0.5 / MUDD_GATE_SCALE
        bs_init[0, pre_attn_start:pre_x0_start].fill_(attn_gate_bias)
        bs_init[1, 0:post_x0_start].fill_(attn_gate_bias)
        bs_init[1, post_skip_lane].fill_(skip_gate_bias)
        self.mudd_gate_b2 = nn.Parameter(bs_init)

    def forward_mudd_gate(self, x, id, num_coef):
        x = F.gelu(F.linear(x, self.mudd_gate_w1[id]))
        return (F.linear(x, self.mudd_gate_w2[id, :num_coef]) + self.mudd_gate_b2[id, :num_coef]) * self._mudd_gate_scale.type_as(x)

    @staticmethod
    def _unpack_lanes(gate, start, layers, gates, width=1):
        """gates[layer] = the next `width`-lane slice of `gate` for each of `layers`; returns the next free lane."""
        for k, layer in enumerate(layers):
            lo = start + k * width
            gates[layer] = gate[..., lo:lo + width]
        return start + len(layers) * width

    def unpack_pre_mudd_gate(self, gate, xsa_alphas, attn_gates, x0_gates):
        lane = self._unpack_lanes(gate, 0, XSA_LAYERS, xsa_alphas, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_ATTN_GATE_LAYERS, attn_gates, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, PRE_GATE_X0_LAYERS, x0_gates)
        assert lane == self._mudd_gate_pre_num_coef

    def unpack_post_mudd_gate(self, gate, attn_gates, x0_gates):
        """Unpacks the post gate; returns the layer-6 skip gate."""
        lane = self._unpack_lanes(gate, 0, POST_GATE_ATTN_GATE_LAYERS, attn_gates, width=MUDD_GATE_HEAD_LANES)
        lane = self._unpack_lanes(gate, lane, POST_GATE_X0_LAYERS, x0_gates)
        assert lane + 1 == self._mudd_gate_post_num_coef
        return gate[..., lane:lane + 1]

    def _inject(self, i, x, x0, x0_gates):
        """Layer i's gated x0 injection into the residual stream (layer 0's is pre-loop)."""
        if i in X0_INJECT_LAYERS:
            x = x + x0 * x0_gates[i]
        return x

    def _mlp(self, x_normed, c_fc, c_proj):
        """relu(x @ c_fc.T)^2 @ c_proj (fp16 for both training and validation)."""
        return ReLUSqrdMLP(x_normed, c_fc, c_proj)

    def _attn_weights(self):
        """Per attention layer: (qk_w, v_w, o_w) from its bank slot, cut to the layer's head widths.

        qk_w is [2 * num_heads * qk_dim, dim] (Q heads, then K heads), v_w is [num_heads * v_dim, dim]
        and o_w is [dim, num_heads * v_dim] (nn.Linear layout). Each bank is unbound once, and the
        d_qk = 64 rows of all their slots are cut in one copy: per-layer indexing of a batched view
        would add a select_backward kernel per access (the same thing mlp_bank's unbind avoids).
        """
        H, head_dim, dim = self.num_heads, self.head_dim, self.qk_bank.shape[-1]
        num_slots = len(ATTN_BANK_ORDER)
        qk_heads = self.qk_bank[:self._num_qk_groups].view(num_slots, 2 * H, head_dim, dim)
        qk_full = qk_heads.flatten(1, 2).unbind(0)
        qk_narrow = qk_heads[:NUM_QK64_SLOTS, :, :NARROW_HEAD_DIM].reshape(
            NUM_QK64_SLOTS, 2 * H * NARROW_HEAD_DIM, dim).unbind(0)
        vo = self.vo_bank[:2 * num_slots].unbind(0)
        weights = {}
        for slot, layer in enumerate(ATTN_BANK_ORDER):
            qk_w = qk_full[slot] if layer in WIDE_QK_LAYERS else qk_narrow[slot]
            v_w, o_w = vo[2 * slot], vo[2 * slot + 1]
            v_dim = self.attn_v_dim(layer)
            if v_dim < head_dim:
                # Each head's leading v_dim V rows, and the O columns they feed (record #360).
                v_w = v_w.view(H, head_dim, dim)[:, :v_dim].reshape(H * v_dim, dim)
                o_w = o_w.view(dim, H, head_dim)[:, :, :v_dim].reshape(dim, H * v_dim)
            weights[layer] = (qk_w, v_w, o_w)
        return weights

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor,
                schedule_cfg: ForwardScheduleConfig):
        """Per-token loss for one packed varlen batch (B=1, documents separated by `seqlens`).

        input_seq may be shorter than target_seq by up to mtp_weights.numel() - 1 rows: the extra
        rows only exist so the multi-token-prediction terms at the batch tail have real targets
        (train_gpt.py extends them at the microbatch boundary; the loss ignores rows beyond
        input_seq's length).

        Layer topology (11 layers, 0-indexed):
          - attention on ATTN_LAYERS (0, 1, 2, 3, 5, 8, 10); short sliding window except layers 3 and 10
            (long window, partial key offset). Layer 6 adds a gated skip from layer 3 instead of attention;
            layers 4 and 9 run only their MLP; layer 7 only rescales and re-injects x0
          - head widths: query/key 128 on the long-window layers 3, 10 and 64 elsewhere; value/output
            64 on layers 1, 8 and 128 elsewhere
          - paired-head attention on layers 0, 2, 5; token value embeddings added to V on 1, 2, 8, 10
          - MUDD gates are computed from x0 (for layers 0-3) and at the start of layer 4 (for layers 4-10);
            the last layer and the post-loop mix use MUDD dense connections over cached layer outputs
        """
        assert input_seq.ndim == 1

        # ---- Schedule and layer topology ----
        mtp_weights = schedule_cfg.mtp_weights
        prefix_weight = schedule_cfg.prefix_weight
        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long
        # sliding-window sizes and key shift: the long windows get the partial key offset
        bm_sizes = [ws_long if i in LONG_WINDOW_LAYERS else ws_short for i in range(self.num_layers)]
        key_offset = [i in LONG_WINDOW_LAYERS for i in range(self.num_layers)]
        # Attention-segment boundaries, cut once here and shared by every layer (one host sync).
        ends = seqlens.tolist()
        segments = [(s, e) for s, e in zip(ends, ends[1:]) if e > s]
        # Per-forward SDPA bias cache; every layer of this forward shares it (model/attention.py).
        mask_cache: dict = {}

        # ---- Unbind parameters (avoid select_backward kernels) ----
        sa_lambdas = self.scalars[: 2 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[2 * self.num_layers]
        skip_lambda = self.scalars[2 * self.num_layers + 1]
        resid_lambdas_attn = self.resid_lambdas[:, 0].half().unbind(0)
        resid_lambdas_mlp  = self.resid_lambdas[:, 1].half().unbind(0)
        post_lambdas_attn = self.post_lambdas[:, 0].half().unbind(0)
        post_lambdas_mlp  = self.post_lambdas[:, 1].half().unbind(0)
        ve_gates = [None] * self.num_layers
        for layer, gate in zip(VALUE_EMBED_GATE_LAYERS, self.ve_gate_bank.unbind(0)):
            ve_gates[layer] = gate
        attn_gates = [None] * self.num_layers
        xsa_alphas = [None] * self.num_layers
        x0_gates = [None] * self.num_layers
        attn_weights = self._attn_weights()
        mlp_all = self.mlp_bank.flatten(0, 1).unbind(0)  # 24 tensors of [mlp_hdim, dim]
        mlp_fcs = mlp_all[0::2]    # even indices: c_fc
        mlp_projs = mlp_all[1::2]  # odd indices: c_proj

        # ---- Embeddings and input preparation ----
        x = self.embed(input_seq) # embed is synced from lm_head during tied phase by optimizer

        # Value embeddings - always computed (not precomputed)
        # Shifted .01 ... 234 structure on token value embeddings by @photomz
        # One plane per VALUE_EMBED_LAYERS entry, read at the input tokens by plain indexing, whose
        # backward lays a dense [num_ve * vocab, model_dim] gradient on value_embeds.
        ve_planes = self.value_embeds.view(len(VALUE_EMBED_LAYERS), self.vocab_size, -1)[:, input_seq]
        ve = [None] * self.num_layers
        for layer, plane in zip(VALUE_EMBED_LAYERS, ve_planes):
            ve[layer] = plane

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda.type_as(x) * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        pre_gate = self.forward_mudd_gate(x0, id=0, num_coef=self._mudd_gate_pre_num_coef)
        self.unpack_pre_mudd_gate(
            pre_gate,
            xsa_alphas,
            attn_gates,
            x0_gates,
        )

        # Initialize residual stream (the pre-loop n-gram injection is gone with the n-gram table).
        x = x0.clone()
        skip_gate_out = None
        post_skip_gate = None

        # cache[k] is the layer-k snapshot used downstream by MUDD.
        # cache[0] = the input to layer 0 (x0 itself now, with no pre-loop injection).
        cache = {0: x}
        # norm(cache[7]): every attention layer after layer 7 reads this same input, so it is normed
        # once and shared. The post-loop mix reads it too.
        late_attn_in = None
        for i in range(self.num_layers):
            c_fc = mlp_fcs[i]
            c_proj = mlp_projs[i]
            mu = None

            if i == POST_GATE_LAYER:
                post_gate = self.forward_mudd_gate(x, id=1, num_coef=self._mudd_gate_post_num_coef)
                post_skip_gate = self.unpack_post_mudd_gate(post_gate, attn_gates, x0_gates)

            # process attn. skip on layer 6 @YouJiacheng
            if i == 6:
                assert post_skip_gate is not None
                skip_gate_out = torch.sigmoid(skip_lambda).type_as(x) * post_skip_gate
                x = x + skip_gate_out * cache[3]
            elif i in NO_ATTN_LAYERS:
                # No attention sublayer: keep the residual scaling and the x0 injections.
                x = self._inject(i, scale(resid_lambdas_attn[i], x), x0, x0_gates)
            else:
                ve_gate_head = None  # norm(attn input)[..., :VALUE_EMBED_GATE_CHANNELS], on VALUE_EMBED_GATE_LAYERS
                if late_attn_in is not None:
                    attn_in_normed = late_attn_in
                else:
                    attn_in = cache.get(7, x)
                    if i in VALUE_EMBED_GATE_LAYERS:
                        attn_in_normed, ve_gate_head = rms_norm_with_head(attn_in, VALUE_EMBED_GATE_CHANNELS)
                    else:
                        attn_in_normed = norm(attn_in)
                    if 7 in cache:
                        late_attn_in = attn_in_normed
                B, T = attn_in_normed.size(0), attn_in_normed.size(1)

                if i == self.num_layers - 1:
                    cache[9] = x
                    mu = self.forward_mudd(x, id=0, num_coef=LAST_LAYER_MUDD_COEFS)
                    v_mudd = mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * x
                    v_mudd = v_mudd.view(B, T, self.num_heads, self.head_dim)
                    x = (1 + mu[5]) * x + mu[3] * cache[0] + mu[4] * cache[7]
                    ve_gate = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(
                        self.num_heads // 2, dim=-1
                    ).unsqueeze(-1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate * ve_view + v_mudd).view(B, T, -1)
                elif ve[i] is not None:
                    # gate pattern g(x[:6] + ve[:6]) by @photomz
                    gate_in = torch.cat([ve_gate_head, ve[i][None, ..., :VALUE_EMBED_GATE_CHANNELS]], dim=-1)
                    ve_gate_out = 2 * torch.sigmoid(F.linear(gate_in, ve_gates[i])).view(B, T, self.num_heads, 1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate_out * ve_view).view(B, T, -1)
                else:
                    aux_v = None

                if i in WIDE_QK_LAYERS:
                    yarn = self.yarn_wide
                elif i in PAIRED_HEAD_LAYERS:
                    yarn = self.yarn_paired_head
                else:
                    yarn = self.yarn
                attn_args = AttnArgs(
                    sa_lambdas=sa_lambdas[i],
                    segments=segments,
                    bm_size=bm_sizes[i],
                    yarn=yarn,
                    key_offset=key_offset[i],
                    attn_gate_w=attn_gates[i] if i in ATTN_GATE_LAYERS else None,
                    aux_v=aux_v,
                    xsa_alpha=xsa_alphas[i],
                    mask_cache=mask_cache,
                    # The post-lambda rides the output projection, except on the MUDD layer (mu[9] is per-token).
                    o_gain=post_lambdas_attn[i] if mu is None else None,
                )
                qk_w, v_w, o_w = attn_weights[i]
                attn_out = self.attn[str(i)](attn_in_normed, attn_args, qk_w, v_w, o_w)

                if mu is not None:
                    x = mu[8] * x + mu[9] * attn_out + mu[10] * cache[0]
                else:
                    x = scale(resid_lambdas_attn[i], x) + attn_out  # attn_out carries post_lambdas_attn[i]
                    x = self._inject(i, x, x0, x0_gates)

            # process mlp
            if i in NO_MLP_LAYERS:
                x = scale(resid_lambdas_mlp[i], x)
                if i in CACHE_LAYERS:
                    cache[i] = x
                continue
            mlp_in = norm(x)
            mlp_out = self._mlp(mlp_in, c_fc, c_proj)
            if mu is not None:
                x = mu[12] * x + mu[13] * mlp_out
            else:
                x = scale_add(resid_lambdas_mlp[i], x, post_lambdas_mlp[i], mlp_out)
            if i == PARALLEL_MLP_LAYER:
                # Same input and the same post-lambda as the layer's own MLP.
                k = PARALLEL_MLP_SLOT
                parallel_out = self._mlp(mlp_in, mlp_fcs[k], mlp_projs[k])
                x = x + scale(post_lambdas_mlp[i], parallel_out)

            if i in CACHE_LAYERS:
                cache[i] = x

        # Post-loop MUDD: mix 10 earlier activations back into the residual, each with a per-token
        # coefficient plus a per-channel-group delta. The last two are the final layer's MLP input and
        # its attention input, norm(cache[7]).
        assert late_attn_in is not None, "norm(cache[7]) is not bound at loop exit"
        sources = [
            cache[0], cache[7], cache[9], ve[1][None].to(dtype=x.dtype), cache[3],
            ve[2][None].to(dtype=x.dtype), ve[10][None].to(dtype=x.dtype), ve[8][None].to(dtype=x.dtype),
            mlp_in, late_attn_in,
        ]
        mu, deltas = self.forward_mudd_grouped(x, id=1, num_coef=len(sources))
        grouped = lambda t: t.unflatten(-1, (MUDD_GROUPS, -1))  # (B, T, D) -> (B, T, G, D/G)
        mixed = grouped(x)
        for k, src in enumerate(sources):
            mixed = mixed + (mu[k] + deltas[..., k, :]).unsqueeze(-1) * grouped(src)
        x = mixed.flatten(-2)

        return self._loss(norm(x), input_seq, target_seq, mtp_weights, prefix_weight, schedule_cfg.sampled_loss)

    def _loss(self, x, input_seq, target_seq, mtp_weights, prefix_weight, sampled_loss):
        """Per-token loss from the final normed hidden state.

        Training: softcapped CE (chunked over the class axis, track_1_short/losses.py) over next-token +
        multi-token + prefix-token targets, normalized over the vocabulary, or over `sampled_loss`'s
        candidate set early in training (sampled_softmax.py).
        Validation: plain next-token CE over the full vocabulary, with non-canonical tokens masked out
        (see canonical_mask.py); `sampled_loss` is ignored.

        target_seq may be longer than input_seq by the MTP lookahead; rows past input_seq's length
        only supply targets (the loss reads x's rows 0..n-1).
        """
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15
        # @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1). @classiclarryd updated to 23*sigmoid((logits+5)/7.5)
        if self.training and sampled_loss is not None:
            # Targets and prefix targets arrive as positions in the candidate set, built on the host;
            # target_pos carries the MTP lookahead rows (sliced by train_gpt.py).
            loss_per_token = softcapped_ce(
                x.view(-1, x.size(-1)), sampled_loss.rows.t(), self.lm_head.weight, mtp_weights,
                prefix_weight, sampled_loss.target_pos, sampled_loss.prefix_pos,
                sampled_loss.vocab_pos,
            )
        elif self.training:
            n = x.size(1)
            prefix_target_seq = self.prefix_table[target_seq[:n]]
            loss_per_token = softcapped_ce(
                x.view(-1, x.size(-1)), self.lm_head.weight, None, mtp_weights, prefix_weight,
                target_seq, prefix_target_seq, None,
            )
        else:
            # Every step is row-local, so slabs of EVAL_CE_SLAB_ROWS rows give the same per-token losses
            # as one pass; only the [rows, vocab] logit block shrinks.
            x = x.view(-1, x.size(-1))
            shifts = torch.arange(8, dtype=torch.uint8, device=x.device)
            slab_losses = []
            for lo in range(0, x.size(0), EVAL_CE_SLAB_ROWS):
                hi = min(lo + EVAL_CE_SLAB_ROWS, x.size(0))
                # The softcap runs in the input dtype (fp16 here, record #360 ran bf16) and upcasts after.
                logits = (23 * torch.sigmoid((self.lm_head(x[lo:hi]) + 5) / 7.5)).float()
                # Drop the tokens the tokenizer would never emit after input_seq. -60 is well below
                # the 0..23 the softcap leaves, so a dropped token contributes nothing to the softmax.
                dropped = (self.canon_mask[input_seq[lo:hi], :, None] >> shifts & 1).view(logits.shape).bool()
                logits = logits.masked_fill(dropped, -60.0)
                slab_losses.append(F.cross_entropy(logits, target_seq[lo:hi], reduction="none"))
            loss_per_token = torch.cat(slab_losses)
        return loss_per_token

    # -------------------------------------------------------------------------
    # Setup and run-level hooks main() calls on the uncompiled model.

    def cast_matrix_weights_fp16(self):
        """Matrix weights train in fp16 (Turing has no bf16; lm_head and value_embeds are created
        fp16); the scalar and lambda parameters stay fp32. Call once, before the optimizer is built."""
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.Linear)):
                m.weight.data = m.weight.data.half()
        for param in (self.ve_gate_bank, self.qk_bank, self.vo_bank, self.mlp_bank,
                      self.mudd_w1, self.mudd_w2, self.mudd_w2g, self.mudd_b2,
                      self.mudd_gate_w1, self.mudd_gate_w2, self.mudd_gate_b2):
            param.data = param.data.half()

    @property
    def yarns(self) -> tuple[Yarn, ...]:
        return (self.yarn, self.yarn_paired_head, self.yarn_wide)

    def limit_yarn_rebuild(self, rows: int):
        """From now on a window change rebuilds only the first `rows` rotary rows (the longest training
        sequence); complete_yarn_tables() fills in the rest before a validation (model/attention.py Yarn)."""
        for yarn in self.yarns:
            yarn.rebuild_rows = rows

    def complete_yarn_tables(self):
        for yarn in self.yarns:
            yarn.ensure_full()

    @property
    def lm_head_weight(self) -> Tensor:
        """The fp16 lm_head [model_dim, vocab]: the sampled-softmax row source."""
        return self.lm_head.weight
