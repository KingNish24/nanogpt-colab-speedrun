"""Training state that changes per step: schedule application and optimizer stepping."""
import copy
from collections.abc import Callable

import torch

from track_1_short.config import BLOCK_SIZE, VIRTUAL_SEQ_CAP
from track_1_short.model.gpt import ForwardScheduleConfig
from track_1_short.optim.anvil import RAIL_ENGAGE_STEP, RAIL_FAST_BETA, RAIL_FAST_WEIGHT, AnvilAndAdam, AnvilBank
from track_1_short.sampled_softmax import SampledLoss
from track_1_short.schedule import TrainingSchedule, get_rail_beta

# value_embeds' Adam (record #360): dense now (the n-gram cadence it once shared with the row-sparse
# table is gone), so it keeps these betas/wd_mul from param_table and updates on every Adam step.
VALUE_EMBED_LR_MUL = 70.0
VALUE_EMBED_BETAS = (0.75, 0.95)
VALUE_EMBED_WD_MUL = 5.0


class TrainingManager():
    """
    Manages the AnvilAndAdam for all parameters with explicit ordering.
        1. Scalars are given higher momentum terms to smooth learning @ChrisJMcCormick
        2. Adam optimizers are only stepped on odd steps @classiclarryd
        3. Explicit scatter_order and work_order for update scheduling (no backward hooks); world of
           one, so there is nothing to gather
        4. ANVIL's fast-rail beta has a linear warmup and cooldown (get_rail_beta); the twin-rail
           blend engages at RAIL_ENGAGE_STEP
        5. Learning rates follow a linear decay schedule
        6. Embed is tied to lm_head until split step (the extension stage), then untied @classiclarryd
        7. value_embeds is a normal dense Adam parameter (its row-sparse pull, shared cadence with the
           deleted n-gram table, are gone in the T4 port)
    """
    def __init__(self, model, schedule: TrainingSchedule, bank_update: Callable[[AnvilBank], None]):
        """bank_update: how the optimizer runs each ANVIL bank's update (anvil_bank_update)."""
        self.model = model
        self.schedule = schedule

        # - Ordering dictates when to launch reduce/reduce_scatter operations
        # - "sharded" parameters use reduce_scatter/all_gather and "replicated" ones use all_reduce
        # - lr_mul and wd_mul are per-parameter learning rate and weight decay multipliers
        self.param_table = {
            "qk_bank":        {"optim": "anvil",   "comms": "sharded",    "adam_betas": None},
            "vo_bank":        {"optim": "anvil",   "comms": "sharded",    "adam_betas": None},
            "mlp_bank":       {"optim": "anvil",   "comms": "sharded",    "adam_betas": None},
            "scalars":        {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 5.0,  "wd_mul": 0.0},
            "smear_gate":     {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99], "lr_mul": 0.01, "wd_mul": 0.0},
            "ve_gate_bank":   {"optim": "adam",    "comms": "replicated", "adam_betas": [0.9,  0.99]},
            "lm_head":        {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
            "post_lambdas":   {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
            "resid_lambdas":  {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 5.0,  "wd_mul": 0.0},
            "value_embeds":   {"optim": "adam",    "comms": "sharded",    "adam_betas": list(VALUE_EMBED_BETAS), "lr_mul": VALUE_EMBED_LR_MUL, "wd_mul": VALUE_EMBED_WD_MUL},
            "embed":          {"optim": "adam",    "comms": "sharded",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
        }

        # ---- MUDD parameter overrides ----
        self.param_table.update({
            "mudd_w1":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
            "mudd_w2":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
            "mudd_w2g":   {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
            "mudd_b2":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25, "wd_mul": 0.0},
            "mudd_gate_w1": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1},
            "mudd_gate_w2": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1},
            "mudd_gate_b2": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1, "wd_mul": 0.0},
            "_mudd_gate_scale": {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.1, "wd_mul": 0.0},
        })

        # scatter_order is the gradient-landing order (record #360); world of one, each "reduce" is a
        # copy into the bank/state buffer (optim/anvil.py). The replicated params' flat copy happens
        # ahead of all of these (perf/replicated_adam.py); their entries here only name them.
        self.scatter_order = [
            "qk_bank", "vo_bank", "mlp_bank",
            "value_embeds",
            "lm_head", "embed",
            "scalars", "smear_gate", "ve_gate_bank", "post_lambdas", "resid_lambdas",
            "mudd_w1", "mudd_w2", "mudd_w2g", "mudd_b2", "mudd_gate_w1", "mudd_gate_w2", "mudd_gate_b2",
            "_mudd_gate_scale",
        ]
        # work_order is the update schedule (record #360):
        # - qk_bank and vo_bank first, lm_head before embed sync (when tied)
        # - value_embeds early (its dense gradient must land before mlp_bank runs)
        # - the replicated params (one fused update at the first of them), then mlp_bank, the largest
        #   bank, last
        self.work_order = [
            "qk_bank", "vo_bank", "lm_head", "embed", "value_embeds",
            "scalars", "smear_gate", "ve_gate_bank", "mudd_b2", "mudd_gate_b2", "_mudd_gate_scale",
            "post_lambdas", "resid_lambdas", "mudd_w2", "mudd_w2g", "mudd_gate_w2", "mudd_w1", "mudd_gate_w1",
            "mlp_bank",
        ]

        self.adam_defaults = adam_defaults = dict(
            lr=0.008,
            eps=1e-10,
            weight_decay=0.005,
        )

        anvil_defaults = dict(
            lr=0.023,
            momentum=0.95,      # Nesterov lookahead; get_rail_beta rewrites it every step
            beta2=0.9,          # lane-energy EMA decay for the equalizer
            weight_decay=2.25,
        )

        self.optimizer = AnvilAndAdam(
            model.named_parameters(),
            param_table=self.param_table,
            scatter_order=self.scatter_order,
            work_order=self.work_order,
            adam_defaults=adam_defaults,
            anvil_defaults=anvil_defaults,
            bank_update=bank_update,
        )

        # Split embed from lm_head at the extension stage (on an odd step so Adam updates)
        self.split_step = self.schedule.split_step

        self.reset()

    def apply_final_ws_ext(self):
        self.ws_long = self.schedule.ws_post_yarn_ext

    def get_forward_args(self, sampled_loss: SampledLoss | None = None):
        return ForwardScheduleConfig(
            mtp_weights = self.mtp_weights,
            prefix_weight = self.prefix_weight,
            ws_short = self.ws_short * BLOCK_SIZE,
            ws_long = self.ws_long * BLOCK_SIZE,
            # The loader cuts documents longer than VIRTUAL_SEQ_CAP into attention segments.
            train_max_seq_len = min(self.train_max_seq_len, VIRTUAL_SEQ_CAP),
            sampled_loss = sampled_loss,
        )

    def is_adam_step(self, step: int):
        """Adam params are only updated on odd steps."""
        return step % 2 == 1

    def get_transition_steps(self):
        return [start for start, _ in self.schedule.boundaries[1:]]

    def advance_schedule(self, step: int):
        stage, _ = self.schedule.lookup(step)
        old_ws_short = self.ws_short
        self.ws_short, new_ws_long = stage.window_sizes
        if new_ws_long != self.ws_long:
            # Each rotary table follows the window its layers attend over (record #360; the window
            # sizes only change together, at the ws_long changes).
            self.model.yarn_wide.apply(self.ws_long * BLOCK_SIZE, new_ws_long * BLOCK_SIZE)
            for yarn in (self.model.yarn, self.model.yarn_paired_head):
                yarn.apply(old_ws_short * BLOCK_SIZE, self.ws_short * BLOCK_SIZE)

        self.train_max_seq_len = stage.train_max_seq_len
        self.ws_long = new_ws_long
        self.mtp_weights = self.schedule.mtp_weights[step]
        self.prefix_weight = self.schedule.prefix_weights[step]

    def step_optimizers(self, step: int):
        """The optimizer step."""
        step_lr = self.schedule.get_lr(step)
        rail_beta = get_rail_beta(step, self.schedule.total_steps)
        # Before the engage step ANVIL reads the fast rail alone, on the scheduled beta.
        if step >= RAIL_ENGAGE_STEP:
            self.optimizer.set_rails(fast_beta=RAIL_FAST_BETA, fast_weight=RAIL_FAST_WEIGHT)
        else:
            self.optimizer.set_rails(fast_beta=rail_beta, fast_weight=1.0)
        do_adam = self.is_adam_step(step)

        # Update learning rates and momentum for all params
        for param, p_cfg in self.optimizer.param_cfgs.items():
            p_cfg.lr = p_cfg.initial_lr * step_lr
            if p_cfg.optim == "anvil":
                p_cfg.momentum = rail_beta

        self.optimizer.step(do_adam=do_adam)

        # At split step: copy lm_head optimizer state to embed and mark as split
        if step == self.split_step:
            self.optimizer.copy_lm_state_to_embed()

    def reset(self, state=None):
        if state is not None:
            self.optimizer.load_state_dict(state)

        # Reset ANVIL velocity/lane state and split_embed state
        self.optimizer.reset()

        stage, _ = self.schedule.lookup(0)
        self.ws_short, self.ws_long = stage.window_sizes
        self.train_max_seq_len = stage.train_max_seq_len
        for yarn in (self.model.yarn, self.model.yarn_paired_head, self.model.yarn_wide):
            yarn.reset()

    def get_state(self):
        return copy.deepcopy(self.optimizer.state_dict())
