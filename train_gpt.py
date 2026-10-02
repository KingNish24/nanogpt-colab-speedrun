"""NanoGPT speedrun, track 1 (T4 port): train a GPT-2 small-scale model toward ≤3.5 canonical
FineWeb val loss on a single Colab T4 GPU (record #360 reached 3.28 on 8xH100; this port relaxes
the target and shrinks the data to fit one 16 GB GPU).

Launch: python train_gpt.py   (plain process; the port dropped torchrun/torch.distributed)

This file is the outline of the whole run. The model, optimizer, data and schedules live in the
`track_1_short/` package; `track_1_short/perf/` holds the kernels and precision tricks, and can be
skipped when reading for the algorithm.

T4-port differences from record #360's 8xH100 driver:
  - one process, one GPU: no dist broadcasts/reduces, no CUDA graphs, no torch.compile;
  - each step's global batch is split into MICRO_BATCH_TOKENS-token microbatches whose gradients
    accumulate into one optimizer step under a static fp16 LOSS_SCALE (halved if a loss ever goes
    non-finite: the step is skipped and its grads dropped);
  - the sampled-softmax candidate set is built per microbatch (its targets), not per rank-step;
  - validation runs in EVAL_CHUNK_TOKENS-token forwards so the CE slabs and SDPA bias caches fit
    16 GB.
"""
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from track_1_short.run_log import log_environment, read_source, start_run_log

# Read the source ASAP, for the run log.
code = read_source(sys.argv[0])

import copy
import gc
import time

import torch
from torch import nn

from track_1_short.canonical_mask import BackgroundCanonicalMask
from track_1_short.config import (
    EVAL_CHUNK_TOKENS,
    LOSS_SCALE,
    LR_COOLDOWN_FRAC,
    MICRO_BATCH_TOKENS,
    SPLIT_EMBED_STAGE,
    TRAINING_STAGES,
    VIRTUAL_SEQ_CAP,
    WS_POST_YARN_EXT,
    Hyperparameters,
)
from track_1_short.data import ScheduledBatches, cu_seqlens_rows, distributed_data_generator
from track_1_short.distributed import setup_distributed
from track_1_short.model.gpt import ATTN_BANK_ORDER, GPT
from track_1_short.model.prefix_prediction import build_prefix_table_bucket
from track_1_short.optim.anvil import anvil_bank_update
from track_1_short.perf.pinned_batches import PinnedBatchStaging
from track_1_short.sampled_softmax import SampledLoss, SampledSoftmax
from track_1_short.schedule import TrainingSchedule
from track_1_short.tail_average import TailAverages
from track_1_short.training import TrainingManager

# Step lines in the timed loop (console and log): every Nth step (override: PRINT_EVERY=25), plus
# the last two (record #360).
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "5"))

# Per-phase timers, cumulative seconds since the clock started (option-1 instrumentation):
# data = batch wait inside batches.peek; cand = candidate upload/gather host time; fwd/bwd = CUDA-event
# GPU spans; opt = optimizer step (ends with a sync); other = the rest of train_time (Python enqueue,
# one-time builds). Reset at clock start, printed on every step line.
PHASE = {"data": 0.0, "cand": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0}


def _collect_gpu_phases(pairs) -> None:
    """Fold this step's fwd/bwd CUDA-event spans into PHASE (call after a synchronize)."""
    for start, mid, end in pairs:
        PHASE["fwd"] += start.elapsed_time(mid) / 1000.0
        PHASE["bwd"] += mid.elapsed_time(end) / 1000.0


def slice_seqlens(ends: list[int], a: int, n: int, cap: int | None = None) -> torch.Tensor:
    """The attention-segment boundaries of rows [a, a + n) as a cumulative list [0, ..., n].

    `ends` is a batch's cum_seqlens tolist() (leading 0, then each segment's start = the previous
    segment's end, padded with the batch length). Boundaries are cut at both slice edges, and the
    slice's own end always closes a segment, so attention never crosses a microbatch/chunk edge.
    `cap` additionally cuts any segment longer than it: training batches arrive pre-capped at
    VIRTUAL_SEQ_CAP (data.py), the unaligned validation stream does not, and a dense SDPA bias for
    an uncapped segment would be [L, L] -- unbounded on the T4. Capping only splits attention (the
    tokens are untouched), the same trick the loader applies to long training documents.
    """
    stop = a + n
    out = [0]

    def push(pos: int):
        if pos <= out[-1]:
            return
        if cap:
            while out[-1] + cap < pos:
                out.append(out[-1] + cap)
        out.append(pos)

    for e in ends:
        if e <= a:
            continue
        if e >= stop:
            break
        push(e - a)
    push(n)
    return torch.tensor(out, dtype=torch.int32)


def train_step(training_manager, model: nn.Module, sampled_softmax: SampledSoftmax,
               batches: ScheduledBatches, step: int, loss_scale: float) -> tuple[float, bool, float]:
    """One optimizer step over the step's microbatches, accumulated under the static loss scale.

    Returns (batch mean loss, skipped, loss_scale). The candidate set (early stages) is built and
    gathered per microbatch, in the race order upload -> gather -> forward -> backward -> read-done,
    so a microbatch's loss never aliases the next one's upload (perf/sampled_softmax_overlap.py).
    Non-finite losses (fp16 overflow spikes) skip the whole step: its gradients are dropped and the
    scale halved.
    """
    t = time.perf_counter()
    batch = batches.peek(step)
    PHASE["data"] += time.perf_counter() - t
    inputs, targets, cpu_targets = batch.inputs, batch.targets, batch.targets_cpu
    total_tokens = inputs.shape[0]
    assert total_tokens % MICRO_BATCH_TOKENS == 0, \
        f"stage batch {total_tokens} is not a whole number of {MICRO_BATCH_TOKENS}-token microbatches"
    n_micro = total_tokens // MICRO_BATCH_TOKENS
    ends = batch.cum_seqlens.tolist()
    # MTP lookahead: target rows beyond the microbatch's inputs so its tail rows have real targets.
    mtp_weights = training_manager.mtp_weights
    lookahead = mtp_weights.numel() - 1
    sampled = sampled_softmax.counts[step] > 0

    loss_sum = torch.zeros((), dtype=torch.float32, device=inputs.device)
    finite = torch.ones((), dtype=torch.bool, device=inputs.device)
    fwd_bwd_events = []
    for m in range(n_micro):
        a = m * MICRO_BATCH_TOKENS
        n = MICRO_BATCH_TOKENS
        ext = min(a + n + lookahead, total_tokens)
        sampled_loss = None
        if sampled:
            # Race order per microbatch: upload -> gather -> forward -> backward -> read-done
            # (perf/sampled_softmax_overlap.py). upload takes this microbatch's build (prefetched at
            # the previous iteration's start, so the host work hides under forward/backward; the
            # first microbatch of a step builds inline, ~1 ms).
            t = time.perf_counter()
            sampled_softmax.upload(step, cpu_targets[a:ext])
            sl = sampled_softmax.gather(step, model.lm_head_weight)
            # The loss reads target_pos[0 : n + K - 1] (MTP) and prefix_pos[0 : n]; the shared
            # rows/rows_t/vocab_pos stay whole.
            sampled_loss = SampledLoss(rows=sl.rows, rows_t=sl.rows_t,
                                       target_pos=sl.target_pos[:ext], prefix_pos=sl.prefix_pos[:n],
                                       vocab_pos=sl.vocab_pos)
            if m + 1 < n_micro:
                a2, ext2 = a + n, min(a + 2 * n + lookahead, total_tokens)
                sampled_softmax.prefetch(step, cpu_targets[a2:ext2])
            PHASE["cand"] += time.perf_counter() - t
        ev0, ev1, ev2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        ev0.record()
        loss = model(
            input_seq=inputs[a:a + n],
            target_seq=targets[a:ext],
            seqlens=slice_seqlens(ends, a, n),
            schedule_cfg=training_manager.get_forward_args(sampled_loss),
        )  # [n] fp32 per-token loss
        ev1.record()
        finite = finite & torch.isfinite(loss).all()
        loss_sum = loss_sum + loss.detach().sum()
        # One optimizer step's worth of the batch mean, scaled for fp16 gradients.
        (loss.mean() * (loss_scale / n_micro)).backward()
        ev2.record()
        fwd_bwd_events.append((ev0, ev1, ev2))
        sampled_softmax.mark_readers_done()
    batches.take(step)

    if not bool(finite):
        torch.cuda.synchronize()
        _collect_gpu_phases(fwd_bwd_events)
        model.zero_grad(set_to_none=True)
        return 0.0, True, loss_scale * 0.5
    t = time.perf_counter()
    training_manager.step_optimizers(step)
    torch.cuda.synchronize()
    PHASE["opt"] += time.perf_counter() - t
    _collect_gpu_phases(fwd_bwd_events)
    return float(loss_sum / total_tokens), False, loss_scale


@torch.no_grad()
def evaluate(model: nn.Module, training_manager, val_batches) -> float:
    """Mean full-vocabulary canonical CE over the validation batches, in EVAL_CHUNK_TOKENS forwards.

    Chunked for memory (logit slabs, SDPA bias caches); each chunk cuts the segments at its own
    edges, the same way record #360's per-rank val slices cut at rank boundaries. Validation
    segments are additionally capped at VIRTUAL_SEQ_CAP (see slice_seqlens): the unaligned val
    stream has no length cap, and an uncapped segment would need an unbounded [L, L] bias.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in val_batches:
        inputs, targets = batch.inputs, batch.targets
        ends = batch.cum_seqlens.tolist()
        for a in range(0, inputs.shape[0], EVAL_CHUNK_TOKENS):
            n = min(EVAL_CHUNK_TOKENS, inputs.shape[0] - a)
            losses = model(
                input_seq=inputs[a:a + n],
                target_seq=targets[a:a + n],
                seqlens=slice_seqlens(ends, a, n, cap=VIRTUAL_SEQ_CAP),
                schedule_cfg=training_manager.get_forward_args(),
            )
            total_loss += float(losses.sum())
            total_tokens += n
    return total_loss / total_tokens


def main():
    args = Hyperparameters()
    env = setup_distributed()
    if args.train_seed is not None:
        torch.manual_seed(args.train_seed)

    # Pinned slots for every batch's H2D copies, training and validation (perf/pinned_batches.py).
    batch_tokens = [s.batch_size for s in TRAINING_STAGES] + [args.val_batch_size]
    batch_staging = PinnedBatchStaging(max(batch_tokens), max(map(cu_seqlens_rows, batch_tokens)), env.device)

    def train_loader():
        return distributed_data_generator(
            args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, batch_staging,
        )

    def val_loader():
        return distributed_data_generator(args.val_files, args.val_batch_size, -1, batch_staging, align_to_bos=False)

    training_schedule = TrainingSchedule(
        TRAINING_STAGES, args.num_scheduled_iterations, args.num_extension_iterations, device=env.device,
        cooldown_frac=LR_COOLDOWN_FRAC, split_embed_stage=SPLIT_EMBED_STAGE, ws_post_yarn_ext=WS_POST_YARN_EXT,
    )

    print0, flush_log = start_run_log(env.master_process, args.run_id)
    log_environment(print0, code)
    flush_log()

    ########################################
    #          Model and optimizer         #
    ########################################
    # max_seq_len bounds the rotary factor tables: the longest forward is an eval chunk (train
    # microbatches are shorter; rotary rows are indexed by absolute position within a forward).
    model: nn.Module = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=max(MICRO_BATCH_TOKENS, EVAL_CHUNK_TOKENS),
        world_size=env.world_size,
        device=env.device,
    ).cuda()
    attn_widths = {layer: (model.attn_qk_dim(layer), model.attn_v_dim(layer)) for layer in ATTN_BANK_ORDER}
    print0(
        f"attention heads={model.num_heads} (qk, v) head widths by layer, in bank order={attn_widths} "
        f"seed={'random' if args.train_seed is None else args.train_seed} "
        f"steps={training_schedule.total_steps} stage boundaries={training_schedule.boundaries}"
    )
    model.cast_matrix_weights_fp16()

    # Early steps train on a sampled softmax (see track_1_short/sampled_softmax.py).
    sampled_softmax = SampledSoftmax(
        training_schedule, model.vocab_size, model_dim=model.lm_head.in_features,
        max_rows=MICRO_BATCH_TOKENS + 8, rank=env.rank, world_size=env.world_size, device=env.device,
    )
    print0(f"sampled softmax: P by step change {[(s, sampled_softmax.counts[s]) for s in range(0, training_schedule.total_steps) if sampled_softmax.counts[s] != sampled_softmax.counts[s - 1 if s else 0]]}")

    training_manager = TrainingManager(model, training_schedule, bank_update=anvil_bank_update)
    # value_embeds writes on every Adam step, so its tail-avg ticks on Adam steps only (the cadence
    # condition inside TailAverages picks the every-4th-step subset).
    tail_averages = TailAverages(training_manager.optimizer, env.rank,
                                 value_embed_updates=training_manager.is_adam_step,
                                 total_steps=training_schedule.total_steps)

    ########################################
    #            Warmup kernels            #
    ########################################
    print0("Warming up kernels on steps 0-1 (Triton JIT, SDPA, the fused optimizer), then resetting state",
           console=True)
    # Warm the kernels on the real paths, then restore the initial state so we are not cheating.
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                         optimizer=training_manager.get_state())
    warmup_batches = ScheduledBatches(train_loader(), training_schedule, steps=[0, 1])
    for step in (0, 1):
        training_manager.advance_schedule(step)
        train_step(training_manager, model, sampled_softmax, warmup_batches, step, LOSS_SCALE)
    print0("Resetting Model", console=True)
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    training_manager.reset(initial_state["optimizer"])
    sampled_softmax.reset()
    del warmup_batches, initial_state
    model.train()
    print0(f"Memory after warmup: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
           f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved", console=True)

    ########################################
    #        Training and validation       #
    ########################################
    batches = ScheduledBatches(train_loader(), training_schedule, steps=range(training_schedule.total_steps))

    # The canonical mask is only needed by the final validation, so it builds in a child process
    # while we train, and model.canon_mask stays all-zero == no masking until then, which is what
    # the intermediate validations run with. Only the buffer is allocated here; the build is started
    # below the clock, so its whole cost -- not just its use -- lands in the timed region.
    canon_mask_builder = BackgroundCanonicalMask(model.vocab_size, owner=env.master_process, print0=print0)

    # No cyclic GC inside the timed loop (record #360): what survives until now is frozen out of all
    # future scans; the garbage the loop makes is collected at each validation, clock stopped.
    gc.collect()
    gc.freeze()
    gc.disable()

    # From here a window change rebuilds only the rotary rows a training step reads; each validation
    # completes the tables first (model/attention.py Yarn; record #360). The warmup rebuilt whole.
    model.limit_yarn_rebuild(MICRO_BATCH_TOKENS)

    training_time_ms = 0
    for k in PHASE:
        PHASE[k] = 0.0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    canon_mask_builder.start()
    # Prefix-token table build, inside the timed region. The tokenizer was loaded at import
    # (get_encoding is cached in tiktoken's registry), so this pays only the table construction.
    model.prefix_table.copy_(build_prefix_table_bucket(model.vocab_size, bucket=env.rank, num_buckets=env.world_size))
    # The candidate build maps prefix targets on the host.
    sampled_softmax.set_prefix_table(model.prefix_table.cpu().numpy())
    # begin training
    loss_scale = LOSS_SCALE
    for step in range(training_schedule.total_steps + 1):
        last_step = (step == training_schedule.total_steps)
        training_manager.advance_schedule(step)
        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            if last_step:
                training_manager.apply_final_ws_ext()
                # Both on the clock: the wait in case the build is somehow not done, and the copy
                # of the result because it is part of the mask's cost.
                canon_mask_builder.wait()
                canon_mask_builder.collect(model.canon_mask)
                # On the clock: evaluate (and keep) the tail-averaged weights, not the final iterate.
                tail_averages.ship()
            # stop the clock: from here to the restart is validation, timed as val_time (the yarn
            # table completion and the val batch loads are validation's cost, not training's).
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            val_t0 = time.perf_counter()
            model.complete_yarn_tables()
            assert args.val_tokens % args.val_batch_size == 0
            val_steps = args.val_tokens // args.val_batch_size
            val_loader_iter = val_loader()
            val_batches = [next(val_loader_iter) for _ in range(val_steps)]
            del val_loader_iter
            val_loss = evaluate(model, training_manager, val_batches)
            del val_batches
            val_ms = 1000 * (time.perf_counter() - val_t0)
            print0(f"step:{step}/{training_schedule.total_steps} val_loss:{val_loss:.4f} val_time:{val_ms:.0f}ms "
                   f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
            # The clock is stopped: flush the log and collect the loop's garbage here.
            flush_log()
            gc.collect()
            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if env.master_process and args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizer=training_manager.get_state())
                os.makedirs(f"logs/{args.run_id}", exist_ok=True)
                torch.save(log, f"logs/{args.run_id}/state_step{step:06d}.pt")
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        batch_loss, skipped, loss_scale = train_step(
            training_manager, model, sampled_softmax, batches, step, loss_scale,
        )
        if skipped:
            print0(f"step:{step} non-finite loss: step skipped, loss scale -> {loss_scale}", console=True)
        tail_averages.tick(step)

        # logging, thinned to every PRINT_EVERY steps and the last two
        if (step + 1) % PRINT_EVERY == 0 or step + 1 >= training_schedule.total_steps - 1:
            approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
            loss_str = f" loss:{batch_loss:.4f}" if not skipped else ""
            phase_str = " ".join(f"{k}:{v:.1f}s" for k, v in PHASE.items())
            other = max(0.0, approx_training_time_ms / 1000 - sum(PHASE.values()))
            print0(f"step:{step+1}/{training_schedule.total_steps} train_time:{approx_training_time_ms:.0f}ms "
                   f"step_avg:{approx_training_time_ms/(step + 1):.2f}ms{loss_str} "
                   f"| {phase_str} other:{other:.1f}s", console=True)

    gc.enable()
    if args.run_evals:
        from evals import hellaswag
        hellaswag.evaluate(model=model, schedule_cfg=training_manager.get_forward_args(),
                           seq_len=EVAL_CHUNK_TOKENS, print0=print0)

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)


if __name__ == "__main__":
    main()
