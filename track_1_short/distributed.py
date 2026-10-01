"""Single-GPU runtime environment. The T4 port always runs as one process on one GPU."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DistEnv:
    rank: int
    world_size: int
    device: torch.device

    @property
    def master_process(self) -> bool:
        # the single process does logging, checkpointing etc.
        return self.rank == 0


def setup_distributed() -> DistEnv:
    """Assert CUDA and return the single-process environment (no process group)."""
    assert torch.cuda.is_available(), "the T4 port requires CUDA"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    return DistEnv(rank=0, world_size=1, device=device)
