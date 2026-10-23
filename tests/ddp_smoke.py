"""DDP smoke test: torchrun --nproc_per_node=2 tests/ddp_smoke.py

Each rank holds half of a batch. After backward, the DDP-averaged gradients must match a
single-process run on the full batch (same mean loss).
"""
import os

import torch
import torch.distributed as dist
from torch import nn

import swvr


class Model(nn.Module):
    def __init__(self, K=256, V=20000, D=16):
        super().__init__()
        self.trunk = nn.Linear(K, K)
        self.head = nn.Linear(K, V)
        self.embed = nn.Embedding(V, D)
        self.loss = swvr.SoftmaxReadoutMSE(self.head, self.embed, z_loss_weight=1e-3,
                                           chunk_size=4096)

    def forward(self, x, target):
        return self.loss(torch.tanh(self.trunk(x)), target)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    torch.manual_seed(0)
    model = Model().cuda()
    ref = Model().cuda()
    ref.load_state_dict(model.state_dict())
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(world * 512, 256, device="cuda", generator=g)
    t = torch.randn(world * 512, 16, device="cuda", generator=g)
    shard = slice(rank * 512, (rank + 1) * 512)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ddp(x[shard], t[shard]).backward()
        ref(x, t).backward()
    worst = 0.0
    for (name, p), q in zip(model.named_parameters(), ref.parameters()):
        err = ((p.grad - q.grad).norm() / q.grad.norm()).item()
        worst = max(worst, err)
        assert err < 2e-2, (name, err)
    if rank == 0:
        print(f"DDP smoke OK: world={world}, worst relative grad error {worst:.1e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
