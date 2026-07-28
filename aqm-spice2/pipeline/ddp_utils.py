import os
import torch
import torch.distributed as dist

def init_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = local_rank >= 0 and world_size > 1
    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    device = torch.device(f"cuda:{local_rank}" if is_ddp else ("cuda" if torch.cuda.is_available() else "cpu"))
    return local_rank, world_size, is_ddp, device

def is_main(local_rank):
    return local_rank <= 0

def cleanup(is_ddp):
    if is_ddp:
        dist.destroy_process_group()

def sync_barrier(is_ddp):
    if is_ddp:
        dist.barrier()
