#!/usr/bin/env python3
"""
guava_worker.py

Run this on EACH GPU machine (Windows desktop with GPUs, Linux GPU box, etc.).

This process:
- Creates a DistributedConfig that matches what orchestrator expects
- Builds the SAME model architecture
- Instantiates NetworkWorker(gpu_id=..., config=..., model_ctor=...)
- Connects to orchestrator at --master-ip/--master-port
- Blocks in connect_and_train(), which:
    * registers with orchestrator
    * waits for commands
    * runs actual forward/backward on the GPU
    * ships gradients / metrics back

Example for one GPU box with GPU 0:
    python guava_worker.py --gpu-id 0 --master-ip 192.168.0.177 --master-port 29500 --world-size 2
"""

import argparse
import torch
import torch.nn as nn

from guava.config import DistributedConfig
from guava.network_worker import NetworkWorker


# MUST MATCH orchestrator_train.py TinyToyModel EXACTLY
class TinyToyModel(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.linear(x)
        return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gpu-id",
        type=int,
        required=True,
        help="Local CUDA device index for THIS worker (0,1,2,...)",
    )
    ap.add_argument(
        "--master-ip",
        required=True,
        help="IP address where orchestrator is listening",
    )
    ap.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Base port of orchestrator (orchestrator also uses +1..+7)",
    )
    ap.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="Total number of workers (must match orchestrator --gpus)",
    )
    args = ap.parse_args()

    # ---------------------------
    # Build a DistributedConfig that mirrors orchestrator
    # ---------------------------
    cfg = DistributedConfig()
    cfg.master_ip = args.master_ip
    cfg.master_port = args.master_port
    cfg.num_workers = args.world_size

    # Strategy: we're doing data parallel right now
    cfg.data_parallel = True
    cfg.model_parallel = False
    cfg.pipeline_parallel = False
    cfg.tensor_parallel = False

    # Hyperparams must match what orchestrator used
    cfg.batch_size = 2
    cfg.vocab_size = 100
    cfg.d_model = 32
    cfg.n_layers = 2
    cfg.n_heads = 4

    # Optim / clip etc. from your default config
    # (DistributedConfig already has learning_rate, weight_decay, max_grad_norm, etc.)

    # ---------------------------
    # Model factory for this worker
    # NetworkWorker will call this to build a fresh model instance
    # and then it will slice (model_parallel) or keep whole (data_parallel)
    # ---------------------------
    def build_model() -> nn.Module:
        return TinyToyModel(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
        )

    # ---------------------------
    # Instantiate runtime
    # ---------------------------
    # NOTE: NetworkWorker expects:
    #   gpu_id, config, model_ctor, master_ip, master_port
    # and internally chooses DataParallelWorker or ModelShardWorker.
    worker_runtime = NetworkWorker(
        gpu_id=args.gpu_id,
        config=cfg,
        model_ctor=build_model,
        master_ip=args.master_ip,
        master_port=args.master_port,
    )

    # ---------------------------
    # Enter training loop
    # This will:
    #   - connect to orchestrator ctrl socket
    #   - announce GPU ID, layers, etc.
    #   - sit in a loop waiting for 'train_data_parallel' commands, etc.
    # ---------------------------
    print("======================================================================")
    print("GUAVA WORKER START")
    print("======================================================================")
    print(f"gpu_id        : {args.gpu_id}")
    print(f"master_ip     : {args.master_ip}")
    print(f"master_port   : {args.master_port}")
    print(f"world_size    : {args.world_size}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"using device  : cuda:{args.gpu_id} -> {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        print("WARNING: CUDA not available. Will run on CPU.")

    print("======================================================================")
    print("connecting to orchestrator and starting training loop...")
    print("======================================================================")

    worker_runtime.connect_and_train()

    print("======================================================================")
    print("worker finished / shutdown")
    print("======================================================================")


if __name__ == "__main__":
    main()
