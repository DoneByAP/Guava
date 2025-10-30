#!/usr/bin/env python3
"""
guava_worker.py

RUN THIS ON THE GPU MACHINE ONE TIME.

This single process will:
- Spawn 1 NetworkWorker per GPU you specify (e.g. GPU 0 AND GPU 1)
- Connect each worker to the orchestrator
- Keep them all running in parallel threads in this same terminal

USAGE EXAMPLE (2 GPUs on one box):
    python guava_worker.py \
        --gpu-ids 0,1 \
        --master-ip 192.168.0.177 \
        --master-port 29500 \
        --world-size 2

ARG NOTES:
- --gpu-ids      Comma-separated list of local CUDA device indices to use.
                 Example: "0" or "0,1" or "0,1,2"
- --world-size   Total number of workers the orchestrator expects across ALL machines.
                 If this is one machine with 2 GPUs and no other machines, use 2.
                 (Must match orchestrator --gpus)

WHAT HAPPENS:
- We build a DistributedConfig that mirrors orchestrator_train.py
- We define TinyToyModel (must match orchestrator EXACTLY)
- For each gpu in --gpu-ids:
    * We create a NetworkWorker(gpu_id=that_gpu,...)
    * We launch connect_and_train() in a thread
- We wait on all threads so training stays alive in this one terminal
"""

import argparse
import threading
from typing import List, Callable

import torch
import torch.nn as nn

from guava.config import DistributedConfig
from guava.network_worker import NetworkWorker


# ======================================================================
# MODEL DEFINITION (MUST MATCH orchestrator_train.py TinyToyModel EXACTLY)
# ======================================================================

class TinyToyModel(nn.Module):
    """Minimal token -> embedding -> linear logits model, same as orchestrator."""
    def __init__(self, vocab_size: int = 100, d_model: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [batch, seq] int64 token ids
        return:   [batch, seq, vocab] unnormalized logits
        """
        x = self.embedding(input_ids)  # [B,T,d_model]
        x = self.linear(x)             # [B,T,vocab]
        return x


# ======================================================================
# SMALL HELPERS
# ======================================================================

def parse_gpu_list(gpu_str: str) -> List[int]:
    """
    Convert something like "0,1,3" -> [0,1,3].
    Also validates that each id is >=0.
    """
    items = [s.strip() for s in gpu_str.split(",") if s.strip() != ""]
    gpu_ids = [int(x) for x in items]
    for g in gpu_ids:
        if g < 0:
            raise ValueError(f"Invalid GPU id {g}")
    return gpu_ids


def build_model_factory(cfg: DistributedConfig) -> Callable[[], nn.Module]:
    """
    Returns a 0-arg function that NetworkWorker can call
    to build a *fresh* model instance on that GPU.
    """
    def ctor() -> nn.Module:
        return TinyToyModel(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
        )
    return ctor


def launch_worker_thread(
    gpu_id: int,
    cfg: DistributedConfig,
    master_ip: str,
    master_port: int,
    model_ctor: Callable[[], nn.Module],
) -> threading.Thread:
    """
    Spin up one NetworkWorker bound to `gpu_id` and return
    a Thread that runs its training loop forever.

    We DO NOT rename NetworkWorker. We just wrap it.
    """
    def worker_main():
        # Create the runtime for this GPU
        runtime = NetworkWorker(
            gpu_id=gpu_id,
            config=cfg,
            model_ctor=model_ctor,
            master_ip=master_ip,
            master_port=master_port,
        )

        print("======================================================================")
        print("GUAVA WORKER START")
        print("======================================================================")
        print(f"gpu_id        : {gpu_id}")
        print(f"master_ip     : {master_ip}")
        print(f"master_port   : {master_port}")
        print(f"world_size    : {cfg.num_workers}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                print(f"using device  : cuda:{gpu_id} -> {torch.cuda.get_device_name(gpu_id)}")
            except Exception as e:
                print(f"WARNING: could not query GPU name for {gpu_id}: {e}")
        else:
            print("WARNING: CUDA not available. Will run on CPU (slow).")

        print("======================================================================")
        print("connecting to orchestrator and starting training loop...")
        print("======================================================================")

        # This blocks while training / serving requests from orchestrator
        runtime.connect_and_train()

        print("======================================================================")
        print(f"worker (gpu {gpu_id}) finished / shutdown")
        print("======================================================================")

    t = threading.Thread(target=worker_main, daemon=False)
    t.start()
    return t


# ======================================================================
# MAIN ENTRY
# ======================================================================

def main() -> None:
    # ---------------------------
    # CLI args
    # ---------------------------
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--gpu-ids",
        required=True,
        help="Comma-separated local GPU ids to run, e.g. '0,1'. Use '0' for single GPU.",
    )
    ap.add_argument(
        "--master-ip",
        required=True,
        help="IP address where orchestrator is listening (the same it printed as master_ip).",
    )
    ap.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Base port of orchestrator (orchestrator also uses +1..+7 internally).",
    )
    ap.add_argument(
        "--world-size",
        type=int,
        required=True,
        help="TOTAL number of workers the orchestrator expects across ALL machines. "
             "For example: if this box runs GPUs 0 and 1 and that's ALL workers, this is 2.",
    )

    # These hyperparams MUST mirror orchestrator_train.py,
    # because both sides must agree about model shape, batch size, etc.
    ap.add_argument("--batch-size", type=int, default=2, help="Per-worker batch size (must match orchestrator).")
    ap.add_argument("--vocab-size", type=int, default=100, help="Toy vocabulary size.")
    ap.add_argument("--d-model", type=int, default=32, help="Embedding dimension.")
    ap.add_argument("--n-layers", type=int, default=2, help="Transformer layer count (parity only).")
    ap.add_argument("--n-heads", type=int, default=4, help="Attention heads (parity only).")

    args = ap.parse_args()

    # ---------------------------
    # Prep GPU list
    # ---------------------------
    gpu_ids = parse_gpu_list(args.gpu_ids)
    print("======================================================================")
    print("GUAVA MULTI-GPU WORKER LAUNCHER")
    print("======================================================================")
    print(f"local gpu_ids    : {gpu_ids}")
    print(f"orchestrator_ip  : {args.master_ip}")
    print(f"orchestrator_port: {args.master_port}")
    print(f"world_size(total): {args.world_size}")
    print("----------------------------------------------------------------------")
    if torch.cuda.is_available():
        for g in gpu_ids:
            try:
                print(f"GPU {g}: {torch.cuda.get_device_name(g)}")
            except Exception:
                print(f"GPU {g}: (name unavailable)")
    else:
        print("WARNING: torch.cuda.is_available() == False (running on CPU fallback)")
    print("======================================================================")

    # ---------------------------
    # Build DistributedConfig shared by ALL threads on this box
    # ---------------------------
    cfg = DistributedConfig()
    cfg.master_ip = args.master_ip
    cfg.master_port = args.master_port
    cfg.num_workers = args.world_size

    # parallelization strategy: data parallel only (same full model on each GPU)
    cfg.data_parallel = True
    cfg.model_parallel = False
    cfg.pipeline_parallel = False
    cfg.tensor_parallel = False

    # Hyperparams MUST MATCH orchestrator_train.py
    cfg.batch_size = args.batch_size
    cfg.vocab_size = args.vocab_size
    cfg.d_model = args.d_model
    cfg.n_layers = args.n_layers
    cfg.n_heads = args.n_heads
    # DistributedConfig should already carry learning_rate, weight_decay, etc.

    # ---------------------------
    # Build per-GPU NetworkWorker threads
    # ---------------------------
    model_ctor = build_model_factory(cfg)

    threads: List[threading.Thread] = []
    for gpu_id in gpu_ids:
        t = launch_worker_thread(
            gpu_id=gpu_id,
            cfg=cfg,
            master_ip=args.master_ip,
            master_port=args.master_port,
            model_ctor=model_ctor,
        )
        threads.append(t)

    # ---------------------------
    # Block main thread until ALL worker threads exit
    # ---------------------------
    # This keeps everything in ONE terminal session for you.
    for t in threads:
        t.join()

    print("======================================================================")
    print("ALL workers finished / shutdown (this machine)")
    print("======================================================================")


if __name__ == "__main__":
    main()
