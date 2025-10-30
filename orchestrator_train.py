#!/usr/bin/env python3
"""
orchestrator_train.py

Launch this on the orchestrator (CPU) machine. It coordinates one or more
`guava_worker.py` processes that live on GPU hosts.

Responsibilities:
- Build the same TinyToyModel as the worker so parameters align
- Construct a DistributedConfig that mirrors the workers
- Create an Orchestrator runtime and register the model
- Generate toy batches/validation data and drive the training loop

Example usage (1 orchestrator + 2 workers):
    python orchestrator_train.py \
        --master-ip 0.0.0.0 \
        --master-port 29500 \
        --gpus 2 \
        --train-batches 100 \
        --val-interval 20

Make sure each worker is launched with matching hyperparameters, e.g.:
    python guava_worker.py --gpu-id 0 --master-ip <orch-ip> --world-size 2
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from guava.config import DistributedConfig
from guava.orchestrator import Orchestrator


# MUST MATCH guava_worker.py TinyToyModel EXACTLY
class TinyToyModel(nn.Module):
    def __init__(self, vocab_size: int = 100, d_model: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.linear(x)
        return x


@dataclass
class ToyDataConfig:
    num_batches: int
    batch_size: int
    seq_len: int
    vocab_size: int
    seed: int = 0


class TinyToyDataset(Dataset):
    """Dataset that returns full batches of random token IDs + shifted labels."""

    def __init__(self, data_cfg: ToyDataConfig):
        self.cfg = data_cfg

    def __len__(self) -> int:  # type: ignore[override]
        return self.cfg.num_batches

    def __getitem__(self, idx: int):  # type: ignore[override]
        g = torch.Generator()
        g.manual_seed(self.cfg.seed + idx)
        input_ids = torch.randint(
            0,
            self.cfg.vocab_size,
            (self.cfg.batch_size, self.cfg.seq_len),
            dtype=torch.long,
            generator=g,
        )
        labels = (input_ids + 1) % self.cfg.vocab_size
        return input_ids, labels


def build_dataloader(data_cfg: ToyDataConfig) -> DataLoader:
    dataset = TinyToyDataset(data_cfg)
    return DataLoader(dataset, batch_size=None, shuffle=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-ip", default="0.0.0.0", help="Interface/IP for orchestrator to bind")
    ap.add_argument("--master-port", type=int, default=29500, help="Base port for orchestrator sockets")
    ap.add_argument("--gpus", type=int, required=True, help="Total number of workers (must match workers' --world-size)")
    ap.add_argument("--epochs", type=int, default=1, help="Number of epochs to run")
    ap.add_argument("--train-batches", type=int, default=100, help="Batches per epoch for training")
    ap.add_argument("--val-batches", type=int, default=20, help="Batches for validation runs")
    ap.add_argument("--val-interval", type=int, default=25, help="Validate every N steps (0 disables validation)")
    ap.add_argument("--seq-len", type=int, default=16, help="Sequence length for toy data")
    ap.add_argument("--batch-size", type=int, default=2, help="Per-worker batch size (must match workers)")
    ap.add_argument("--vocab-size", type=int, default=100, help="Toy vocabulary size")
    ap.add_argument("--d-model", type=int, default=32, help="Embedding dimension")
    ap.add_argument("--n-layers", type=int, default=2, help="Number of transformer layers (unused in TinyToyModel but kept for config parity)")
    ap.add_argument("--n-heads", type=int, default=4, help="Number of attention heads (for config parity)")
    ap.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory to store orchestrator checkpoints")
    ap.add_argument(
        "--worker-timeout",
        type=float,
        default=None,
        help="Seconds to wait for workers before aborting (None waits forever)",
    )
    ap.add_argument("--seed", type=int, default=1234, help="Base seed for deterministic toy data")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ---------------------------
    # Distributed config mirroring workers
    # ---------------------------
    cfg = DistributedConfig()
    cfg.master_ip = args.master_ip
    cfg.master_port = args.master_port
    cfg.num_workers = args.gpus
    cfg.batch_size = args.batch_size
    cfg.vocab_size = args.vocab_size
    cfg.d_model = args.d_model
    cfg.n_layers = args.n_layers
    cfg.n_heads = args.n_heads
    cfg.checkpoint_dir = args.checkpoint_dir

    cfg.data_parallel = True
    cfg.model_parallel = False
    cfg.pipeline_parallel = False
    cfg.tensor_parallel = False

    # ---------------------------
    # Instantiate orchestrator + model
    # ---------------------------
    orch = Orchestrator(cfg)
    model = TinyToyModel(vocab_size=cfg.vocab_size, d_model=cfg.d_model)
    orch.register_model(model)

    # ---------------------------
    # Build toy train/val loaders
    # ---------------------------
    train_cfg = ToyDataConfig(
        num_batches=args.train_batches,
        batch_size=cfg.batch_size,
        seq_len=args.seq_len,
        vocab_size=cfg.vocab_size,
        seed=args.seed,
    )
    train_loader = build_dataloader(train_cfg)

    val_loader: Optional[DataLoader] = None
    if args.val_interval > 0 and args.val_batches > 0:
        val_cfg = ToyDataConfig(
            num_batches=args.val_batches,
            batch_size=cfg.batch_size,
            seq_len=args.seq_len,
            vocab_size=cfg.vocab_size,
            seed=args.seed + 10_000,
        )
        val_loader = build_dataloader(val_cfg)

    print("======================================================================")
    print("GUAVA ORCHESTRATOR START")
    print("======================================================================")
    print(f"master_ip    : {cfg.master_ip}")
    print(f"master_port  : {cfg.master_port}")
    print(f"num_workers  : {cfg.num_workers}")
    print(f"batch_size   : {cfg.batch_size}")
    print(f"seq_len      : {args.seq_len}")
    print(f"train_batches: {args.train_batches}")
    if val_loader is not None:
        print(f"val_batches  : {args.val_batches} (every {args.val_interval} steps)")
    else:
        print("validation   : disabled")
    print("----------------------------------------------------------------------")
    print("Workers should launch like:")
    print(
        "  python guava_worker.py --gpu-id <ID> "
        f"--master-ip {cfg.master_ip} --master-port {cfg.master_port} "
        f"--world-size {cfg.num_workers}"
    )
    print("======================================================================")
    print("waiting for workers to register...")
    print("======================================================================")

    try:
        orch.wait_for_workers(timeout=args.worker_timeout)
    except TimeoutError as exc:
        print(f"ERROR: {exc}")
        return

    print(f"registered workers: {len(orch.registered_workers)}/{cfg.num_workers}")

    orch.start_training(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        val_interval=max(args.val_interval, 1),
    )

    orch.save_checkpoint(f"{cfg.checkpoint_dir}/orchestrator_final.pt")
    print("======================================================================")
    print("orchestrator finished")
    print("======================================================================")


if __name__ == "__main__":
    main()
