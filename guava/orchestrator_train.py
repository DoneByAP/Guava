#!/usr/bin/env python3
"""
orchestrator_train.py

Distributed training orchestrator (Guava).
Loads text from ./training_data/test/*.txt (byte-level LM).
- RETURNS TUPLES (X, Y) so Guava can .tolist() cleanly.
- FINITE train/val via --train-batches/--val-batches (prevents hanging).
- Auto-saves + explicit DONE print + clean shutdown.
- RESUME TRAINING: Can continue from recent checkpoints
"""

import argparse, glob, os, sys, time
from typing import List, Dict, Any, Tuple, Optional
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

from guava.config import DistributedConfig
from guava.orchestrator import Orchestrator


# ========================= Model (MUST MATCH WORKER) ========================= #
class TinySpikeTransformerLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # additive causal mask
        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), 1)
        self.register_buffer("causal_add", mask, persistent=False)

        # stable init + tie
        nn.init.normal_(self.embedding.weight, 0.0, 0.02)
        nn.init.normal_(self.pos_embedding.weight, 0.0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)
        self.lm_head.weight = self.embedding.weight  # tie

    def forward(self, input_ids: torch.Tensor):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        x = self.encoder(x, mask=self.causal_add[:T, :T])
        x = self.ln_f(x)
        return self.lm_head(x)


# ========================= Real Text Dataset (FINITE) ========================= #
class TextDataset(Dataset):
    """
    Loads ./training_data/test/*.txt → byte tokens (0..255).
    Returns a FINITE number of batches per epoch via steps_per_epoch.
    Each __getitem__ yields a dict {input_ids, labels} with shape [T].
    """
    def __init__(self, data_dir: str, seq_len: int, vocab_size: int,
                 steps_per_epoch: int, seed: int = 0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.rng = torch.Generator().manual_seed(seed)

        files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        assert files, f"No .txt files found in {data_dir}"

        print("\n📂 TextDataset loading files:")
        for f in files:
            print("  -", os.path.basename(f))
        print(f"✅ Total: {len(files)} files\n")

        text = "\n".join(open(f, "r", encoding="utf-8", errors="ignore").read() for f in files)
        if not text.strip():
            raise RuntimeError(f"Training text in {data_dir} is empty")

        encoded = torch.tensor(list(text.encode("utf-8")), dtype=torch.long) % vocab_size

        # safety: repeat if too short
        if len(encoded) < seq_len + 2:
            reps = (seq_len * 2) // len(encoded) + 1
            encoded = encoded.repeat(reps)

        self.data = encoded

    def __len__(self) -> int:
        return self.steps_per_epoch  # finite!

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        max_off = len(self.data) - self.seq_len - 1
        start = 0 if max_off <= 1 else torch.randint(0, max_off, (1,), generator=self.rng).item()
        x = self.data[start:start+self.seq_len]
        y = self.data[start+1:start+self.seq_len+1]
        return {"input_ids": x, "labels": y}


# ========================= Collate (tuple for Guava) ========================= #
def collate_to_tuple(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.stack([b["input_ids"] for b in batch], dim=0)  # [B,T]
    Y = torch.stack([b["labels"]    for b in batch], dim=0)  # [B,T]
    return X, Y


# ========================= Checkpoint Utils ========================= #
def get_recent_checkpoints(checkpoint_dir: str, n: int = 3) -> List[Tuple[str, float]]:
    """
    Return list of (filepath, timestamp) for the N most recent .pt files.
    Sorted newest first.
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    pt_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not pt_files:
        return []
    
    # Get modification time for each file
    files_with_time = [(f, os.path.getmtime(f)) for f in pt_files]
    files_with_time.sort(key=lambda x: x[1], reverse=True)  # newest first
    
    return files_with_time[:n]


def display_checkpoint_menu(checkpoint_dir: str) -> Optional[str]:
    """
    Show user recent checkpoints and let them choose to resume or start fresh.
    Returns: checkpoint path to load, or None for fresh start.
    """
    recent = get_recent_checkpoints(checkpoint_dir, n=3)
    
    print("\n" + "="*70)
    print("🔄 RESUME TRAINING OPTIONS")
    print("="*70)
    
    if not recent:
        print("❌ No existing checkpoints found in", checkpoint_dir)
        print("✨ Starting fresh training...\n")
        return None
    
    print(f"📦 Found {len(recent)} recent checkpoint(s):\n")
    
    for idx, (fpath, mtime) in enumerate(recent, 1):
        fname = os.path.basename(fpath)
        fsize = os.path.getsize(fpath) / (1024 * 1024)  # MB
        timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{idx}] {fname}")
        print(f"      📅 {timestamp}  |  💾 {fsize:.2f} MB")
    
    print(f"\n  [0] 🆕 Start fresh training (ignore checkpoints)")
    print("="*70)
    
    while True:
        try:
            choice = input("\n👉 Choose option [0-{}]: ".format(len(recent))).strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                print("✨ Starting fresh training...\n")
                return None
            elif 1 <= choice_num <= len(recent):
                selected_path = recent[choice_num - 1][0]
                print(f"✅ Resuming from: {os.path.basename(selected_path)}\n")
                return selected_path
            else:
                print(f"❌ Invalid choice. Enter 0-{len(recent)}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input. Enter a number.")
        except EOFError:
            # Non-interactive mode (e.g., piped input)
            print("⚠️  Non-interactive mode detected. Starting fresh training...\n")
            return None


def load_checkpoint_if_exists(model: nn.Module, checkpoint_path: Optional[str]) -> bool:
    """
    Load model weights from checkpoint if path is provided.
    Returns True if loaded successfully, False otherwise.
    """
    if checkpoint_path is None:
        return False
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("✅ Checkpoint loaded successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print("⚠️  Starting with fresh weights instead...\n")
        return False


# ========================= Args ========================= #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-ip", default="0.0.0.0")
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=1)

    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=2048)

    # NEW: finite steps to avoid hangs
    ap.add_argument("--train-batches", type=int, default=100, help="Dataset size (samples, not batches)")
    ap.add_argument("--val-batches",   type=int, default=20,  help="Validation dataset size")
    ap.add_argument("--val-interval",  type=int, default=100, help="Validate every N train steps")

    ap.add_argument("--checkpoint-dir", default="./checkpoints")
    ap.add_argument("--worker-timeout", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1234)

    # NEW: Resume training options
    ap.add_argument("--resume", type=str, default=None, 
                    help="Path to checkpoint to resume from (skips menu if provided)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Skip resume menu and start fresh")

    # LR hints (into cfg; workers use internally)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--spike-interval", type=int, default=1000)
    ap.add_argument("--spike-len", type=int, default=4)
    ap.add_argument("--spike-mult", type=float, default=3.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--socket-buffer-mb", type=int, default=64)
    return ap.parse_args()


# ========================= Main ========================= #
def main():
    args = parse_args()

    cfg = DistributedConfig()
    cfg.master_ip, cfg.master_port, cfg.num_workers = args.master_ip, args.master_port, args.gpus
    cfg.batch_size, cfg.vocab_size = args.batch_size, args.vocab_size
    cfg.d_model, cfg.n_layers, cfg.n_heads = args.d_model, args.n_layers, args.n_heads
    cfg.dropout, cfg.max_seq_len = args.dropout, args.max_seq_len
    cfg.checkpoint_dir = args.checkpoint_dir

    cfg.data_parallel = True
    cfg.model_parallel = False
    cfg.pipeline_parallel = False
    cfg.tensor_parallel = False

    cfg.learning_rate = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.warmup_steps = args.warmup
    cfg.max_grad_norm = args.max_grad_norm
    cfg.socket_buffer_mb = args.socket_buffer_mb

    cfg.lr_scheduler = "spike_interval"
    cfg.spike_interval, cfg.spike_len, cfg.spike_mult = args.spike_interval, args.spike_len, args.spike_mult

    # Create model first
    model = TinySpikeTransformerLM(
        vocab_size=cfg.vocab_size, d_model=cfg.d_model,
        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        dropout=cfg.dropout, max_len=cfg.max_seq_len
    )

    # ============= RESUME LOGIC ============= #
    checkpoint_to_load = None
    
    if args.resume:
        # Explicit checkpoint path provided
        checkpoint_to_load = args.resume
    elif not args.no_resume:
        # Show interactive menu
        checkpoint_to_load = display_checkpoint_menu(cfg.checkpoint_dir)
    else:
        # --no-resume flag: skip menu
        print("⚡ --no-resume flag: starting fresh training\n")
    
    # Load checkpoint into model if selected
    load_checkpoint_if_exists(model, checkpoint_to_load)

    # Register model with orchestrator
    orch = Orchestrator(cfg)
    orch.register_model(model)

    # Finite train/val datasets
    train_ds = TextDataset("./training_data/test", args.seq_len, args.vocab_size,
                           steps_per_epoch=args.train_batches, seed=args.seed)
    val_ds   = TextDataset("./training_data/test", args.seq_len, args.vocab_size,
                           steps_per_epoch=args.val_batches, seed=args.seed + 10000)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_to_tuple)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_to_tuple)

    print("======================================================================")
    print("GUAVA ORCHESTRATOR TRAIN LOOP")
    print(f"num_workers: {cfg.num_workers}")
    print(f"data files : ./training_data/test/*.txt  | seq_len={args.seq_len}")
    print(f"train/val  : train_batches={args.train_batches}  val_batches={args.val_batches}  val_interval={args.val_interval}")
    print(f"batch_size : {args.batch_size} → ~{args.train_batches // args.batch_size} training steps per epoch")
    if checkpoint_to_load:
        print(f"🔄 RESUMED : {os.path.basename(checkpoint_to_load)}")
    else:
        print(f"✨ FRESH   : Training from scratch")
    print("======================================================================")

    orch.wait_for_workers(timeout=args.worker_timeout)

    # Let Guava orchestrator handle steps/val internally; loaders are finite now.
    orch.start_training(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        val_interval=args.val_interval,
    )

    # === AUTO SAVE + CLEAN EXIT ===
    print("\n================ FINALIZING TRAINING ================")
    try:
        # Save with timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{cfg.checkpoint_dir}/model_{timestamp}_final.pt"
        print("📦 Saving final model...")
        orch.save_checkpoint(path)
        print(f"✅ Saved final model to: {path}")
        
        # Also save as "latest" for easy resumption
        latest_path = f"{cfg.checkpoint_dir}/model_latest.pt"
        orch.save_checkpoint(latest_path)
        print(f"✅ Saved as latest: {latest_path}")
    except Exception as e:
        print("❌ Save failed:", e)

    # optional notifications/cleanup
    try: orch.broadcast({"cmd": "CONTROL_FINISHED"})
    except: pass
    try: orch.close_all_connections()
    except: pass

    print("🎉 Training complete — orchestrator shutting down\n")
    sys.stdout.flush()
    time.sleep(0.5)
    sys.exit(0)


if __name__ == "__main__":
    main()
