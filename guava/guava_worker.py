#!/usr/bin/env python3
"""
guava_worker.py

Spawn one NetworkWorker per GPU ID, connect to orchestrator (pure data-parallel).
No edits to NetworkWorker. Model DEFINITION MATCHES orchestrator exactly.
"""

import argparse, threading, random
from typing import List, Callable
import torch, torch.nn as nn

from guava.config import DistributedConfig
from guava.network_worker import NetworkWorker

# Fast-safe mixed precision features for Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ========================= Model (MUST MATCH ORCHESTRATOR) ========================= #
class TinySpikeTransformerLM(nn.Module):
    """Tiny, stable causal Transformer LM with tied weights and additive causal mask."""
    def __init__(self, vocab_size: int = 256, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 4, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.embedding     = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), 1)
        self.register_buffer("causal_add", mask, persistent=False)

        nn.init.normal_(self.embedding.weight, 0.0, 0.02)
        nn.init.normal_(self.pos_embedding.weight, 0.0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.lm_head.weight = self.embedding.weight  # tie

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        h    = self.embedding(input_ids) + self.pos_embedding(pos)
        h    = self.encoder(h, mask=self.causal_add[:T, :T])
        h    = self.ln_f(h)
        return self.lm_head(h)


# ----------------------------- Helpers ----------------------------- #
def parse_gpu_list(gpu_str: str) -> List[int]:
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]

def build_model_factory(cfg: DistributedConfig) -> Callable[[], nn.Module]:
    def ctor() -> nn.Module:
        m = TinySpikeTransformerLM(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model,
            n_layers=cfg.n_layers, n_heads=cfg.n_heads,
            dropout=cfg.dropout, max_len=cfg.max_seq_len,
        )
        # Stability guard against NaN/Inf logits (keeps CE happy).
        orig_forward = m.forward
        def safe_forward(x: torch.Tensor) -> torch.Tensor:
            out = orig_forward(x)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
            return out.clamp_(-30, 30)
        m.forward = safe_forward  # type: ignore[method-assign]
        return m
    return ctor

def launch_worker_thread(gpu_id, cfg, master_ip, master_port, model_ctor):
    def worker_main():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        runtime = NetworkWorker(
            gpu_id=gpu_id, config=cfg, model_ctor=model_ctor,
            master_ip=master_ip, master_port=master_port,
        )
        print(f"[Worker GPU{gpu_id}] Connecting...")
        runtime.connect_and_train()
        print(f"[Worker GPU{gpu_id}] Finished")
    t = threading.Thread(target=worker_main, daemon=False)
    t.start()
    return t


# ----------------------------- Main ----------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-ids", required=True)
    ap.add_argument("--master-ip", required=True)
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("--world-size", type=int, required=True)

    # Model
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=2048)

    # Optim + scheduler hints propagated via cfg
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--spike-interval", type=int, default=1000)
    ap.add_argument("--spike-len", type=int, default=4)
    ap.add_argument("--spike-mult", type=float, default=3.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--socket-buffer-mb", type=int, default=64)
    args = ap.parse_args()

    gpu_ids = parse_gpu_list(args.gpu_ids)

    # Shared DistributedConfig
    cfg = DistributedConfig()
    cfg.master_ip, cfg.master_port, cfg.num_workers = args.master_ip, args.master_port, args.world_size
    cfg.data_parallel, cfg.model_parallel, cfg.pipeline_parallel, cfg.tensor_parallel = True, False, False, False

    cfg.batch_size, cfg.vocab_size = args.batch_size, args.vocab_size
    cfg.d_model, cfg.n_layers, cfg.n_heads = args.d_model, args.n_layers, args.n_heads
    cfg.dropout, cfg.max_seq_len = args.dropout, args.max_seq_len

    cfg.learning_rate, cfg.weight_decay = args.lr, args.weight_decay
    cfg.warmup_steps, cfg.max_grad_norm = args.warmup, args.max_grad_norm
    cfg.socket_buffer_mb = args.socket_buffer_mb

    cfg.lr_scheduler, cfg.spike_interval = "spike_interval", args.spike_interval
    cfg.spike_len, cfg.spike_mult = args.spike_len, args.spike_mult

    # Quick proof check (CPU)
    print("\n=== QUICK MODEL SANITY TEST ===")
    m = TinySpikeTransformerLM(vocab_size=cfg.vocab_size, d_model=cfg.d_model,
                               n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                               dropout=cfg.dropout, max_len=cfg.max_seq_len)
    x = torch.randint(0, cfg.vocab_size, (1, 64))
    with torch.no_grad():
        out = m(x)
    print("logits mean:", out.mean().item(), "std:", out.std().item())
    print("=== END CHECK ===\n")

    model_ctor = build_model_factory(cfg)

    # Launch one thread per GPU with deterministic seeds
    threads = []
    for gid in gpu_ids:
        seed = 1337 + gid
        random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        threads.append(launch_worker_thread(gid, cfg, args.master_ip, args.master_port, model_ctor))

    for t in threads:
        t.join()
    print("ALL workers done.")

if __name__ == "__main__":
    main()
