"""
Configuration classes for distributed training.
Defines DistributedConfig, which every process (orchestrator + workers)
shares so they agree on model shape, parallelism mode, networking, etc.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List
import os
import platform


class ParallelismStrategy(Enum):
    DATA_PARALLEL = "data"
    MODEL_PARALLEL = "model"
    PIPELINE_PARALLEL = "pipeline"
    TENSOR_PARALLEL = "tensor"
    HYBRID = "hybrid"


@dataclass
class DistributedConfig:
    # ---------------- Model Architecture ----------------
    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1

    # ---------------- Training Hyperparameters ----------------
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_amp: bool = False

    # ---------------- Distributed / Cluster ----------------
    num_workers: int = 1
    master_ip: str = "localhost"
    master_port: int = 29500
    layers_per_gpu: List[int] = field(default_factory=list)

    # ---------------- Parallelism Strategy Flags ----------------
    data_parallel: bool = True
    model_parallel: bool = False
    pipeline_parallel: bool = False
    # NOTE: keep `tensor_parallel` but also expose property alias
    tensor_parallel: bool = False

    micro_batches: int = 4
    tensor_parallel_size: int = 2

    # ---------------- Communication / Socket Tuning ----------------
    socket_buffer_size: int = 16 * 1024 * 1024
    tcp_nodelay: bool = True
    enable_keepalive: bool = True

    # ---------------- Timeouts / Reliability ----------------
    activation_timeout: float = 0.0
    ack_timeout: float = 0.0
    max_resends: int = 0
    resend_probe_interval: float = 5.0

    # ---------------- Activation Caching / Replay ----------------
    activation_cache_steps: int = 256
    allow_activation_reuse: bool = False

    # ---------------- Logging / Telemetry ----------------
    log_step_every: int = 100
    compact_log: bool = True
    log_activations: bool = False

    # ---------------- Checkpointing ----------------
    checkpoint_dir: str = "./model/checkpoints"
    save_interval: int = 1000

    # ---------------- Post-init validation / normalization ----------------
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.num_workers > 0, "num_workers must be positive"

        # TP divisibility + grouping sanity
        if self.tensor_parallel:
            assert self.d_model % self.tensor_parallel_size == 0, (
                "d_model must be divisible by tensor_parallel_size when tensor_parallel=True"
            )
            # If you intend to use all GPUs in one TP group, require divisibility
            assert self.tensor_parallel_size <= self.num_workers, (
                "tensor_parallel_size cannot exceed num_workers"
            )

        # macOS buffer clamp
        if platform.system() == "Darwin":
            self.socket_buffer_size = min(self.socket_buffer_size, 8 * 1024 * 1024)

        # Ensure layers_per_gpu is always populated
        if not self.layers_per_gpu:
            self.layers_per_gpu = self.get_layers_per_gpu()

    # ---------------- Convenience: TP alias & groups ----------------
    @property
    def enable_tensor_parallel(self) -> bool:
        """
        Alias used by workers; keeps backward compatibility with code
        that checks `cfg.enable_tensor_parallel`.
        """
        return bool(self.tensor_parallel)

    def tensor_parallel_groups(self) -> List[List[int]]:
        """
        Return TP groups like [[0,1], [2,3], ...] given num_workers and tp_size.
        """
        tp = int(self.tensor_parallel_size or 1)
        if not self.tensor_parallel or tp <= 1:
            return [[i] for i in range(self.num_workers)]
        groups = []
        for base in range(0, self.num_workers, tp):
            grp = list(range(base, min(base + tp, self.num_workers)))
            if len(grp) == tp:
                groups.append(grp)
        return groups

    # ---------------- GPU topology helpers ----------------
    def adapt_to_gpus(self, num_gpus: int) -> None:
        self.num_workers = num_gpus

        if num_gpus == 1:
            self.data_parallel = False
            self.model_parallel = False
            self.pipeline_parallel = False
            # allow tensor_parallel to remain whatever user set; but with 1 GPU it's a no-op

        else:
            # If user didn't explicitly choose a strategy, default heuristics:
            # Prefer sharding (pipeline+model) only if depth allows; otherwise DP.
            if not (self.model_parallel or self.pipeline_parallel or self.tensor_parallel or not self.data_parallel):
                deep_enough = (self.n_layers >= num_gpus * 2)
                if deep_enough:
                    self.model_parallel = True
                    self.pipeline_parallel = True
                    self.data_parallel = False
                else:
                    self.data_parallel = True
                    self.model_parallel = False
                    self.pipeline_parallel = False
                # tensor_parallel left as user choice

        # Always recompute mapping
        self.layers_per_gpu = self.get_layers_per_gpu()

    def get_layers_per_gpu(self) -> List[int]:
        """
        Decide how many transformer blocks each GPU "owns".

        PURE DATA PARALLEL:
            [n_layers] * num_workers   (each GPU full stack)

        PURE TENSOR PARALLEL (no model/pipeline sharding):
            [n_layers] * num_workers   (each GPU full stack; intra-layer split happens inside layers)

        SHARDED (model_parallel or pipeline_parallel, with/without TP):
            split evenly across GPUs (e.g., 12 -> [6,6], 13 -> [5,4,4]).
        """
        pure_data = self.data_parallel and not (self.model_parallel or self.pipeline_parallel or self.tensor_parallel)
        pure_tensor = self.tensor_parallel and not (self.model_parallel or self.pipeline_parallel)

        if pure_data or pure_tensor:
            return [self.n_layers] * self.num_workers

        # Any model/pipeline sharding (with or without TP): split depth
        base = self.n_layers // self.num_workers
        extra = self.n_layers % self.num_workers
        layers = [base] * self.num_workers
        for i in range(extra):
            layers[i] += 1
        return layers

    def get_parallelism_strategy(self) -> ParallelismStrategy:
        strategies: List[ParallelismStrategy] = []
        if self.data_parallel:
            strategies.append(ParallelismStrategy.DATA_PARALLEL)
        if self.model_parallel:
            strategies.append(ParallelismStrategy.MODEL_PARALLEL)
        if self.pipeline_parallel:
            strategies.append(ParallelismStrategy.PIPELINE_PARALLEL)
        if self.tensor_parallel:
            strategies.append(ParallelismStrategy.TENSOR_PARALLEL)

        if len(strategies) == 0:
            return ParallelismStrategy.DATA_PARALLEL
        if len(strategies) == 1:
            return strategies[0]
        return ParallelismStrategy.HYBRID

    # ---------------- Serialization / bootstrapping ----------------
    @classmethod
    def from_env(cls) -> "DistributedConfig":
        cfg = cls()

        # Cluster / networking
        cfg.master_ip = os.environ.get("MASTER_IP", cfg.master_ip)
        cfg.master_port = int(os.environ.get("MASTER_PORT", cfg.master_port))
        cfg.num_workers = int(os.environ.get("NUM_WORKERS", cfg.num_workers))

        # Parallelism flags
        cfg.data_parallel     = os.environ.get("DATA_PARALLEL",     "1") == "1"
        cfg.model_parallel    = os.environ.get("MODEL_PARALLEL",    "0") == "1"
        cfg.pipeline_parallel = os.environ.get("PIPELINE_PARALLEL", "0") == "1"
        # Accept both TENSOR_PARALLEL and ENABLE_TENSOR_PARALLEL as env keys
        tp_env = os.environ.get("TENSOR_PARALLEL")
        if tp_env is None:
            tp_env = os.environ.get("ENABLE_TENSOR_PARALLEL", "0")
        cfg.tensor_parallel = (tp_env == "1")

        # Pipeline / tensor knobs
        cfg.micro_batches        = int(os.environ.get("MICRO_BATCHES",        str(cfg.micro_batches)))
        cfg.tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", str(cfg.tensor_parallel_size)))

        # Logging
        cfg.compact_log     = os.environ.get("COMPACT_LOG", "1") == "1"
        cfg.log_step_every  = int(os.environ.get("LOG_STEP_EVERY", str(cfg.log_step_every)))
        cfg.log_activations = os.environ.get("LOG_ACT", "0") == "1"

        # Activation reuse / caching
        cfg.allow_activation_reuse = os.environ.get("ALLOW_ACT_REUSE", "0") == "1"
        cfg.activation_cache_steps = int(os.environ.get("ACT_CACHE_STEPS", str(cfg.activation_cache_steps)))

        # Timeouts / retries
        cfg.activation_timeout    = float(os.environ.get("ACT_TIMEOUT_SEC",   str(cfg.activation_timeout)))
        cfg.ack_timeout           = float(os.environ.get("ACK_TIMEOUT_SEC",   str(cfg.ack_timeout)))
        cfg.max_resends           = int(os.environ.get("RESENDS_MAX",         str(cfg.max_resends)))
        cfg.resend_probe_interval = float(os.environ.get("RESEND_PROBE_SEC",  str(cfg.resend_probe_interval)))

        # Model hyperparams (optional overrides)
        if "VOCAB_SIZE"  in os.environ: cfg.vocab_size  = int(os.environ["VOCAB_SIZE"])
        if "D_MODEL"     in os.environ: cfg.d_model     = int(os.environ["D_MODEL"])
        if "N_HEADS"     in os.environ: cfg.n_heads     = int(os.environ["N_HEADS"])
        if "N_LAYERS"    in os.environ: cfg.n_layers    = int(os.environ["N_LAYERS"])
        if "D_FF"        in os.environ: cfg.d_ff        = int(os.environ["D_FF"])
        if "MAX_SEQ_LEN" in os.environ: cfg.max_seq_len = int(os.environ["MAX_SEQ_LEN"])
        if "DROPOUT"     in os.environ: cfg.dropout     = float(os.environ["DROPOUT"])

        # Training hyperparams (optional overrides)
        if "BATCH_SIZE"     in os.environ: cfg.batch_size     = int(os.environ["BATCH_SIZE"])
        if "LEARNING_RATE"  in os.environ: cfg.learning_rate  = float(os.environ["LEARNING_RATE"])
        if "WEIGHT_DECAY"   in os.environ: cfg.weight_decay   = float(os.environ["WEIGHT_DECAY"])
        if "MAX_GRAD_NORM"  in os.environ: cfg.max_grad_norm  = float(os.environ["MAX_GRAD_NORM"])
        if "USE_AMP"        in os.environ: cfg.use_amp        = (os.environ["USE_AMP"] == "1")

        # Finalize mapping
        cfg.layers_per_gpu = cfg.get_layers_per_gpu()
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DistributedConfig":
        return cls(**config_dict)
