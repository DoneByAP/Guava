"""
DistributedConfig
Shared runtime contract between orchestrator and workers.
Describes:
- model shape & tokenizer space
- parallelism strategy (data / pipeline / tensor / hybrid)
- layer mapping per GPU
- socket + reliability tuning
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Tuple
import os, platform


class ParallelismStrategy(Enum):
    DATA_PARALLEL = "data"
    MODEL_PARALLEL = "model"
    PIPELINE_PARALLEL = "pipeline"
    TENSOR_PARALLEL = "tensor"
    HYBRID = "hybrid"


@dataclass
class DistributedConfig:
    # ---------------- Model Architecture ----------------
    vocab_size:    int = 50257
    d_model:       int = 768
    n_heads:       int = 12
    n_layers:      int = 12
    d_ff:          int = 3072
    max_seq_len:   int = 1024
    dropout:       float = 0.1

    # ---------------- Dynamic Model Loading ----------------
    model_source_code: Optional[str] = None
    model_class_name:  Optional[str] = None
    model_init_kwargs: Optional[dict] = None

    # ---------------- Training Hyperparameters ----------------
    batch_size:      int   = 8
    learning_rate:   float = 3e-4
    weight_decay:    float = 0.01
    max_grad_norm:   float = 1.0
    use_amp:         bool  = False

    # ---------------- Cluster ----------------
    num_workers:   int    = 1
    master_ip:     str    = "localhost"
    master_port:   Optional[int] = None  # ← CHANGE from 29500 to None

    # pipeline layout info
    layers_per_gpu: List[int] = field(default_factory=list)
    layer_ranges:   List[Tuple[int,int]] = field(default_factory=list)  # <<< NEW

    # ---------------- Parallelism Strategy Flags ----------------
    data_parallel:     bool = True
    model_parallel:    bool = False
    pipeline_parallel: bool = False
    tensor_parallel:   bool = False  # keep name "tensor_parallel" for backward compat

    micro_batches:        int = 4
    tensor_parallel_size: int = 2

    # ---------------- Communication / Socket Tuning ----------------
    socket_buffer_size:    int   = 16 * 1024 * 1024
    tcp_nodelay:           bool  = True
    enable_keepalive:      bool  = True
    max_message_size_gb:   float = 5.0

    # ---------------- Timeouts / Reliability ----------------
    activation_timeout:     float = 0.0
    ack_timeout:            float = 0.0
    max_resends:            int   = 0
    resend_probe_interval:  float = 5.0

    # ---------------- Activation Caching / Replay ----------------
    activation_cache_steps:  int  = 256
    allow_activation_reuse:  bool = False

    # ---------------- Logging / Telemetry ----------------
    log_step_every:  int   = 100
    compact_log:     bool  = True
    log_activations: bool  = False

    # ---------------- Checkpointing ----------------
    checkpoint_dir: str = "./model/checkpoints"
    save_interval:  int = 1000

    # ---------------- Sanity + derived fields ----------------
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_layers > 0,               "n_layers must be positive"
        assert self.num_workers > 0,            "num_workers must be positive"

        if self.tensor_parallel:
            assert self.d_model % self.tensor_parallel_size == 0, \
                "d_model must be divisible by tensor_parallel_size when tensor_parallel=True"
            assert self.tensor_parallel_size <= self.num_workers, \
                "tensor_parallel_size cannot exceed num_workers"

        # macOS recv/send buffer clamp
        if platform.system() == "Darwin":
            self.socket_buffer_size = min(self.socket_buffer_size, 8 * 1024 * 1024)

        # ensure GPU mapping is filled
        if not self.layers_per_gpu:
            self.layers_per_gpu = self.get_layers_per_gpu()

        # layer_ranges gives explicit [start,end) per GPU for pipeline/model parallel
        self.layer_ranges = self.get_layer_ranges()  # <<< NEW

    # ---------------- Convenience: TP alias ----------------
    @property
    def enable_tensor_parallel(self) -> bool:
        """Backward-compat alias some worker code expects."""
        return bool(self.tensor_parallel)

    def tensor_parallel_groups(self) -> List[List[int]]:
        """
        Example:
        num_workers=4, tensor_parallel_size=2 → [[0,1],[2,3]]
        That defines which GPUs shard a single 'logical layer'.
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
        """
        Normalize/force strategy once we know how many total GPUs are in play.
        We're allowing true HYBRID: data_parallel + pipeline_parallel + tensor_parallel.
        """
        self.num_workers = num_gpus

        # If user explicitly turned on flags, respect them.
        # Otherwise, apply fallback heuristics.
        no_strategy_forced = not (
            self.data_parallel or
            self.model_parallel or
            self.pipeline_parallel or
            self.tensor_parallel
        )

        if num_gpus == 1:
            # With 1 GPU, pipeline/model parallel doesn't make sense.
            self.pipeline_parallel = False
            self.model_parallel    = False
            # tensor_parallel can stay True; it just degenerates to "local split"
            # data_parallel with 1 GPU is kind of meaningless but harmless.
        else:
            if no_strategy_forced:
                # Heuristic: if model is deep enough, we *shard* via pipeline,
                # otherwise just replicate via data-parallel.
                deep_enough = (self.n_layers >= num_gpus * 2)
                if deep_enough:
                    self.pipeline_parallel = True
                    self.model_parallel    = True
                    self.data_parallel     = False
                else:
                    self.data_parallel     = True
                    self.pipeline_parallel = False
                    self.model_parallel    = False
                # tensor_parallel stays False unless user set it.

        # After deciding modes, recompute layout.
        self.layers_per_gpu = self.get_layers_per_gpu()
        self.layer_ranges   = self.get_layer_ranges()  # <<< NEW

    def get_layers_per_gpu(self) -> List[int]:
        """
        Decide how many transformer blocks each GPU 'owns'.

        CASE 1: pure data parallel (replicated full model on each GPU)
        or pure tensor parallel (shard inside layers, but each GPU still
        runs the full depth):
            => every GPU claims all n_layers

        CASE 2: pipeline/model parallel (and maybe also tensor parallel):
            => split depth across GPUs as evenly as possible.
        """
        # pure_data means: only DP, no pipeline/model/tensor
        pure_data = (
            self.data_parallel
            and not self.model_parallel
            and not self.pipeline_parallel
            and not self.tensor_parallel
        )

        # pure_tensor means: only TP (no pipeline/model split)
        pure_tensor = (
            self.tensor_parallel
            and not self.model_parallel
            and not self.pipeline_parallel
        )

        if pure_data or pure_tensor:
            return [self.n_layers] * self.num_workers

        # otherwise, some form of sharding (pipeline/model/hybrid)
        base = self.n_layers // self.num_workers
        extra = self.n_layers % self.num_workers
        layers = [base] * self.num_workers
        for i in range(extra):
            layers[i] += 1
        return layers

    def get_layer_ranges(self) -> List[Tuple[int,int]]:
        """
        Translate layers_per_gpu -> [(start,end), ...].
        GPU0: [0, L0)
        GPU1: [L0, L0+L1)
        etc.

        For DP-only / TP-only (replicated full model),
        everyone just gets (0, n_layers).
        """
        if not self.layers_per_gpu:
            return []

        pure_shared = all(x == self.n_layers for x in self.layers_per_gpu)
        if pure_shared:
            return [(0, self.n_layers) for _ in range(self.num_workers)]

        ranges = []
        cursor = 0
        for L in self.layers_per_gpu:
            start = cursor
            end   = cursor + L
            ranges.append((start, end))
            cursor = end
        return ranges

    def get_parallelism_strategy(self) -> ParallelismStrategy:
        modes = []
        if self.data_parallel:     modes.append(ParallelismStrategy.DATA_PARALLEL)
        if self.model_parallel:    modes.append(ParallelismStrategy.MODEL_PARALLEL)
        if self.pipeline_parallel: modes.append(ParallelismStrategy.PIPELINE_PARALLEL)
        if self.tensor_parallel:   modes.append(ParallelismStrategy.TENSOR_PARALLEL)

        if len(modes) == 0:
            return ParallelismStrategy.DATA_PARALLEL
        if len(modes) == 1:
            return modes[0]
        return ParallelismStrategy.HYBRID  # <<< HYBRID

    def get_max_message_bytes(self) -> int:
        """Convert max_message_size_gb → bytes for socket safety."""
        return int(self.max_message_size_gb * 1024 * 1024 * 1024)

    # ---------------- Serialization helpers ----------------
    @classmethod
    def from_env(cls) -> "DistributedConfig":
        cfg = cls()

        # cluster / networking
        cfg.master_ip     = os.environ.get("MASTER_IP", cfg.master_ip)
        cfg.master_port   = int(os.environ.get("MASTER_PORT", cfg.master_port))
        cfg.num_workers   = int(os.environ.get("NUM_WORKERS", cfg.num_workers))

        # parallel flags
        cfg.data_parallel     = os.environ.get("DATA_PARALLEL",     "1") == "1"
        cfg.model_parallel    = os.environ.get("MODEL_PARALLEL",    "0") == "1"
        cfg.pipeline_parallel = os.environ.get("PIPELINE_PARALLEL", "0") == "1"

        tp_env = os.environ.get("TENSOR_PARALLEL")
        if tp_env is None:
            tp_env = os.environ.get("ENABLE_TENSOR_PARALLEL", "0")
        cfg.tensor_parallel = (tp_env == "1")

        # knobs
        cfg.micro_batches        = int(os.environ.get("MICRO_BATCHES",        str(cfg.micro_batches)))
        cfg.tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", str(cfg.tensor_parallel_size)))

        if "MAX_MESSAGE_SIZE_GB" in os.environ:
            cfg.max_message_size_gb = float(os.environ["MAX_MESSAGE_SIZE_GB"])

        # logging / telemetry
        cfg.compact_log     = os.environ.get("COMPACT_LOG", "1") == "1"
        cfg.log_step_every  = int(os.environ.get("LOG_STEP_EVERY", str(cfg.log_step_every)))
        cfg.log_activations = os.environ.get("LOG_ACT", "0") == "1"

        # activation reuse
        cfg.allow_activation_reuse = os.environ.get("ALLOW_ACT_REUSE", "0") == "1"
        cfg.activation_cache_steps = int(os.environ.get("ACT_CACHE_STEPS", str(cfg.activation_cache_steps)))

        # timeouts / resend
        cfg.activation_timeout    = float(os.environ.get("ACT_TIMEOUT_SEC",   str(cfg.activation_timeout)))
        cfg.ack_timeout           = float(os.environ.get("ACK_TIMEOUT_SEC",   str(cfg.ack_timeout)))
        cfg.max_resends           = int(os.environ.get("RESENDS_MAX",         str(cfg.max_resends)))
        cfg.resend_probe_interval = float(os.environ.get("RESEND_PROBE_SEC",  str(cfg.resend_probe_interval)))

        # model hyperparams (optional overrides)
        if "VOCAB_SIZE"  in os.environ: cfg.vocab_size  = int(os.environ["VOCAB_SIZE"])
        if "D_MODEL"     in os.environ: cfg.d_model     = int(os.environ["D_MODEL"])
        if "N_HEADS"     in os.environ: cfg.n_heads     = int(os.environ["N_HEADS"])
        if "N_LAYERS"    in os.environ: cfg.n_layers    = int(os.environ["N_LAYERS"])
        if "D_FF"        in os.environ: cfg.d_ff        = int(os.environ["D_FF"])
        if "MAX_SEQ_LEN" in os.environ: cfg.max_seq_len = int(os.environ["MAX_SEQ_LEN"])
        if "DROPOUT"     in os.environ: cfg.dropout     = float(os.environ["DROPOUT"])

        # training hyperparams (optional overrides)
        if "BATCH_SIZE"     in os.environ: cfg.batch_size     = int(os.environ["BATCH_SIZE"])
        if "LEARNING_RATE"  in os.environ: cfg.learning_rate  = float(os.environ["LEARNING_RATE"])
        if "WEIGHT_DECAY"   in os.environ: cfg.weight_decay   = float(os.environ["WEIGHT_DECAY"])
        if "MAX_GRAD_NORM"  in os.environ: cfg.max_grad_norm  = float(os.environ["MAX_GRAD_NORM"])
        if "USE_AMP"        in os.environ: cfg.use_amp        = (os.environ["USE_AMP"] == "1")

        cfg.layers_per_gpu = cfg.get_layers_per_gpu()
        cfg.layer_ranges   = cfg.get_layer_ranges()
        return cfg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model_source_code"] = self.model_source_code
        d["model_class_name"]  = self.model_class_name
        d["model_init_kwargs"] = self.model_init_kwargs
        return d

    @classmethod  # <<< FIX: only once
    def from_dict(cls, config_dict: dict) -> "DistributedConfig":
        cfg = cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
        cfg.model_source_code = config_dict.get("model_source_code")
        cfg.model_class_name  = config_dict.get("model_class_name")
        cfg.model_init_kwargs = config_dict.get("model_init_kwargs")

        # make sure derived fields exist
        if not cfg.layers_per_gpu:
            cfg.layers_per_gpu = cfg.get_layers_per_gpu()
        cfg.layer_ranges = cfg.get_layer_ranges()

        return cfg
