"""
Configuration classes for distributed training.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List
import os
import platform


class ParallelismStrategy(Enum):
    """Parallelism strategies for distributed training."""
    DATA_PARALLEL = "data"
    MODEL_PARALLEL = "model"
    PIPELINE_PARALLEL = "pipeline"
    TENSOR_PARALLEL = "tensor"
    HYBRID = "hybrid"


@dataclass
class DistributedConfig:
    """
    Configuration for distributed neural network training.

    This object is passed everywhere (orchestrator + all workers) so everyone
    agrees on model shape, parallelism mode, networking, safety knobs, etc.

    Attributes:
        # ------------------------
        # Model Architecture
        # ------------------------
        vocab_size: Size of the vocabulary
        d_model: Model hidden size / embedding dim
        n_heads: Number of attention heads
        n_layers: Number of transformer layers (full depth)
        d_ff: Feed-forward MLP dimension
        max_seq_len: Max context length
        dropout: Dropout probability

        # ------------------------
        # Training Hyperparameters
        # ------------------------
        batch_size: Batch size PER WORKER (not global unless data_parallel=False)
        learning_rate: Optimizer LR
        weight_decay: Optimizer weight decay
        max_grad_norm: Gradient clipping max norm
        use_amp: Enable mixed precision (AMP) on forward/backward

        # ------------------------
        # Distributed / Cluster
        # ------------------------
        num_workers: Total workers/GPUs expected in this job
        master_ip: IP of orchestrator / main process
        master_port: Base port; we use +0,+1,... for sockets
        layers_per_gpu: Computed mapping of how many transformer layers each GPU owns
                        (used in model/pipeline parallel mode)
                        Example: [6,6] for 12-layer model on 2 GPUs

        # ------------------------
        # Parallelism Strategy
        # ------------------------
        data_parallel: If True, each GPU has full model, different data batches
        model_parallel: If True, model layers are sharded across GPUs
        pipeline_parallel: If True, we pipeline micro-batches through shards
        tensor_parallel: If True, split big matmuls across GPUs (intra-layer)
        micro_batches: Number of micro-batches for pipeline parallel
        tensor_parallel_size: Size of tensor-parallel group

        # ------------------------
        # Communication / Socket Tuning
        # ------------------------
        socket_buffer_size: Desired TCP buffer size in bytes
        tcp_nodelay: Disable Nagle for low latency
        enable_keepalive: Keep TCP alive so we notice dead peers

        # ------------------------
        # Timeouts / Reliability
        # ------------------------
        activation_timeout: Seconds to wait for activations (0 = no timeout / infinite wait)
        ack_timeout: Seconds to wait for ACK (0 = no timeout / infinite wait)
        max_resends: How many resend probes we'll attempt
        resend_probe_interval: Seconds between resend probes

        # ------------------------
        # Activation Caching
        # ------------------------
        activation_cache_steps: How many steps to keep cached activations
        allow_activation_reuse: Allow replaying cached activations if resend requested

        # ------------------------
        # Logging / Telemetry
        # ------------------------
        log_step_every: Print/log cadence in steps
        compact_log: Minimize verbosity
        log_activations: Whether to log activation transfers

        # ------------------------
        # Checkpointing
        # ------------------------
        checkpoint_dir: Where orchestrator stores checkpoints
        save_interval: Save every N steps
    """

    # ------------------------
    # Model Architecture
    # ------------------------
    vocab_size: int = 50257          # GPT-2 style default
    d_model: int = 768               # hidden size
    n_heads: int = 12                # attention heads
    n_layers: int = 12               # transformer depth
    d_ff: int = 3072                 # feedforward size
    max_seq_len: int = 1024          # context length
    dropout: float = 0.1

    # ------------------------
    # Training Hyperparameters
    # ------------------------
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_amp: bool = False

    # ------------------------
    # Distributed / Cluster
    # ------------------------
    num_workers: int = 1
    master_ip: str = "localhost"
    master_port: int = 29500

    # IMPORTANT: this did not exist before, but NetworkWorker depends on it.
    # We'll fill this in using adapt_to_gpus() or from_env().
    layers_per_gpu: List[int] = field(default_factory=list)

    # ------------------------
    # Parallelism Strategy
    # ------------------------
    data_parallel: bool = True
    model_parallel: bool = False
    pipeline_parallel: bool = False
    tensor_parallel: bool = False

    micro_batches: int = 4              # For pipeline parallel
    tensor_parallel_size: int = 2       # For tensor parallel

    # ------------------------
    # Communication / Socket Tuning
    # ------------------------
    socket_buffer_size: int = 16 * 1024 * 1024  # 16MB default (Linux/Win); macOS will clamp lower
    tcp_nodelay: bool = True
    enable_keepalive: bool = True

    # ------------------------
    # Timeouts / Reliability
    # ------------------------
    activation_timeout: float = 0.0       # 0 = infinite wait; matches "infinite wait + resend"
    ack_timeout: float = 0.0              # 0 = infinite wait
    max_resends: int = 0                  # 0 = do not resend
    resend_probe_interval: float = 5.0    # seconds between resend probes

    # ------------------------
    # Activation Caching / Replay
    # ------------------------
    activation_cache_steps: int = 256
    allow_activation_reuse: bool = False

    # ------------------------
    # Logging / Telemetry
    # ------------------------
    log_step_every: int = 100
    compact_log: bool = True
    log_activations: bool = False

    # ------------------------
    # Checkpointing
    # ------------------------
    checkpoint_dir: str = "./model/checkpoints"
    save_interval: int = 1000  # Save every N steps

    def __post_init__(self):
        """
        Final sanity checks + platform tweaks.
        Runs automatically after dataclass init.
        """
        # d_model must split cleanly across heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        # we can't have a model with 0 layers
        assert self.n_layers > 0, "n_layers must be positive"

        # you must have >=1 worker
        assert self.num_workers > 0, "num_workers must be positive"

        # if we're doing model_parallel, make sure we even have enough layers to shard
        if self.model_parallel:
            assert self.n_layers >= self.num_workers, (
                f"n_layers ({self.n_layers}) must be >= num_workers ({self.num_workers}) "
                "for model parallelism"
            )

        # if we're doing tensor parallel, your d_model must be splittable
        if self.tensor_parallel:
            assert self.d_model % self.tensor_parallel_size == 0, (
                "d_model must be divisible by tensor_parallel_size"
            )

        # macOS can't do > ~8MB TCP buffers by default
        if platform.system() == 'Darwin':
            self.socket_buffer_size = min(self.socket_buffer_size, 8 * 1024 * 1024)

        # If layers_per_gpu wasn't explicitly set, try to fill it to avoid surprises.
        # This is especially important for model/pipeline parallel workers.
        if not self.layers_per_gpu:
            self.layers_per_gpu = self.get_layers_per_gpu()

    def adapt_to_gpus(self, num_gpus: int) -> None:
        """
        Auto-adjust config based on how many GPUs we detect on this machine.

        Rules:
        - num_workers = num_gpus (each GPU == one worker)
        - If single GPU: disable fancy parallelism.
        - If we have depth to shard (n_layers >= num_gpus * 2):
              default to model_parallel.
          else:
              default to data_parallel.
        - Always compute layers_per_gpu so NetworkWorker can slice [start:end].

        Args:
            num_gpus: how many CUDA devices are visible on this node.
        """
        self.num_workers = num_gpus

        if num_gpus == 1:
            # All parallel splits off, it's just 1 GPU doing everything.
            self.data_parallel = False
            self.model_parallel = False
            self.pipeline_parallel = False
            self.tensor_parallel = False
        elif self.n_layers >= num_gpus * 2:
            # Deep enough to slice layers across GPUs
            if not self.data_parallel and not self.model_parallel:
                self.model_parallel = True
        else:
            # Default fallback: data parallel (replicate full model on each GPU)
            if not self.data_parallel and not self.model_parallel:
                self.data_parallel = True

        # Precompute shard plan for model/pipeline parallel usage.
        self.layers_per_gpu = self.get_layers_per_gpu()

    def get_layers_per_gpu(self) -> List[int]:
        """
        Compute how many transformer layers each GPU should own.

        Returns:
            layers_per_gpu: list[int] length == num_workers.
                - If data_parallel only: everyone "has" the full stack, so just [n_layers]*num_workers.
                - If model_parallel: we split n_layers as evenly as possible.
        """
        if not self.model_parallel:
            # In pure data parallel, each GPU logically "has" all layers.
            return [self.n_layers] * self.num_workers

        base_layers = self.n_layers // self.num_workers
        extra_layers = self.n_layers % self.num_workers

        layers = [base_layers] * self.num_workers
        for i in range(extra_layers):
            layers[i] += 1

        return layers

    def get_parallelism_strategy(self) -> ParallelismStrategy:
        """
        Work out what mode(s) we're actually running.

        Returns:
            ParallelismStrategy enum.
            - If exactly one flag is set, that enum.
            - If multiple flags are set, we say HYBRID.
        """
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
            # default fallback is data parallel
            return ParallelismStrategy.DATA_PARALLEL
        elif len(strategies) == 1:
            return strategies[0]
        else:
            return ParallelismStrategy.HYBRID

    @classmethod
    def from_env(cls) -> "DistributedConfig":
        """
        Build a DistributedConfig from environment variables.
        This is what lets a worker node boot with basically just `export MASTER_IP=...`.

        Recognized environment variables:

        # Cluster / networking
        MASTER_IP                (e.g. "192.168.1.100")
        MASTER_PORT              (e.g. "29500")
        NUM_WORKERS              (e.g. "4")

        # Parallelism flags
        DATA_PARALLEL            ("1" or "0")
        MODEL_PARALLEL           ("1" or "0")
        PIPELINE_PARALLEL        ("1" or "0")
        TENSOR_PARALLEL          ("1" or "0")

        # Pipeline / tensor parms
        MICRO_BATCHES            (int)
        TENSOR_PARALLEL_SIZE     (int)

        # Logging
        COMPACT_LOG              ("1" or "0")
        LOG_STEP_EVERY           (int)
        LOG_ACT                  ("1" or "0")

        # Activation reuse / caching
        ALLOW_ACT_REUSE          ("1" or "0")
        ACT_CACHE_STEPS          (int)

        # Timeouts / resend handling
        ACT_TIMEOUT_SEC          (float seconds)
        ACK_TIMEOUT_SEC          (float seconds)
        RESENDS_MAX              (int)
        RESEND_PROBE_SEC         (float seconds)

        # Model hyperparams (optional overrides)
        VOCAB_SIZE               (int)
        D_MODEL                  (int)
        N_HEADS                  (int)
        N_LAYERS                 (int)
        D_FF                     (int)
        MAX_SEQ_LEN              (int)
        DROPOUT                  (float)

        # Training hyperparams (optional overrides)
        BATCH_SIZE               (int)
        LEARNING_RATE            (float)
        WEIGHT_DECAY             (float)
        MAX_GRAD_NORM            (float)
        USE_AMP                  ("1" or "0")
        """
        cfg = cls()

        # --- Cluster / networking ---
        if "MASTER_IP" in os.environ:
            cfg.master_ip = os.environ["MASTER_IP"]
        if "MASTER_PORT" in os.environ:
            cfg.master_port = int(os.environ["MASTER_PORT"])
        if "NUM_WORKERS" in os.environ:
            cfg.num_workers = int(os.environ["NUM_WORKERS"])

        # --- Parallelism flags ---
        cfg.data_parallel     = os.environ.get("DATA_PARALLEL", "1") == "1"
        cfg.model_parallel    = os.environ.get("MODEL_PARALLEL", "0") == "1"
        cfg.pipeline_parallel = os.environ.get("PIPELINE_PARALLEL", "0") == "1"
        cfg.tensor_parallel   = os.environ.get("TENSOR_PARALLEL", "0") == "1"

        # --- Pipeline / tensor settings ---
        cfg.micro_batches        = int(os.environ.get("MICRO_BATCHES",        str(cfg.micro_batches)))
        cfg.tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", str(cfg.tensor_parallel_size)))

        # --- Logging ---
        cfg.compact_log     = os.environ.get("COMPACT_LOG", "1") == "1"
        cfg.log_step_every  = int(os.environ.get("LOG_STEP_EVERY", str(cfg.log_step_every)))
        cfg.log_activations = os.environ.get("LOG_ACT", "0") == "1"

        # --- Activation reuse / caching ---
        cfg.allow_activation_reuse = os.environ.get("ALLOW_ACT_REUSE", "0") == "1"
        cfg.activation_cache_steps = int(os.environ.get("ACT_CACHE_STEPS", str(cfg.activation_cache_steps)))

        # --- Timeouts / retries ---
        cfg.activation_timeout    = float(os.environ.get("ACT_TIMEOUT_SEC",   str(cfg.activation_timeout)))
        cfg.ack_timeout           = float(os.environ.get("ACK_TIMEOUT_SEC",   str(cfg.ack_timeout)))
        cfg.max_resends           = int(os.environ.get("RESENDS_MAX",         str(cfg.max_resends)))
        cfg.resend_probe_interval = float(os.environ.get("RESEND_PROBE_SEC",  str(cfg.resend_probe_interval)))

        # --- Model hyperparams (optional overrides) ---
        if "VOCAB_SIZE"  in os.environ: cfg.vocab_size  = int(os.environ["VOCAB_SIZE"])
        if "D_MODEL"     in os.environ: cfg.d_model     = int(os.environ["D_MODEL"])
        if "N_HEADS"     in os.environ: cfg.n_heads     = int(os.environ["N_HEADS"])
        if "N_LAYERS"    in os.environ: cfg.n_layers    = int(os.environ["N_LAYERS"])
        if "D_FF"        in os.environ: cfg.d_ff        = int(os.environ["D_FF"])
        if "MAX_SEQ_LEN" in os.environ: cfg.max_seq_len = int(os.environ["MAX_SEQ_LEN"])
        if "DROPOUT"     in os.environ: cfg.dropout     = float(os.environ["DROPOUT"])

        # --- Training hyperparams (optional overrides) ---
        if "BATCH_SIZE"     in os.environ: cfg.batch_size     = int(os.environ["BATCH_SIZE"])
        if "LEARNING_RATE"  in os.environ: cfg.learning_rate  = float(os.environ["LEARNING_RATE"])
        if "WEIGHT_DECAY"   in os.environ: cfg.weight_decay   = float(os.environ["WEIGHT_DECAY"])
        if "MAX_GRAD_NORM"  in os.environ: cfg.max_grad_norm  = float(os.environ["MAX_GRAD_NORM"])
        if "USE_AMP"        in os.environ: cfg.use_amp        = (os.environ["USE_AMP"] == "1")

        # Finalize shard allocation so NetworkWorker._layer_assignment() won't explode.
        cfg.layers_per_gpu = cfg.get_layers_per_gpu()

        return cfg

    def to_dict(self) -> dict:
        """
        Return a plain dict of config values.
        Helpful for logging, checkpoint metadata, RPC broadcast, etc.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DistributedConfig":
        """
        Recreate a config that was previously serialized with to_dict().
        """
        return cls(**config_dict)
