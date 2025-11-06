#!/usr/bin/env python3
"""
network_worker.py

One NetworkWorker = one GPU "stage".

What this process does:
1. Connect to orchestrator control socket (master_port+0), send CONTROL_HELLO.
2. Receive CONTROL_ACK, build the correct model (strict or legacy).
3. Run training/inference steps on command:
   - CONTROL_DATA_PARALLEL_STEP      (pure data-parallel step)
   - CONTROL_PIPELINE_PHASE1/PHASE2  (pipeline/model-parallel shards)
   - ACTIVATION_FRAME                (future activation relay)
   - CONTROL_STOP                    (shutdown+upload checkpoint)
4. Upload:
   - METRICS_STEP        → master_port+1  (loss per step, etc.)
   - GRADIENTS_UPLOAD    → master_port+2  (grad shards)
   - CHECKPOINT_SHARD_UPLOAD → master_port+7 (weights)
5. Telemetry / Energy:
   - Live per-GPU heartbeat + watt draw, temps, CPU load, RAM, etc.
   - Per-step sections ("forward", "backward") emit FLOPs, TFLOPs/s, TFLOPs/J.
   - Task tagging: train.gpu0 vs train.gpu1 vs infer.gpu0, so Universe/Lord can see who is doing what.

Parallel modes supported:
- data_parallel       (full model replica per GPU)
- pipeline/model split (each GPU has block range)
- tensor_parallel     (split inside each layer; orchestrator does gather/reduce collectives)

NEW IN THIS VERSION:
- EnergyMonitor integrated back in
- Marked forward/backward sections so you get forward_start/forward_end/backward_end, FLOPs, etc.
- set_model_info() populated with model param count
- Background heartbeat thread started automatically on connect
- PERSISTENT gradient and checkpoint channels with TCP keep-alive
- Exponential backoff reconnect logic with proper rate limiting

NOTE:
- We assume orchestrator_train.py has started:
  - control listener at master_port+0
  - metrics listener at master_port+1
  - gradients listener at master_port+2
  - checkpoint listener at master_port+7
  - UDP energy listener at master_port+8
"""

import os, socket, time, pickle, zlib, traceback
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DistributedConfig
from .socket_utils import optimize_socket_for_network, send_with_size, recv_with_size
from .base_worker import DataParallelWorker, ModelShardWorker
from .protocol import MessageType, Message, MessageProtocol, MessageCorruptedError
from .energy_monitor import EnergyMonitor, get_error_tracker, ErrorSeverity  # <-- telemetry



def _tune_windows_keepalive(sock: socket.socket) -> None:
    """Windows-only keepalive tuning. Safe no-op if unsupported."""
    try:
        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 30_000, 10_000))
    except (AttributeError, OSError):
        pass


def enable_tcp_keepalive(sock: socket.socket) -> None:
    """
    Harden a TCP socket for long-lived training channels.

    What this does:
    - Enables SO_KEEPALIVE so the OS sends 'are you alive?' probes.
    - Tunes keepalive so we detect dead peers fast instead of hanging forever.
    - Disables Nagle (TCP_NODELAY) to reduce control-message latency.
    - Removes Python-level timeouts so recv() can block normally.
    - On Linux: also sets TCP_USER_TIMEOUT so a "half-dead" link gets killed
      in ~120s instead of pretending it's fine forever.

    This does NOT guarantee the socket will live forever (Wi-Fi drop, router
    reboot, laptop sleep will still kill it) — but it makes idle timeout
    much less likely and lets us fail fast so reconnect logic can take over.
    """

    # Send small messages immediately (don't coalesce packets)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass  # not fatal on weird platforms

    # Always request TCP keepalive
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    if os.name == "nt":
        # Windows: start probing after 30s idle, then probe every 10s.
        # Without this tweak, Windows might wait hours before noticing a dead peer.
        try:
            sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 30_000, 10_000))
            # (onoff=1, keepalive_idle_ms=30000, keepalive_interval_ms=10000)
        except (AttributeError, OSError):
            pass  # older Python / non-TCP socket / loopback edge cases
    else:
        # Linux / most UNIX. We only set an option if the constant exists,
        # so this won't blow up on macOS where some are missing.
        try:
            TCP_KEEPIDLE   = getattr(socket, "TCP_KEEPIDLE",   None)   # seconds before first probe
            TCP_KEEPINTVL  = getattr(socket, "TCP_KEEPINTVL",  None)   # seconds between probes
            TCP_KEEPCNT    = getattr(socket, "TCP_KEEPCNT",    None)   # # of failed probes before drop
            TCP_USER_TIMEOUT = getattr(socket, "TCP_USER_TIMEOUT", None)  # ms with no ACK before abort

            if TCP_KEEPIDLE is not None:
                sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPIDLE, 30)      # idle 30s → start probing
            if TCP_KEEPINTVL is not None:
                sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPINTVL, 10)     # probe every 10s
            if TCP_KEEPCNT is not None:
                sock.setsockopt(socket.IPPROTO_TCP, TCP_KEEPCNT, 6)        # 6 failed probes -> dead

            # Fail-fast safety: if the other side stops ACKing entirely,
            # kill this socket after ~120s so caller can reconnect.
            if TCP_USER_TIMEOUT is not None:
                sock.setsockopt(socket.IPPROTO_TCP, TCP_USER_TIMEOUT, 120_000)  # ms
        except OSError:
            # Some OSes (like macOS) don't allow some of these; that's fine.
            pass

    # Block forever at Python layer (no per-call timeout exceptions)
    # We'll rely on kernel keepalive + USER_TIMEOUT instead.
    try:
        sock.settimeout(None)
    except OSError:
        pass

class NetworkWorker:
    """
    Runtime wrapper for ONE GPU.

    Typical legacy boot:
        cfg = DistributedConfig.from_env()
        nw = NetworkWorker(
            gpu_id=0,
            config=cfg,
            model_ctor=lambda: MyTransformer(cfg.vocab_size, cfg.d_model),
            master_ip="192.168.0.177",
            master_port=29500,
        )
        nw.connect_and_train()

    Strict boot (orchestrator = single source of truth for model code):
        nw = NetworkWorker(
            gpu_id=0,
            config=cfg,
            model_ctor=lambda: None,   # ignored
            master_ip="192.168.0.177",
            master_port=29500,
            strict_bootstrap=True,
        )

    Args:
        strict_bootstrap: if True, we ignore local model_ctor() and instead
        exec() the source code + kwargs sent in CONTROL_ACK, so all workers
        build the EXACT SAME model definition broadcast by the orchestrator.
    """

    def __init__(
        self,
        gpu_id: int,
        config: DistributedConfig,
        model_ctor,
        master_ip: str,
        master_port: int,
        mode: str = "train",
        strict_bootstrap: bool = False,
    ):
        self.gpu_id           = gpu_id
        self.cfg              = config
        self.master_ip        = master_ip
        self.master_port      = master_port
        self.model_ctor       = model_ctor
        self.mode             = mode
        self.strict_bootstrap = bool(strict_bootstrap)
        self._inference_only  = (self.mode.lower() == "infer")

        # socket buffer size (bytes) used for TCP tuning
        # Change to:
        self._sock_buf_bytes = int(getattr(self.cfg, "socket_buffer_mb", 10240)) * 1024 * 1024
        # 10240 MB = 10 GB
        # long-lived control socket (master_port+0)
        self.ctrl_sock: Optional[socket.socket] = None

        # long-lived gradient (+2) / checkpoint (+7) channels
        self.grad_sock: Optional[socket.socket]  = None  # master_port+2
        self.chkpt_sock: Optional[socket.socket] = None  # master_port+7

        # single-use upload sockets / addr tuples provided by orchestrator layout
        self.metric_sock_addr: Tuple[str, int] = (master_ip, master_port + 1)
        self.grad_sock_addr:   Tuple[str, int] = (master_ip, master_port + 2)
        self.chkpt_sock_addr:  Tuple[str, int] = (master_ip, master_port + 7)

        # worker wrapper:
        #   DataParallelWorker (full model)
        #   OR ModelShardWorker(start_layer:end_layer)
        self.worker: Optional[Any] = None

        # pipeline cache (final stage keeps logits between PHASE1 & PHASE2)
        self._last_logits: Optional[torch.Tensor] = None

        # step counter bookkeeping (optional use for debugging)
        self._step_counter = 0

        # Energy / telemetry monitor (actual UDP send happens after connect)
        self.energy_monitor: Optional[EnergyMonitor] = None

        # --------- Shutdown control ---------
        self._shutdown_flag = False  # Signal for keepalive threads to stop

        # --------- Backoff/reconnect bookkeeping for persistent channels ---------
        # gradient channel backoff
        self._grad_backoff_sec   = 1.0     # current backoff duration
        self._grad_next_attempt  = 0.0     # unix ts when we may next retry
        self._grad_last_log_time = 0.0     # rate-limit noisy prints
        # checkpoint channel backoff
        self._ckpt_backoff_sec   = 1.0
        self._ckpt_next_attempt  = 0.0
        self._ckpt_last_log_time = 0.0

    # -------------------------------------------------------------------------
    # layer assignment / slicing for pipeline+model parallel
    # -------------------------------------------------------------------------
    def _layer_assignment(self) -> Tuple[int, int]:
        """
        Return [start_layer, end_layer) for this GPU.

        Pure data-parallel case:
          - this GPU "owns" all layers.

        Pipeline / model-parallel case:
          - cfg.layers_per_gpu[gpu_id] tells how many layers we own sequentially.
        """
        pure_dp = (
            self.cfg.data_parallel
            and not self.cfg.model_parallel
            and not self.cfg.pipeline_parallel
            and not getattr(self.cfg, "enable_tensor_parallel", False)
        )
        if pure_dp:
            return (0, self.cfg.n_layers)

        start = sum(self.cfg.layers_per_gpu[: self.gpu_id])
        end   = start + self.cfg.layers_per_gpu[self.gpu_id]
        return (start, end)

    def _extract_shard(self, full_model: nn.Module, start: int, end: int) -> nn.Module:
        """
        Build a shard module that:
          - embeds tokens (only on first shard)
          - runs just [start:end] transformer blocks
          - applies ln_f + lm_head (only on last shard)

        We try to support common naming:
          full_model.tok_emb / pos_emb
          OR full_model.embedding / pos_embedding
          full_model.blocks (or .transformer_blocks)
          full_model.ln_f
          full_model.lm_head
        """

        class ShardModule(nn.Module):
            def __init__(self, parent, start_idx, end_idx, is_first, is_last):
                super().__init__()
                self.is_first = is_first
                self.is_last  = is_last

                # embeddings: support both naming patterns
                self.tok_emb       = getattr(parent, "tok_emb", None)
                self.pos_emb       = getattr(parent, "pos_emb", None)
                self.embedding     = getattr(parent, "embedding", None)
                self.pos_embedding = getattr(parent, "pos_embedding", None)

                # pick which set we're going to use (ChatTransformerLM style uses tok_emb)
                self._use_tok_emb = self.tok_emb is not None

                # if not first shard, nuke embeddings so forward() doesn't try to embed twice
                if not self.is_first:
                    self.tok_emb = None
                    self.pos_emb = None
                    self.embedding = None
                    self.pos_embedding = None

                # blocks: accept .blocks or .transformer_blocks
                blocks = getattr(parent, "blocks", None)
                if blocks is None:
                    blocks = getattr(parent, "transformer_blocks", None)
                if blocks is None:
                    raise RuntimeError("Model missing .blocks/.transformer_blocks")

                self.transformer_blocks = nn.ModuleList(blocks[start_idx:end_idx])

                # output head only on last shard
                self.ln_f    = parent.ln_f    if is_last else None
                self.lm_head = parent.lm_head if is_last else None

            def forward(self, hidden_ids: torch.Tensor) -> torch.Tensor:
                """
                If first shard:
                  hidden_ids is token IDs [B,T] → embed now
                Else:
                  hidden_ids is already activations [B,T,C]
                """
                x = hidden_ids
                if self.is_first:
                    input_ids = hidden_ids
                    B, T = input_ids.shape
                    pos_idx = torch.arange(0, T, device=input_ids.device).unsqueeze(0)

                    if self._use_tok_emb and self.tok_emb is not None:
                        tok = self.tok_emb(input_ids)
                        if self.pos_emb is not None:
                            tok = tok + self.pos_emb(pos_idx)
                        x = tok
                    else:
                        if self.embedding is None:
                            raise RuntimeError("First shard missing embedding layer")
                        tok = self.embedding(input_ids)
                        if self.pos_embedding is not None:
                            tok = tok + self.pos_embedding(pos_idx)
                        x = tok

                # run our slice of transformer blocks
                for block in self.transformer_blocks:
                    x = block(x)

                # last shard projects to logits
                if self.is_last and self.ln_f is not None and self.lm_head is not None:
                    x = self.ln_f(x)
                    x = self.lm_head(x)

                return x

        is_first = (start == 0)
        is_last  = (end == self.cfg.n_layers)
        return ShardModule(full_model, start, end, is_first, is_last)

    # -------------------------------------------------------------------------
    # Tensor Parallel helpers (orchestrator mediates gather/reduce)
    # -------------------------------------------------------------------------
    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """GPU tensor → CPU pickle → zlib. (Sent to orchestrator for TP collective.)"""
        return zlib.compress(pickle.dumps(tensor.detach().cpu(), protocol=4))

    def _deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """Inverse of _serialize_tensor; bring it back onto this GPU."""
        t = pickle.loads(zlib.decompress(data))
        return t.cuda(self.gpu_id) if torch.cuda.is_available() else t

    def _tp_send_recv(self, msg_type: MessageType, step: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ask orchestrator to run a TP collective:
        - TENSOR_FORWARD_GATHER: all-gather shards → concat along last dim
        - TENSOR_BACKWARD_REDUCE: all-reduce/avg grads across peers
        """
        assert self.ctrl_sock is not None, "control socket not connected"
        hdr = Message(msg_type=msg_type, step=step, gpu_id=self.gpu_id)
        MessageProtocol.send_message(self.ctrl_sock, hdr)
        send_with_size(self.ctrl_sock, self._serialize_tensor(tensor))
        data = recv_with_size(self.ctrl_sock)
        return self._deserialize_tensor(data)

    def tensor_gather(self, local: torch.Tensor, step: int) -> torch.Tensor:
        """All-gather partial activations/logits across tensor-parallel peers."""
        tp_en = bool(getattr(self.cfg, "enable_tensor_parallel", False))
        tp_sz = int(getattr(self.cfg, "tensor_parallel_size", 1) or 1)
        if not tp_en or tp_sz <= 1:
            return local
        return self._tp_send_recv(MessageType.TENSOR_FORWARD_GATHER, step, local)

    def tensor_reduce_grad(self, local_grad: torch.Tensor, step: int) -> torch.Tensor:
        """All-reduce/avg grads across tensor-parallel peers."""
        tp_en = bool(getattr(self.cfg, "enable_tensor_parallel", False))
        tp_sz = int(getattr(self.cfg, "tensor_parallel_size", 1) or 1)
        if not tp_en or tp_sz <= 1 or local_grad is None:
            return local_grad
        return self._tp_send_recv(MessageType.TENSOR_BACKWARD_REDUCE, step, local_grad)

    # -------------------------------------------------------------------------
    # cfg sync from orchestrator
    # -------------------------------------------------------------------------
    def _apply_model_config(self, model_cfg: Dict[str, Any]) -> None:
        """
        Take orchestrator's authoritative model_config block from CONTROL_ACK
        and apply it locally so we're in total sync on:
        - dims (d_model, n_layers, etc.)
        - topology (layer_ranges, tp_groups)
        - which parallel modes are active
        """
        self.cfg.d_model     = model_cfg.get("d_model",     self.cfg.d_model)
        self.cfg.n_layers    = model_cfg.get("n_layers",    self.cfg.n_layers)
        self.cfg.n_heads     = model_cfg.get("n_heads",     self.cfg.n_heads)
        self.cfg.vocab_size  = model_cfg.get("vocab_size",  self.cfg.vocab_size)
        self.cfg.max_seq_len = model_cfg.get("max_seq_len", self.cfg.max_seq_len)
        self.cfg.dropout     = model_cfg.get("dropout",     self.cfg.dropout)

        self.cfg.batch_size    = model_cfg.get("batch_size",    self.cfg.batch_size)
        self.cfg.max_grad_norm = model_cfg.get("max_grad_norm", self.cfg.max_grad_norm)

        self.cfg.data_parallel     = bool(model_cfg.get("data_parallel",     self.cfg.data_parallel))
        self.cfg.model_parallel    = bool(model_cfg.get("model_parallel",    self.cfg.model_parallel))
        self.cfg.pipeline_parallel = bool(model_cfg.get("pipeline_parallel", self.cfg.pipeline_parallel))
        self.cfg.tensor_parallel   = bool(model_cfg.get("tensor_parallel",   self.cfg.tensor_parallel))

        self.cfg.tensor_parallel_size = int(model_cfg.get("tensor_parallel_size", self.cfg.tensor_parallel_size))
        self.cfg.micro_batches        = int(model_cfg.get("micro_batches",        self.cfg.micro_batches))

        self.cfg.layers_per_gpu = model_cfg.get("layers_per_gpu", self.cfg.layers_per_gpu)
        self.cfg.layer_ranges   = model_cfg.get("layer_ranges",   getattr(self.cfg, "layer_ranges", []))

        tp_groups        = model_cfg.get("tp_groups", [])
        strat            = model_cfg.get("parallelism_strategy", "?")

        print(f"[Worker GPU{self.gpu_id}] CONTROL_ACK summary:")
        print(f"  strategy={strat}")
        print(f"  d_model={self.cfg.d_model} n_layers={self.cfg.n_layers} n_heads={self.cfg.n_heads}")
        print(f"  data_parallel={self.cfg.data_parallel} pipeline_parallel={self.cfg.pipeline_parallel} tensor_parallel={self.cfg.tensor_parallel}")
        print(f"  layer_ranges={self.cfg.layer_ranges}")
        print(f"  tp_groups={tp_groups}")

    # -------------------------------------------------------------------------
    # strict_bootstrap: build model from orchestrator-sent source
    # -------------------------------------------------------------------------
    def _build_worker_from_ack_strict(self, ack_msg: Message) -> None:
        """
        STRICT PATH:
        - Extract CONTROL_ACK.model_blob {source, class_name, init_kwargs}
        - exec() that source to define the model class here
        - instantiate it with init_kwargs
        - slice if needed for pipeline/model-parallel
        - wrap in DataParallelWorker or ModelShardWorker
        """
        payload    = ack_msg.payload or {}
        model_cfg  = payload.get("model_config", {})
        blob       = payload.get("model_blob", {})

        self._apply_model_config(model_cfg)

        src_text    = blob.get("source")
        class_name  = blob.get("class_name")
        init_kwargs = blob.get("init_kwargs", {}) or {}

        if not src_text or not class_name:
            raise RuntimeError(
                f"[GPU{self.gpu_id}] strict_bootstrap=True but missing model_blob.source/class_name"
            )

        # exec the broadcast model source into a temp namespace
        temp_ns = {"torch": torch, "nn": nn, "F": F}
        try:
            exec(src_text, temp_ns)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"[GPU{self.gpu_id}] exec(model source) failed: {e}")

        if class_name not in temp_ns:
            raise RuntimeError(f"[GPU{self.gpu_id}] class '{class_name}' not found in broadcasted source")

        ModelClass = temp_ns[class_name]

        # build actual nn.Module
        try:
            model = ModelClass(**init_kwargs)
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"[GPU{self.gpu_id}] init {class_name}{init_kwargs} failed: {e}"
            )

        model.train(not self._inference_only)
        for p in model.parameters():
            p.requires_grad_(True)

        start_layer, end_layer = self._layer_assignment()
        pure_dp = (
            self.cfg.data_parallel
            and not self.cfg.model_parallel
            and not self.cfg.pipeline_parallel
            and not self.cfg.tensor_parallel
        )

        if pure_dp:
            w = DataParallelWorker(self.gpu_id, self.cfg)
            w.register_model(model)
            print(f"[GPU{self.gpu_id}] ✅ strict DP worker ready")
        else:
            shard_model = self._extract_shard(model, start_layer, end_layer)
            w = ModelShardWorker(
                self.gpu_id,
                self.cfg,
                layer_start=start_layer,
                layer_end=end_layer,
            )
            w.register_model(shard_model)
            print(f"[GPU{self.gpu_id}] ✅ strict shard worker [{start_layer}:{end_layer})")

        w.set_training_mode(not self._inference_only)
        self.worker = w

        total_params = sum(p.numel() for p in self.worker.model.parameters())
        print(f"[GPU{self.gpu_id}] params={total_params:,} (strict mode)")

        # tell EnergyMonitor what model this GPU is actually running
        if self.energy_monitor:
            self.energy_monitor.set_model_info(
                model_name=class_name,
                params=total_params,
            )

    # -------------------------------------------------------------------------
    # legacy_bootstrap: build model locally (old path)
    # -------------------------------------------------------------------------
    def _build_worker_from_ack_legacy(self, ack_msg: Message) -> None:
        """
        LEGACY PATH:
        - Use local model_ctor() to make the model
        - Slice if pipeline/model-parallel
        - Wrap into DataParallelWorker or ModelShardWorker
        """
        payload   = ack_msg.payload or {}
        model_cfg = payload.get("model_config", {})
        self._apply_model_config(model_cfg)

        full_model = self.model_ctor()
        if full_model is None:
            raise RuntimeError(
                f"[GPU{self.gpu_id}] legacy path but model_ctor() returned None"
            )

        start_layer, end_layer = self._layer_assignment()
        pure_dp = (
            self.cfg.data_parallel
            and not self.cfg.model_parallel
            and not self.cfg.pipeline_parallel
            and not self.cfg.tensor_parallel
        )

        if pure_dp:
            w = DataParallelWorker(self.gpu_id, self.cfg)
            w.register_model(full_model)
            print(f"[GPU{self.gpu_id}] ✅ legacy DP worker ready")
        else:
            shard_model = self._extract_shard(full_model, start_layer, end_layer)
            w = ModelShardWorker(
                self.gpu_id,
                self.cfg,
                layer_start=start_layer,
                layer_end=end_layer,
            )
            w.register_model(shard_model)
            print(f"[GPU{self.gpu_id}] ✅ legacy shard worker [{start_layer}:{end_layer})")

        w.set_training_mode(not self._inference_only)
        self.worker = w

        total_params = sum(p.numel() for p in self.worker.model.parameters())
        print(f"[GPU{self.gpu_id}] params={total_params:,} (legacy mode)")

        # tell EnergyMonitor what model this GPU is actually running
        if self.energy_monitor:
            self.energy_monitor.set_model_info(
                model_name=getattr(self.cfg, "model_class_name", "unknown_model"),
                params=total_params,
            )
        # pick a device for THIS worker thread
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_id}")
        else:
            self.device = torch.device("cpu")

        # move model to that device
        if self.worker and hasattr(self.worker, "model"):
            try:
                self.worker.model.to(self.device, non_blocking=True)
            except Exception:
                self.worker.model.to(self.device)

        # backfill worker.device so older code doesn't explode
        if self.worker and not hasattr(self.worker, "device"):
            self.worker.device = self.device

    # -------------------------------------------------------------------------
    # PERSISTENT CHANNEL RECONNECT HELPERS
    # -------------------------------------------------------------------------
    def _ensure_grad_channel(self) -> None:
        """
        Keep self.grad_sock alive/connected to master_port+2.
        If it's None (dropped / WinError 10053), reconnect with exponential backoff.
        """
        now = time.time()
        if self.grad_sock is not None:
            return
        if now < self._grad_next_attempt:
            return

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(s, buf_bytes=self._sock_buf_bytes)
            enable_tcp_keepalive(s)  # CRITICAL: prevent Windows from killing idle connections
            s.connect(self.grad_sock_addr)

            # identify ourselves on this gradients channel
            hello_msg = Message(
                msg_type=MessageType.GRADIENTS_UPLOAD,
                gpu_id=self.gpu_id,
                step=None,
                payload={"hello": True},
            )
            MessageProtocol.send_message(s, hello_msg)

            self.grad_sock = s
            self._grad_backoff_sec = 1.0
            if now - self._grad_last_log_time > 2.0:
                print(f"[GPU {self.gpu_id}] ✅ gradient channel re-established")
                self._grad_last_log_time = now
        except OSError as e:
            if now - self._grad_last_log_time > 2.0:
                print(f"[GPU {self.gpu_id}] gradient channel connect failed ({e}), backing off {self._grad_backoff_sec:.1f}s")
                self._grad_last_log_time = now
            self._grad_next_attempt = now + self._grad_backoff_sec
            self._grad_backoff_sec = min(self._grad_backoff_sec * 2.0, 10.0)

    def _ensure_chkpt_channel(self) -> None:
        """
        Keep self.chkpt_sock alive/connected to master_port+7.
        We do same style backoff as gradients.
        """
        now = time.time()
        if self.chkpt_sock is not None:
            return
        if now < self._ckpt_next_attempt:
            return

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(s, buf_bytes=self._sock_buf_bytes)
            enable_tcp_keepalive(s)  # CRITICAL: prevent Windows from killing idle connections
            s.connect(self.chkpt_sock_addr)

            hello_msg = Message(
                msg_type=MessageType.CHECKPOINT_SHARD_UPLOAD,
                gpu_id=self.gpu_id,
                payload={"hello": True},
            )
            MessageProtocol.send_message(s, hello_msg)

            self.chkpt_sock = s
            self._ckpt_backoff_sec = 1.0
            if now - self._ckpt_last_log_time > 2.0:
                print(f"[GPU {self.gpu_id}] ✅ checkpoint channel re-established")
                self._ckpt_last_log_time = now
        except OSError as e:
            if now - self._ckpt_last_log_time > 2.0:
                print(f"[GPU {self.gpu_id}] checkpoint channel connect failed ({e}), backing off {self._ckpt_backoff_sec:.1f}s")
                self._ckpt_last_log_time = now
            self._ckpt_next_attempt = now + self._ckpt_backoff_sec
            self._ckpt_backoff_sec = min(self._ckpt_backoff_sec * 2.0, 10.0)

    # -------------------------------------------------------------------------
    # KEEPALIVE THREADS (prevent Windows from killing idle TCP connections)
    # -------------------------------------------------------------------------
    def _start_keepalive_threads(self) -> None:
        """
        Start background threads that send periodic heartbeats on persistent channels
        to prevent Windows from killing idle TCP connections.
        
        Windows can kill TCP connections under various conditions:
        - Idle timeout (even with SO_KEEPALIVE, Windows may be aggressive)
        - High network load / buffer pressure
        - Certain firewall/antivirus software
        - Power management settings
        
        Active heartbeats every 15s keep the connection "warm" and prevent drops.
        """
        import threading
        
        def grad_keepalive():
            while not self._shutdown_flag:
                time.sleep(15.0)  # Every 15 seconds
                if self.grad_sock is not None:
                    try:
                        ping = Message(
                            msg_type=MessageType.CONTROL_HEARTBEAT,
                            gpu_id=self.gpu_id,
                        )
                        MessageProtocol.send_message(self.grad_sock, ping)
                    except (OSError, Exception):
                        # Channel died, reconnect logic will handle it
                        pass
        
        def chkpt_keepalive():
            while not self._shutdown_flag:
                time.sleep(15.0)
                if self.chkpt_sock is not None:
                    try:
                        ping = Message(
                            msg_type=MessageType.CONTROL_HEARTBEAT,
                            gpu_id=self.gpu_id,
                        )
                        MessageProtocol.send_message(self.chkpt_sock, ping)
                    except (OSError, Exception):
                        pass
        
        threading.Thread(target=grad_keepalive, daemon=True, name=f"grad_keepalive_gpu{self.gpu_id}").start()
        threading.Thread(target=chkpt_keepalive, daemon=True, name=f"chkpt_keepalive_gpu{self.gpu_id}").start()
        
        print(f"[GPU {self.gpu_id}] keepalive threads started for persistent channels")

    # -------------------------------------------------------------------------
    # sockets / handshake / bootstrap
    # -------------------------------------------------------------------------
    def _connect_ctrl(self) -> None:
        """
        1. Open long-lived control socket to orchestrator (master_port+0).
        2. Send CONTROL_HELLO {gpu_id,...}.
        3. Receive CONTROL_ACK {model_config, model_blob,...}.
        4. Build self.worker using either strict or legacy path.
        5. Start background heartbeat telemetry.
        """
        # grad socket
        gs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        optimize_socket_for_network(gs, buf_bytes=self._sock_buf_bytes)
        enable_tcp_keepalive(gs)  # CRITICAL: prevent Windows from killing idle connections
        gs.connect(self.grad_sock_addr)
        self.grad_sock = gs

        # checkpoint socket
        cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        optimize_socket_for_network(cs, buf_bytes=self._sock_buf_bytes)
        enable_tcp_keepalive(cs)  # CRITICAL: prevent Windows from killing idle connections
        cs.connect(self.chkpt_sock_addr)
        self.chkpt_sock = cs

        # --- identify ourselves on both persistent data channels ---
        try:
            grad_hello = Message(
                msg_type=MessageType.GRADIENTS_UPLOAD,
                gpu_id=self.gpu_id,
                step=None,
                payload={"hello": True},
            )
            MessageProtocol.send_message(self.grad_sock, grad_hello)
            print(f"[GPU {self.gpu_id}] sent gradient channel hello")
        except Exception as e:
            print(f"[GPU {self.gpu_id}] gradient channel hello failed: {e}")
            # if it dies instantly, _ensure_grad_channel() will revive it later
            try:
                self.grad_sock.close()
            except Exception:
                pass
            self.grad_sock = None

        try:
            ckpt_hello = Message(
                msg_type=MessageType.CHECKPOINT_SHARD_UPLOAD,
                gpu_id=self.gpu_id,
                payload={"hello": True},
            )
            MessageProtocol.send_message(self.chkpt_sock, ckpt_hello)
            print(f"[GPU {self.gpu_id}] sent checkpoint channel hello")
        except Exception as e:
            print(f"[GPU {self.gpu_id}] checkpoint channel hello failed: {e}")
            try:
                self.chkpt_sock.close()
            except Exception:
                pass
            self.chkpt_sock = None

        # connect TCP control line
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        optimize_socket_for_network(s, buf_bytes=self._sock_buf_bytes)
        s.connect((self.master_ip, self.master_port + 0))
        self.ctrl_sock = s

        # set up telemetry monitor BEFORE we start work
        self.energy_monitor = EnergyMonitor(
            role="worker",
            udp_host=self.master_ip,
            udp_port=self.master_port + 8,  # orchestrator UDP listener
            gpu_ids=[self.gpu_id],
            model_d_model=getattr(self.cfg, "d_model", 0),
            model_layers=getattr(self.cfg, "n_layers", 0),
        )

        # spawn background heartbeat loop (1Hz)
        import threading
        threading.Thread(
            target=self.energy_monitor.background,
            kwargs={"evt": "heartbeat", "interval": 1.0},
            daemon=True,
        ).start()

        # announce ourselves
        # NEW: Check if this is a reconnection attempt
        is_reconnecting = getattr(self, '_is_reconnecting', False)
        last_step = getattr(self, '_last_step', 0)
        
        hello = Message(
            msg_type=MessageType.CONTROL_HELLO,
            payload={
                "gpu_id": self.gpu_id,
                "start_layer": 0,
                "end_layer": 0,
                "hostname": socket.gethostname(),
                "tensor_parallel_size": int(getattr(self.cfg, "tensor_parallel_size", 1) or 1),
                "reconnect": is_reconnecting,  
                "last_step": last_step,
            },
            gpu_id=self.gpu_id,
        )
        MessageProtocol.send_message(self.ctrl_sock, hello)
    # wait for CONTROL_ACK (orchestrator registers us + ships model info)
        ack_msg = MessageProtocol.receive_message(self.ctrl_sock, timeout=None, channel_name=f"control[GPU{self.gpu_id}]")
        if (
            ack_msg is None
            or ack_msg.msg_type != MessageType.CONTROL_ACK
            or not isinstance(ack_msg.payload, dict)
            or ack_msg.payload.get("status") != "registered"
        ):
            raise RuntimeError("[worker] registration refused by orchestrator")

        # ✅ NEW: Check if this is a reconnection
        is_reconnect = ack_msg.payload.get("is_reconnect", False)
        current_step = ack_msg.payload.get("current_step", 0)

        if is_reconnect:
            print(f"[GPU {self.gpu_id}] 🔄 RECONNECTION confirmed by orchestrator")
            print(f"[GPU {self.gpu_id}] 🔄 Syncing to orchestrator step: {current_step}")
            self._step_counter = current_step
        else:
            print(f"[GPU {self.gpu_id}] ✅ Initial connection established")

        # actually build the local model wrapper
        if self.strict_bootstrap:
            self._build_worker_from_ack_strict(ack_msg)
        else:
            self._build_worker_from_ack_legacy(ack_msg)

        # ✅ NEW: Send ready confirmation after model is built (only on reconnect)
        if is_reconnect:
            print(f"[GPU {self.gpu_id}] 📡 Sending ready confirmation to orchestrator...")
            ready_ack = Message(
                msg_type=MessageType.CONTROL_ACK,
                payload={"status": "ready"},
                gpu_id=self.gpu_id,
            )
            MessageProtocol.send_message(self.ctrl_sock, ready_ack)
            print(f"[GPU {self.gpu_id}] ✅ Sent ready confirmation")

        # Start keepalive threads AFTER channels are established
        self._start_keepalive_threads()

        # NOTE: your original file called self._connect_data_channels(), keep it:
        # (But we already connected grad/chkpt above, so this might be a no-op)
        # self._connect_data_channels()

    # -------------------------------------------------------------------------
    # uploads / ACK helpers
    # -------------------------------------------------------------------------
    def _send_command_ack(self, cmd_type: str, step: int) -> None:
        """Barrier ACK over ctrl_sock back to orchestrator after we finish a task."""
        if self.ctrl_sock is None:
            return
        try:
            ack = Message(
                msg_type=MessageType.CONTROL_ACK,
                payload={"cmd_type": cmd_type},
                step=step,
                gpu_id=self.gpu_id,
            )
            MessageProtocol.send_message(self.ctrl_sock, ack)
        except Exception:
            pass

    def _send_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Send METRICS_STEP to orchestrator (master_port+1).
        This is non-blocking for training; if it dies, we just continue.
        """
        try:
            ms = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(ms, buf_bytes=self._sock_buf_bytes)
            ms.connect(self.metric_sock_addr)

            msg = Message(
                msg_type=MessageType.METRICS_STEP,
                payload=metrics,
                step=metrics.get("step"),
                gpu_id=self.gpu_id,
                phase=metrics.get("phase"),
            )
            MessageProtocol.send_message(ms, msg)
            ms.close()
        except Exception:
            pass

    # --------- PERSISTENT gradient upload with auto-reconnect ----------
    def _send_gradients(self, grad_payload: Dict[str, torch.Tensor], step: int) -> None:
        """
        Stream grads to orchestrator on the long-lived +2 socket.

        - If the socket died (Windows killing idle TCP etc.), we try to
          re-establish it using _ensure_grad_channel() with exponential backoff.
        - If send fails mid-flight, we mark grad_sock=None so next step will
          attempt reconnect, but we don't crash training.
        """
        # make sure we have a channel (or have scheduled a retry)
        self._ensure_grad_channel()
        if self.grad_sock is None:
            # still backing off, skip quietly so loop keeps going
            return

        msg = Message(
            msg_type=MessageType.GRADIENTS_UPLOAD,
            gpu_id=self.gpu_id,
            step=step,
            payload={"gradients": grad_payload},
        )
        try:
            MessageProtocol.send_message(self.grad_sock, msg, compress=True)
        except OSError as e:
            # kill it, let backoff handle reconnect next step
            try:
                self.grad_sock.close()
            except Exception:
                pass
            self.grad_sock = None

            now = time.time()
            if now - self._grad_last_log_time > 2.0:
                print(f"[GPU {self.gpu_id}] gradient channel dropped ({e}), reconnect scheduled")
                self._grad_last_log_time = now

    def _send_checkpoint(self) -> None:
        """
        Upload a final checkpoint shard to orchestrator (master_port+7).
        Uses persistent chkpt_sock w/ backoff just like gradients.
        Orchestrator replies with a 1-byte OK/NACK on that SAME socket.
        """
        if self._inference_only or not self.worker:
            return
        try:
            state = self.worker.get_model_state()
        except Exception as e:
            print(f"[GPU {self.gpu_id}] checkpoint state grab failed: {e}")
            return

        # make sure chkpt_sock is alive or scheduled
        self._ensure_chkpt_channel()
        if self.chkpt_sock is None:
            print(f"[GPU {self.gpu_id}] ⚠ no checkpoint channel, skipping shard upload")
            return

        payload = {
            "gpu_id": self.gpu_id,
            "filename": f"worker{self.gpu_id}_final.pt",
            "state_dict": state,
        }
        msg = Message(
            msg_type=MessageType.CHECKPOINT_SHARD_UPLOAD,
            payload=payload,
            gpu_id=self.gpu_id,
        )

        s = self.chkpt_sock
        try:
            MessageProtocol.send_message(s, msg, compress=True)
            # expect 1-byte ACK
            try:
                ok = s.recv(1)
                if ok == b"\x01":
                    print(f"[GPU {self.gpu_id}] checkpoint uploaded OK")
                else:
                    print(f"[GPU {self.gpu_id}] checkpoint upload FAIL (NACK)")
            except Exception:
                print(f"[GPU {self.gpu_id}] checkpoint ACK missing (socket error)")
        except OSError as e:
            now = time.time()
            if now - self._ckpt_last_log_time > 2.0:
                print(f"[GPU {self.gpu_id}] checkpoint channel dropped ({e})")
                self._ckpt_last_log_time = now
            try:
                self.chkpt_sock.close()
            except Exception:
                pass
            self.chkpt_sock = None

    # -------------------------------------------------------------------------
    # main loop: wait for commands from orchestrator
    # -------------------------------------------------------------------------
    def connect_and_train(self) -> None:
        """
        Main worker loop with connection resilience and corruption recovery.
        - Build model + start heartbeat via _connect_ctrl()
        - Process commands from orchestrator
        - Gracefully handle disconnections and reconnection attempts
        - Handle message corruption with automatic resend requests
        - On ANY unhandled exception, send an 'error' telemetry packet
        """
        self._connect_ctrl()

        running = True
        connection_lost = False
        reconnect_attempts = 0
        max_reconnect_attempts = 15  # INCREASED from 5
        reconnect_backoff = 2.0
        consecutive_failures = 0  # NEW: track repeated failures
        max_consecutive_failures = 3
        
        try:
            while running:
                try:
                    # If we lost connection, attempt to reconnect
                    if connection_lost:
                        if reconnect_attempts >= max_reconnect_attempts:
                            print(f"[GPU {self.gpu_id}] ❌ Max reconnect attempts ({max_reconnect_attempts}) reached. Giving up.")
                            break
                        
                        reconnect_attempts += 1
                        wait_time = min(reconnect_backoff * reconnect_attempts, 30.0)  # Cap at 30s
                        print(f"[GPU {self.gpu_id}] 🔄 Reconnection attempt {reconnect_attempts}/{max_reconnect_attempts} in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        
                        try:
                            # Try full reconnection
                            self._reconnect_all_channels()
                            connection_lost = False
                            reconnect_attempts = 0
                            consecutive_failures = 0
                            print(f"[GPU {self.gpu_id}] ✅ Reconnection successful! Resuming training.")
                            continue
                        except Exception as reconn_err:
                            print(f"[GPU {self.gpu_id}] ⚠️ Reconnection failed: {reconn_err}")
                            consecutive_failures += 1
                            
                            # If we keep failing, orchestrator may be down
                            if consecutive_failures >= max_consecutive_failures:
                                print(f"[GPU {self.gpu_id}] ❌ {consecutive_failures} consecutive failures.")
                                print(f"[GPU {self.gpu_id}] ⏸️ Orchestrator may be down. Long wait (60s)...")
                                time.sleep(60)
                                consecutive_failures = 0
                            
                            continue
                    
                    # ✅ RECEIVE MESSAGE WITH CORRUPTION HANDLING
                    try:
                        from .protocol import MessageCorruptedError
                        msg = MessageProtocol.receive_message(self.ctrl_sock, timeout=None, channel_name=f"control[GPU{self.gpu_id}]" )
                    
                    except MessageCorruptedError as mce:
                        # Corrupt data received - request resend from orchestrator
                        print(f"[GPU {self.gpu_id}] ❌ Message corruption detected: {mce}")
                        print(f"[GPU {self.gpu_id}] 📡 Requesting resend from orchestrator...")
                        
                        try:
                            resend_req = Message(
                                msg_type=MessageType.CONTROL_RESEND_REQUEST,
                                gpu_id=self.gpu_id,
                                step=self._step_counter,
                                payload={"reason": "corruption", "details": str(mce)},
                            )
                            MessageProtocol.send_message(self.ctrl_sock, resend_req, compress=False)
                            print(f"[GPU {self.gpu_id}] ✅ Resend request sent")
                        except Exception as e:
                            print(f"[GPU {self.gpu_id}] ⚠️ Failed to request resend: {e}")
                            # Mark connection as lost if we can't even send the resend request
                            connection_lost = True
                            
                            # Mark all sockets as dead
                            try:
                                if self.ctrl_sock:
                                    self.ctrl_sock.close()
                            except Exception:
                                pass
                            self.ctrl_sock = None
                            
                            try:
                                if self.grad_sock:
                                    self.grad_sock.close()
                            except Exception:
                                pass
                            self.grad_sock = None
                            
                            try:
                                if self.chkpt_sock:
                                    self.chkpt_sock.close()
                            except Exception:
                                pass
                            self.chkpt_sock = None
                            
                            continue
                        
                        # Wait a bit for orchestrator to process and resend
                        print(f"[GPU {self.gpu_id}] ⏳ Waiting 3s for orchestrator to resend...")
                        time.sleep(3)
                        
                        # Loop back to receive the resent message
                        continue
                    
                except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
                    # Control link died mid-run
                    print(f"[GPU {self.gpu_id}] ⚠️ Connection error: {e}")
                    self._send_worker_error(
                        phase="control_link",
                        step=self._step_counter,
                        exc=e,
                        note="control socket closed, attempting reconnect",
                    )
                    connection_lost = True
                    
                    # Mark all sockets as dead so reconnect logic handles them
                    try:
                        if self.ctrl_sock:
                            self.ctrl_sock.close()
                    except Exception:
                        pass
                    self.ctrl_sock = None
                    
                    try:
                        if self.grad_sock:
                            self.grad_sock.close()
                    except Exception:
                        pass
                    self.grad_sock = None
                    
                    try:
                        if self.chkpt_sock:
                            self.chkpt_sock.close()
                    except Exception:
                        pass
                    self.chkpt_sock = None
                    
                    continue

                if msg is None:
                    # Orchestrator closed cleanly
                    print(f"[GPU {self.gpu_id}] 👋 Orchestrator closed connection cleanly")
                    break

                mtype = msg.msg_type

                if mtype == MessageType.CONTROL_START:
                    print(f"[GPU {self.gpu_id}] 🚀 Received START command")
                    continue

                if mtype == MessageType.CONTROL_STOP:
                    print(f"[GPU {self.gpu_id}] 🛑 Received STOP command")
                    self._send_checkpoint()
                    self._send_goodbye()
                    running = False
                    continue

                if mtype == MessageType.CONTROL_DATA_PARALLEL_STEP:
                    safe_call(self._handle_train_data_parallel, msg, context="train_step", severity=ErrorSeverity.CRITICAL)
                    continue

                if mtype == MessageType.CONTROL_PIPELINE_PHASE1:
                    safe_call(self._handle_pipeline_phase1, msg, context="pipeline_p1", severity=ErrorSeverity.ERROR)
                    continue

                if mtype == MessageType.CONTROL_PIPELINE_PHASE2:
                    safe_call(self._handle_pipeline_phase2, msg, context="pipeline_p2", severity=ErrorSeverity.ERROR)
                    continue

                if mtype == MessageType.ACTIVATION_FRAME:
                    safe_call(self._handle_activation_frame, msg, context="activation", severity=ErrorSeverity.WARNING)
                    continue

            # Normal teardown
            print(f"[GPU {self.gpu_id}] 🧹 Starting graceful shutdown...")
            self._shutdown_flag = True  # Signal keepalive threads to stop
            time.sleep(0.5)  # Give threads a moment to exit
            
            # Close all connections
            try:
                if self.ctrl_sock:
                    self.ctrl_sock.close()
                    print(f"[GPU {self.gpu_id}] Closed control socket")
            except Exception as e:
                print(f"[GPU {self.gpu_id}] Error closing ctrl_sock: {e}")
                
            try:
                if self.grad_sock:
                    self.grad_sock.close()
                    print(f"[GPU {self.gpu_id}] Closed gradient socket")
            except Exception as e:
                print(f"[GPU {self.gpu_id}] Error closing grad_sock: {e}")
                
            try:
                if self.chkpt_sock:
                    self.chkpt_sock.close()
                    print(f"[GPU {self.gpu_id}] Closed checkpoint socket")
            except Exception as e:
                print(f"[GPU {self.gpu_id}] Error closing chkpt_sock: {e}")
                
            if self.worker:
                self.worker.cleanup()
                print(f"[GPU {self.gpu_id}] Worker cleanup complete")
            
            print(f"[GPU {self.gpu_id}] ✅ Graceful shutdown complete")

        except KeyboardInterrupt:
            print(f"[GPU {self.gpu_id}] ⚠️ KeyboardInterrupt received, shutting down...")
            self._send_goodbye()
            self._shutdown_flag = True
            
            # Cleanup
            for sock in [self.ctrl_sock, self.grad_sock, self.chkpt_sock]:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass
            
            if self.worker:
                self.worker.cleanup()
        
        except Exception as e:
            # Something bubbled all the way out (e.g. CUDA launch failure that escaped)
            print(f"[GPU {self.gpu_id}] ❌ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            self._send_worker_error(
                phase="fatal",
                step=self._step_counter,
                exc=e,
                note="unhandled in connect_and_train",
            )
            
            # Best-effort goodbye
            self._send_goodbye()
            
            # Best-effort cleanup even in fatal
            self._shutdown_flag = True
            try:
                if self.ctrl_sock:
                    self.ctrl_sock.close()
            except Exception:
                pass
            try:
                if self.grad_sock:
                    self.grad_sock.close()
            except Exception:
                pass
            try:
                if self.chkpt_sock:
                    self.chkpt_sock.close()
            except Exception:
                pass
            if self.worker:
                self.worker.cleanup()

    def _send_goodbye(self) -> None:
        """
        Tell orchestrator we're shutting down gracefully.
        This helps orchestrator distinguish between crashes and normal exits.
        """
        if self.ctrl_sock is None:
            return
        
        try:
            goodbye = Message(
                msg_type=MessageType.CONTROL_GOODBYE,
                gpu_id=self.gpu_id,
                payload={
                    "reason": "normal_shutdown",
                    "final_step": self._step_counter,
                },
            )
            MessageProtocol.send_message(self.ctrl_sock, goodbye)
            print(f"[GPU {self.gpu_id}] 👋 Sent goodbye to orchestrator")
        except Exception as e:
            print(f"[GPU {self.gpu_id}] ⚠️ Could not send goodbye: {e}")

    def _reconnect_all_channels(self) -> None:
        """
        Attempt to reconnect all three persistent channels with state sync.
        
        Raises:
            Exception if any reconnection fails
        """
        print(f"[GPU {self.gpu_id}] 🔌 Starting full reconnection sequence...")
        
        # STEP 0: Clear local state
        self._last_logits = None
        print(f"[GPU {self.gpu_id}] 🧹 Cleared cached activations")
        
        # STEP 1: Reconnect control socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(s, buf_bytes=self._sock_buf_bytes)
            enable_tcp_keepalive(s)
            s.settimeout(30.0)  # 30s timeout for connection attempt
            s.connect((self.master_ip, self.master_port + 0))
            
            # Re-send HELLO
            hello = Message(
                msg_type=MessageType.CONTROL_HELLO,
                payload={
                    "gpu_id": self.gpu_id,
                    "start_layer": 0,
                    "end_layer": 0,
                    "hostname": socket.gethostname(),
                    "tensor_parallel_size": int(getattr(self.cfg, "tensor_parallel_size", 1) or 1),
                    "reconnect": True,
                    "last_step": self._step_counter,  # Tell orchestrator where we were
                },
                gpu_id=self.gpu_id,
            )
            MessageProtocol.send_message(s, hello)
            
            # CRITICAL: Wait for ACK
            ack_msg = MessageProtocol.receive_message(s, timeout=30.0, channel_name=f"control[GPU{self.gpu_id}]-reconnect" )
            if (
                ack_msg is None
                or ack_msg.msg_type != MessageType.CONTROL_ACK
                or not isinstance(ack_msg.payload, dict)
                or ack_msg.payload.get("status") != "registered"
            ):
                raise RuntimeError("Control socket reconnect ACK failed or timeout")
            
            # Sync step counter
            orchestrator_step = ack_msg.payload.get("current_step", self._step_counter)
            if orchestrator_step != self._step_counter:
                print(f"[GPU {self.gpu_id}] 🔄 Syncing step: {self._step_counter} -> {orchestrator_step}")
                self._step_counter = orchestrator_step
            
            self.ctrl_sock = s
            print(f"[GPU {self.gpu_id}] ✅ Control socket reconnected (step {orchestrator_step})")
        
        except Exception as e:
            raise RuntimeError(f"Control socket reconnect failed: {e}")
        
        # STEP 2: Reconnect gradient socket
        try:
            gs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(gs, buf_bytes=self._sock_buf_bytes)
            enable_tcp_keepalive(gs)
            gs.settimeout(15.0)
            gs.connect(self.grad_sock_addr)
            
            grad_hello = Message(
                msg_type=MessageType.GRADIENTS_UPLOAD,
                gpu_id=self.gpu_id,
                step=None,
                payload={"hello": True, "reconnect": True},
            )
            MessageProtocol.send_message(gs, grad_hello)
            
            # Give orchestrator time to register
            time.sleep(1.0)
            
            self.grad_sock = gs
            self._grad_backoff_sec = 1.0
            print(f"[GPU {self.gpu_id}] ✅ Gradient socket reconnected")
        
        except Exception as e:
            # If control succeeded but grad fails, still raise
            # Caller will retry entire sequence
            raise RuntimeError(f"Gradient socket reconnect failed: {e}")
        
        # STEP 3: Reconnect checkpoint socket
        try:
            cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(cs, buf_bytes=self._sock_buf_bytes)
            enable_tcp_keepalive(cs)
            cs.settimeout(15.0)
            cs.connect(self.chkpt_sock_addr)
            
            ckpt_hello = Message(
                msg_type=MessageType.CHECKPOINT_SHARD_UPLOAD,
                gpu_id=self.gpu_id,
                payload={"hello": True, "reconnect": True},
            )
            MessageProtocol.send_message(cs, ckpt_hello)
            
            time.sleep(1.0)
            
            self.chkpt_sock = cs
            self._ckpt_backoff_sec = 1.0
            print(f"[GPU {self.gpu_id}] ✅ Checkpoint socket reconnected")
        
        except Exception as e:
            raise RuntimeError(f"Checkpoint socket reconnect failed: {e}")
        
        # STEP 4: Restart keepalive threads
        self._shutdown_flag = False
        self._start_keepalive_threads()
        
        print(f"[GPU {self.gpu_id}] 🎉 All channels reconnected successfully!")

    # -------------------------------------------------------------------------
    # DATA PARALLEL STEP (main training path in data-parallel mode)
    # -------------------------------------------------------------------------
    def _handle_train_data_parallel(self, msg: Message) -> None:
        """
        One global step for this GPU in data-parallel mode.

        Flow:
        1. tensors -> GPU
        2. forward()  (marked, so we log FLOPs/TFLOPs/s/etc.)
        3. CE loss
        4. backward() (marked, so we log FLOPs/TFLOPs/J)
        5. clip grads, tensor-reduce grads, send grads
        6. local optimizer step
        7. metrics upload
        8. CONTROL_ACK

        Telemetry:
        - self.energy_monitor.task("train.gpu{gpu_id}") tags everything in this block
        - self.energy_monitor.mark("forward"/"backward") emits forward_start / forward_end /
          backward_start / backward_end events with FLOPs, GPU watts, TFLOPs/J, etc.
        """
        step         = int(msg.step if msg.step is not None else -1)
        phase        = msg.phase or "train"
        batch        = msg.payload or {}
        ack_required = bool((msg.metadata or {}).get("ack_required", False))

        # ---- make sure we have a device fallback ----
        # ----- lock in device shortcut -----
        if not hasattr(self, "device"):
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{self.gpu_id}")
            else:
                self.device = torch.device("cpu")
        dev = self.device

        # ----- get raw batch from orchestrator -----
        input_ids_raw = batch.get("input_ids")
        labels_raw    = batch.get("labels")

        # normalize input_ids to tensor on OUR device
        # Auto-detect dtype: if raw data contains floats, keep float32; otherwise use long
        if isinstance(input_ids_raw, torch.Tensor):
            # Preserve existing dtype
            input_ids = input_ids_raw.to(dev, non_blocking=True)
        else:
            # Detect dtype from raw data
            import numpy as np
            if isinstance(input_ids_raw, (list, tuple)) and len(input_ids_raw) > 0:
                sample = input_ids_raw[0]
                if isinstance(sample, (list, tuple)) and len(sample) > 0:
                    sample = sample[0]
                # Use float32 if data contains floats, long for integers
                dtype = torch.float32 if isinstance(sample, float) else torch.long
            else:
                dtype = torch.long  # Default for language models
            
            input_ids = torch.as_tensor(input_ids_raw, dtype=dtype, device=dev)

        # normalize labels (may be None)
        if labels_raw is not None:
            if isinstance(labels_raw, torch.Tensor):
                labels = labels_raw.to(dev, non_blocking=True)
            else:
                # Match input_ids dtype
                labels = torch.as_tensor(labels_raw, dtype=input_ids.dtype, device=dev)
        else:
            labels = None

        training_enabled = (phase == "train") and (not self._inference_only)
        self.worker.set_training_mode(training_enabled)

        token_count = int(labels.numel()) if labels is not None else int(input_ids.numel())
        loss_val = None

        # ----- FORWARD (and loss calc) -----
        if self.energy_monitor:
            with self.energy_monitor.task(f"train.gpu{self.gpu_id}"):
                with self.energy_monitor.mark("forward", step=step, tokens=token_count):
                    logits = self.worker.forward(input_ids)
                    # TP gather so we have full vocab/logits on every peer
                    if getattr(self.cfg, "enable_tensor_parallel", False):
                        logits = self.tensor_gather(logits, step)

                    if labels is not None:
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                        )
                        loss_val = float(loss.item())
                    else:
                        loss = None
                # ----- BACKWARD / OPT -----
                if training_enabled and loss is not None:
                    with self.energy_monitor.mark("backward", step=step, tokens=token_count):
                        loss.backward()

                        # grad clip
                        if getattr(self.cfg, "max_grad_norm", 0) and self.cfg.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.worker.model.parameters(),
                                self.cfg.max_grad_norm,
                            )

                        # tensor-parallel grad reduce
                        if getattr(self.cfg, "enable_tensor_parallel", False):
                            for _, param in self.worker.model.named_parameters():
                                if param.grad is not None:
                                    param.grad = self.tensor_reduce_grad(param.grad, step)

                        # send grads + local optimizer step
                        grads_list = self.worker.get_gradients()
                        grad_payload: Dict[str, torch.Tensor] = {}
                        for (pname, param), g in zip(self.worker.model.named_parameters(), grads_list):
                            grad_payload[pname] = (
                                g.detach().cpu() if g is not None
                                else torch.zeros_like(param).cpu()
                            )
                            safe_call(
                                self._send_gradients, 
                                grad_payload, 
                                step, 
                                context="grad_upload", 
                                severity=ErrorSeverity.WARNING,
                                reraise=False  # Swallow error, keep training
                            )
                        self._send_gradients(grad_payload, step)
                        self.worker.update_weights()

                    # custom "backward_end" line w/ loss for convenience
                    try:
                        self.energy_monitor.send(
                            event="backward_end",
                            step=step,
                            meta={"loss": loss_val},
                        )
                    except Exception:
                        pass

        else:
            # fallback path if somehow no monitor (shouldn't really happen now)
            logits = self.worker.forward(input_ids)
            if getattr(self.cfg, "enable_tensor_parallel", False):
                logits = self.tensor_gather(logits, step)
            if labels is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
                loss_val = float(loss.item())
                if training_enabled:
                    loss.backward()
                    if getattr(self.cfg, "max_grad_norm", 0) and self.cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.worker.model.parameters(),
                            self.cfg.max_grad_norm,
                        )
                    if getattr(self.cfg, "enable_tensor_parallel", False):
                        for _, param in self.worker.model.named_parameters():
                            if param.grad is not None:
                                param.grad = self.tensor_reduce_grad(param.grad, step)
                    grads_list = self.worker.get_gradients()
                    grad_payload = {}
                    for (pname, param), g in zip(self.worker.model.named_parameters(), grads_list):
                        grad_payload[pname] = (
                            g.detach().cpu() if g is not None
                            else torch.zeros_like(param).cpu()
                        )
                    self._send_gradients(grad_payload, step)
                    # ✅ NEW: Safe gradient upload
                    safe_call(
                        self._send_gradients, 
                        grad_payload, 
                        step, 
                        context="grad_upload_fallback", 
                        severity=ErrorSeverity.WARNING,
                        reraise=False
                    )
                    self.worker.update_weights()

        # ship metrics (loss, token_count, etc.) to orchestrator metrics server
        safe_call(
            self._send_metrics,
            {
                "gpu_id": self.gpu_id,
                "step": step,
                "phase": phase,
                "loss": loss_val,
                "token_count": token_count,
                "timestamp": time.time(),
            },
            context="metrics_upload",
            severity=ErrorSeverity.INFO,
            reraise=False  # Best effort - don't stop training if metrics fail
        )

        # unblock orchestrator if it wants ACK barrier
        if ack_required:
            self._send_command_ack("CONTROL_DATA_PARALLEL_STEP", step)

    # -------------------------------------------------------------------------
    # PIPELINE / MODEL PARALLEL PHASE 1 (forward / cache logits on last shard)
    # -------------------------------------------------------------------------
    def _handle_pipeline_phase1(self, msg: Message) -> None:
        """
        Phase1 = forward pass through our shard.
        - First shard: gets input_ids.
        - Middle shards: (future) will receive ACTIVATION_FRAME from previous stage.
        - Last shard: will cache logits in self._last_logits for Phase2.
        We still tag telemetry as train.gpu{X} and mark forward().
        """
        step         = int(msg.step if msg.step is not None else -1)
        phase        = msg.phase or "train"
        batch        = msg.payload or {}
        ack_required = bool((msg.metadata or {}).get("ack_required", False))

        self.worker.set_training_mode(phase == "train")

        input_ids_raw = batch.get("input_ids", None)
        activ = None
        if input_ids_raw is not None:
            input_ids = torch.tensor(
                input_ids_raw,
                dtype=torch.long,
                device=self.worker.device,
            )

            if self.energy_monitor:
                with self.energy_monitor.task(f"train.gpu{self.gpu_id}"):
                    with self.energy_monitor.mark("forward", step=step, tokens=int(input_ids.numel())):
                        activ = self.worker.forward(input_ids)
                        # if we're last shard, gather full TP logits
                        start_layer, end_layer = self._layer_assignment()
                        if end_layer == self.cfg.n_layers and getattr(self.cfg, "enable_tensor_parallel", False):
                            activ = self.tensor_gather(activ, step)
            else:
                activ = self.worker.forward(input_ids)
                start_layer, end_layer = self._layer_assignment()
                if end_layer == self.cfg.n_layers and getattr(self.cfg, "enable_tensor_parallel", False):
                    activ = self.tensor_gather(activ, step)

        # if this GPU is final shard, stash logits for PHASE2
        start_layer, end_layer = self._layer_assignment()
        if activ is not None and end_layer == self.cfg.n_layers:
            self._last_logits = activ

        # (future) send ACTIVATION_FRAME to next shard if not final

        if ack_required:
            self._send_command_ack("CONTROL_PIPELINE_PHASE1", step)

    # -------------------------------------------------------------------------
    # PIPELINE / MODEL PARALLEL PHASE 2 (loss + backward only on last shard)
    # -------------------------------------------------------------------------
    def _handle_pipeline_phase2(self, msg: Message) -> None:
        """
        Phase2 runs only on the LAST shard in a pipeline-style split.
        Steps:
        1. Take cached logits from Phase1.
        2. Compute CE loss using labels from orchestrator.
        3. backward(), clip, TP grad reduce, send grads, local step.
        4. Send metrics + ACK.
        Telemetry: we mark backward() here so you still get FLOPs / TFLOPs/J.
        """
        step         = int(msg.step if msg.step is not None else -1)
        phase        = msg.phase or "train"
        batch        = msg.payload or {}
        ack_required = bool((msg.metadata or {}).get("ack_required", False))

        # only final shard should actually do PHASE2 work
        last_gpu = self.cfg.num_workers - 1
        if self.gpu_id != last_gpu:
            if ack_required:
                self._send_command_ack("CONTROL_PIPELINE_PHASE2", step)
            return

        self.worker.set_training_mode(phase == "train")

        labels_raw = batch.get("labels", None)
        if labels_raw is None or self._last_logits is None:
            # nothing to compute, just report no loss
            self._send_metrics({
                "gpu_id": self.gpu_id,
                "step": step,
                "phase": phase,
                "loss": None,
                "timestamp": time.time(),
            })
            if ack_required:
                self._send_command_ack("CONTROL_PIPELINE_PHASE2", step)
            return

        # gather logits if tensor-parallel
        if getattr(self.cfg, "enable_tensor_parallel", False):
            self._last_logits = self.tensor_gather(self._last_logits, step)

        labels = torch.tensor(
            labels_raw,
            dtype=torch.long,
            device=self.worker.device,
        )
        token_count = int(labels.numel())

        logits = self._last_logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss_val = float(loss.item())

        if phase == "train":
            if self.energy_monitor:
                with self.energy_monitor.task(f"train.gpu{self.gpu_id}"):
                    with self.energy_monitor.mark("backward", step=step, tokens=token_count):
                        loss.backward()

                        # clip grads
                        if getattr(self.cfg, "max_grad_norm", 0) and self.cfg.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.worker.model.parameters(),
                                self.cfg.max_grad_norm,
                            )

                        # TP grad reduce
                        if getattr(self.cfg, "enable_tensor_parallel", False):
                            for _, param in self.worker.model.named_parameters():
                                if param.grad is not None:
                                    param.grad = self.tensor_reduce_grad(param.grad, step)

                        # ship grads + step weights
                        grad_payload: Dict[str, torch.Tensor] = {}
                        for pname, param in self.worker.model.named_parameters():
                            grad_payload[pname] = (
                                param.grad.detach().cpu() if param.grad is not None
                                else torch.zeros_like(param).cpu()
                            )
                        safe_call(
                            self._send_gradients, 
                            grad_payload, 
                            step, 
                            context="grad_upload_pipeline", 
                            severity=ErrorSeverity.WARNING,
                            reraise=False
                        )
                        self.worker.update_weights()

                    # send final backward_end event w/ loss
                    try:
                        self.energy_monitor.send(
                            event="backward_end",
                            step=step,
                            meta={"loss": loss_val},
                        )
                    except Exception:
                        pass
            else:
                # fallback if no monitor
                loss.backward()
                if getattr(self.cfg, "max_grad_norm", 0) and self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.worker.model.parameters(),
                        self.cfg.max_grad_norm,
                    )
                if getattr(self.cfg, "enable_tensor_parallel", False):
                    for _, param in self.worker.model.named_parameters():
                        if param.grad is not None:
                            param.grad = self.tensor_reduce_grad(param.grad, step)
                grad_payload = {}
                for pname, param in self.worker.model.named_parameters():
                    grad_payload[pname] = (
                        param.grad.detach().cpu() if param.grad is not None
                        else torch.zeros_like(param).cpu()
                    )
                safe_call(
                    self._send_gradients, 
                    grad_payload, 
                    step, 
                    context="grad_upload_pipeline_fallback", 
                    severity=ErrorSeverity.WARNING,
                    reraise=False
                )

                self.worker.update_weights()

        # per-step metrics to orchestrator
        safe_call(
            self._send_metrics,
            {
                "gpu_id": self.gpu_id,
                "step": step,
                "phase": phase,
                "loss": loss_val,
                "token_count": token_count,
                "timestamp": time.time(),
            },
            context="metrics_upload_pipeline",
            severity=ErrorSeverity.INFO,
            reraise=False
        )

        if ack_required:
            self._send_command_ack("CONTROL_PIPELINE_PHASE2", step)

        # clear cache for next global step
        self._last_logits = None

    # -------------------------------------------------------------------------
    # ACTIVATION_FRAME (future: relay activations from one shard to next)
    # -------------------------------------------------------------------------
    def _handle_activation_frame(self, msg: Message) -> None:
        """
        This is for pipeline/multi-shard forwarding. The orchestrator can send
        an activation tensor blob that was output by an earlier GPU. We:
        1. rebuild that tensor on *this* GPU
        2. run our local shard forward()
        3. if final shard, cache logits into self._last_logits
        4. ACK back to orchestrator

        NOTE: We'll eventually extend this to forward the activation on to the
        *next* shard via another ACTIVATION_FRAME send.
        """
        step = int(msg.step if msg.step is not None else -1)
        pl   = msg.payload or {}

        tpay = pl.get("tensor_payload")
        if tpay is None:
            self._send_command_ack("ACTIVATION_FRAME", step)
            return

        # unwrap tensor payload (MessageProtocol helper builds a GPU tensor)
        act = MessageProtocol.unwrap_tensor_payload(
            tpay,
            device=self.worker.device,
        )

        # forward through our shard
        if self.energy_monitor:
            with self.energy_monitor.task(f"train.gpu{self.gpu_id}"):
                with self.energy_monitor.mark("forward", step=step, tokens=int(act.numel())):
                    out = self.worker.forward(act)
        else:
            out = self.worker.forward(act)

        start_layer, end_layer = self._layer_assignment()
        if end_layer == self.cfg.n_layers:
            self._last_logits = out

        # (future) send 'out' to next shard via ACTIVATION_FRAME

        self._send_command_ack("ACTIVATION_FRAME", step)

    # -------------------------------------------------------------------------
    # ERROR TELEMETRY HELPERS
    # -------------------------------------------------------------------------
    def _send_worker_error(self, phase: str, step: int, exc: Exception, note: str = "") -> None:
        """
        Emit a last-gasp telemetry packet with event='error' so orchestrator
        logs why this GPU bailed. Safe to call even during a crash.
        """
        if not self.energy_monitor:
            return

        import traceback as _tb
        tb_txt = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))

        try:
            self.energy_monitor.send(
                event="error",
                step=step,
                task=f"train.gpu{self.gpu_id}",
                meta={
                    "phase": phase,
                    "note": note,
                    "exception": str(exc),
                    "traceback": tb_txt[-4000:],  # clip so packet isn't gigantic
                },
            )
        except Exception:
            # if UDP send fails, nothing else we can do
            pass

    def _safe_handle(self, fn, msg, phase: str) -> None:
        """
        Wrap a step handler (like _handle_train_data_parallel). If it throws
        (CUDA launch failure, OOM, etc), we immediately report that to universe
        via _send_worker_error, including traceback + which step we were on.
        """
        try:
            fn(msg)
        except Exception as e:
            # pick a reasonable "current step" for logging
            step_guess = int(getattr(msg, "step", self._step_counter))
            self._send_worker_error(
                phase=phase,
                step=step_guess,
                exc=e,
                note=f"handler {fn.__name__} crashed",
            )
            raise  

def safe_call(func, *args, context="unknown", severity=ErrorSeverity.ERROR, reraise=True, **kwargs):
    """
    Ultra-minimal error-tracking wrapper. Call any function with automatic error reporting.
    
    Usage:
        safe_call(self._handle_train_data_parallel, msg, context="train_step", severity=ErrorSeverity.CRITICAL)
    
    Returns:
        Function result, or None if error was swallowed
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Extract self if it's a method call
        self_obj = args[0] if args and hasattr(args[0], 'gpu_id') else None
        
        try:
            get_error_tracker().report_error(
                error=e,
                context=context,
                severity=severity,
                gpu_id=getattr(self_obj, 'gpu_id', None),
                step=getattr(self_obj, '_step_counter', None),
                metadata={
                    "function": func.__name__,
                    "cuda_error": "CUDA" in str(e) or "cuda" in str(e).lower(),
                },
            )
        except Exception:
            pass  # Never crash on error reporting
        
        if reraise:
            raise
        return None