"""
network_worker.py

One NetworkWorker runs on one GPU "stage".

Responsibilities:
1. Connect to the orchestrator's control socket (master_port+0),
   register (CONTROL_HELLO), and keep that socket open.
2. Receive step commands as Message objects:
   - CONTROL_DATA_PARALLEL_STEP
   - CONTROL_PIPELINE_PHASE1
   - CONTROL_PIPELINE_PHASE2
   - CONTROL_STOP
3. Run local compute via DataParallelWorker or ModelShardWorker.
4. Upload metrics (METRICS_STEP) to master_port+1.
5. Upload gradients (GRADIENTS_UPLOAD) to master_port+2.
6. Upload checkpoint shards (CHECKPOINT_SHARD_UPLOAD) to master_port+7.

We support:
- data_parallel (every GPU full model)
- model_parallel / pipeline_parallel (layers sharded across GPUs)
- tensor_parallel (true intra-layer splitting with gather/reduce per layer)
"""

import os
import socket
import time
import pickle
import zlib
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DistributedConfig
from .socket_utils import optimize_socket_for_network, send_with_size, recv_with_size
from .base_worker import DataParallelWorker, ModelShardWorker
from .protocol import MessageType, Message, MessageProtocol


class NetworkWorker:
    """
    Runtime wrapper for ONE GPU.

    Usage (example launcher on the worker box):

        cfg = DistributedConfig.from_env()
        nw = NetworkWorker(
            gpu_id=0,
            config=cfg,
            model_ctor=lambda: MyTransformer(cfg.vocab_size, cfg.d_model),
            master_ip=os.environ["MASTER_IP"],
            master_port=int(os.environ["MASTER_PORT"]),
        )
        nw.connect_and_train()

    model_ctor:
        must return the FULL model (so we can slice it if needed).
    """

    def __init__(
        self,
        gpu_id: int,
        config: DistributedConfig,
        model_ctor,
        master_ip: str,
        master_port: int,
    ):
        self.gpu_id = gpu_id
        self.cfg = config
        self.master_ip = master_ip
        self.master_port = master_port
        self.model_ctor = model_ctor  # callable -> nn.Module

        # long-lived control socket to orchestrator (+0)
        self.ctrl_sock: Optional[socket.socket] = None

        # short-lived sockets addresses for uploads
        self.metric_sock_addr: Tuple[str, int] = (master_ip, master_port + 1)
        self.grad_sock_addr: Tuple[str, int] = (master_ip, master_port + 2)
        self.chkpt_sock_addr: Tuple[str, int] = (master_ip, master_port + 7)
        # +3,+4,+5,+6 are reserved (activation relay / heartbeats),
        # we piggyback ACKs/controls on ctrl_sock.

        # figure out which slice of layers we own
        start_layer, end_layer = self._layer_assignment()

        # build local compute worker (full replica OR shard)
        full_model = self.model_ctor()

        pure_data_parallel = (
            self.cfg.data_parallel
            and not self.cfg.model_parallel
            and not self.cfg.pipeline_parallel
            and not getattr(self.cfg, "enable_tensor_parallel", False)
        )

        if pure_data_parallel:
            # data parallel: keep full model replica
            self.worker = DataParallelWorker(self.gpu_id, self.cfg)
            self.worker.register_model(full_model)
        else:
            # model/pipeline/tensor parallel: keep only [start_layer:end_layer)
            shard_model = self._extract_shard(full_model, start_layer, end_layer)
            self.worker = ModelShardWorker(
                self.gpu_id,
                self.cfg,
                layer_start=start_layer,
                layer_end=end_layer,
            )
            self.worker.register_model(shard_model)

        self.worker.set_training_mode(True)

        # for pipeline last-stage use: cache logits from PHASE1 for PHASE2 usage
        self._last_logits: Optional[torch.Tensor] = None

    # -------------------------------------------------------------------------
    # layer assignment and model slicing
    # -------------------------------------------------------------------------
    def _layer_assignment(self) -> Tuple[int, int]:
        """
        Return [start_layer, end_layer) for this GPU.

        data-parallel:
            we "own" the full stack.
        pipeline/model/tensor-parallel:
            cfg.layers_per_gpu[gpu_id] defines how many layers this GPU is responsible for.
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
        end = start + self.cfg.layers_per_gpu[self.gpu_id]
        return (start, end)

    def _extract_shard(
        self,
        full_model: nn.Module,
        start: int,
        end: int,
    ) -> nn.Module:
        """
        Slice the transformer into [start:end] blocks for this GPU.

        Assumptions on full_model:
        - full_model.embedding
        - full_model.pos_embedding (optional)
        - full_model.transformer_blocks  (nn.ModuleList of blocks)
        - full_model.ln_f
        - full_model.lm_head

        Rules:
        - First shard keeps embeddings (and pos_embedding if present).
        - Middle shards hold only their block subset.
        - Last shard holds ln_f + lm_head.
        """

        class ShardModule(nn.Module):
            def __init__(self, parent, start_idx, end_idx, is_first, is_last):
                super().__init__()
                self.is_first = is_first
                self.is_last = is_last

                # first shard: embeddings
                if is_first:
                    self.embedding = parent.embedding
                    self.pos_embedding = getattr(parent, "pos_embedding", None)
                else:
                    self.embedding = None
                    self.pos_embedding = None

                # slice transformer layers
                self.transformer_blocks = nn.ModuleList(
                    parent.transformer_blocks[start_idx:end_idx]
                )

                # last shard: ln_f + lm_head
                if is_last:
                    self.ln_f = parent.ln_f
                    self.lm_head = parent.lm_head
                else:
                    self.ln_f = None
                    self.lm_head = None

            def forward(self, hidden_ids: torch.Tensor) -> torch.Tensor:
                """
                If first shard:
                    hidden_ids are token IDs [B,T] -> run embedding.
                Else:
                    hidden_ids are already activations from prev shard.
                """
                x = hidden_ids
                if self.is_first:
                    input_ids = hidden_ids
                    seq_len = input_ids.size(1)
                    pos_idx = torch.arange(
                        0, seq_len, device=input_ids.device
                    ).unsqueeze(0)
                    tok = self.embedding(input_ids)
                    if self.pos_embedding is not None:
                        tok = tok + self.pos_embedding(pos_idx)
                    x = tok

                # run our sub-blocks
                for block in self.transformer_blocks:
                    x = block(x)

                # last shard produces logits
                if self.is_last:
                    x = self.ln_f(x)
                    x = self.lm_head(x)

                return x

        is_first = (start == 0)
        is_last = (end == self.cfg.n_layers)

        return ShardModule(
            parent=full_model,
            start_idx=start,
            end_idx=end,
            is_first=is_first,
            is_last=is_last,
        )

    # -------------------------------------------------------------------------
    # Tensor Parallel helpers (Megatron-LM style via orchestrator)
    # -------------------------------------------------------------------------
    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        # CPU-serialize + compress
        return zlib.compress(pickle.dumps(tensor.detach().cpu(), protocol=4))

    def _deserialize_tensor(self, data: bytes) -> torch.Tensor:
        # Decompress + load, move to this GPU if available
        t = pickle.loads(zlib.decompress(data))
        return t.cuda(self.gpu_id) if torch.cuda.is_available() else t

    def _tp_peers(self) -> List[int]:
        """
        Return all GPUs that co-own the same tensor-parallel group.
        Example: tp_size=2 → groups [0,1],[2,3],...
        """
        tp = int(getattr(self.cfg, "tensor_parallel_size", 1) or 1)
        if tp <= 1:
            return [self.gpu_id]
        base = self.gpu_id - (self.gpu_id % tp)
        return [base + i for i in range(tp)]

    def tensor_split(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Local shard of a full tensor along dim.
        Typically used to split weights or activations per layer.
        """
        tp = int(getattr(self.cfg, "tensor_parallel_size", 1) or 1)
        if tp <= 1:
            return tensor
        rank = self.gpu_id % tp
        chunks = torch.chunk(tensor, tp, dim=dim)
        return chunks[rank].contiguous()

    def _tp_send_recv(self, msg_type: MessageType, step: int, tensor: torch.Tensor) -> torch.Tensor:
        """
        Send a TENSOR_* request with a tensor payload on the existing control socket,
        then block for a single tensor response from the orchestrator.

        Contract with orchestrator:
          - Worker sends Message(msg_type=TENSOR_* , step=step), then a sized payload.
          - Orchestrator waits for all peers in the group for this step,
            performs the collective, and replies to each with the result tensor.
        """
        assert self.ctrl_sock is not None, "control socket not connected"
        # 1) header
        hdr = Message(
            msg_type=msg_type,
            step=step,
            gpu_id=self.gpu_id,
        )
        MessageProtocol.send_message(self.ctrl_sock, hdr)
        # 2) payload
        send_with_size(self.ctrl_sock, self._serialize_tensor(tensor))
        # 3) response tensor
        data = recv_with_size(self.ctrl_sock)
        return self._deserialize_tensor(data)

    def tensor_gather(self, local: torch.Tensor, step: int) -> torch.Tensor:
        """
        All-gather partial outputs across tensor peers to build the full activation.
        """
        tp_enabled = bool(getattr(self.cfg, "enable_tensor_parallel", False))
        tp_size = int(getattr(self.cfg, "tensor_parallel_size", 1) or 1)
        if not tp_enabled or tp_size <= 1:
            return local
        return self._tp_send_recv(MessageType.TENSOR_FORWARD_GATHER, step, local)

    def tensor_reduce_grad(self, local_grad: torch.Tensor, step: int) -> torch.Tensor:
        """
        All-reduce (average) gradients across tensor peers.
        """
        tp_enabled = bool(getattr(self.cfg, "enable_tensor_parallel", False))
        tp_size = int(getattr(self.cfg, "tensor_parallel_size", 1) or 1)
        if not tp_enabled or tp_size <= 1 or local_grad is None:
            return local_grad
        return self._tp_send_recv(MessageType.TENSOR_BACKWARD_REDUCE, step, local_grad)

    # -------------------------------------------------------------------------
    # socket helpers
    # -------------------------------------------------------------------------
    def _connect_ctrl(self) -> None:
        """
        Connect long-lived control socket (master_port+0),
        send CONTROL_HELLO registration, receive CONTROL_ACK.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        optimize_socket_for_network(s)
        s.connect((self.master_ip, self.master_port + 0))
        self.ctrl_sock = s

        start_layer, end_layer = self._layer_assignment()

        hello = Message(
            msg_type=MessageType.CONTROL_HELLO,
            payload={
                "gpu_id": self.gpu_id,
                "start_layer": start_layer,
                "end_layer": end_layer,
                "hostname": socket.gethostname(),
                "tensor_parallel_size": int(getattr(self.cfg, "tensor_parallel_size", 1) or 1),
            },
            gpu_id=self.gpu_id,
        )
        MessageProtocol.send_message(self.ctrl_sock, hello)

        # wait for CONTROL_ACK(status='registered')
        ack_msg = MessageProtocol.receive_message(self.ctrl_sock, timeout=None)
        if (
            ack_msg is None
            or ack_msg.msg_type != MessageType.CONTROL_ACK
            or not isinstance(ack_msg.payload, dict)
            or ack_msg.payload.get("status") != "registered"
        ):
            raise RuntimeError("registration refused by orchestrator")

    def _send_command_ack(self, cmd_type: str, step: int) -> None:
        """
        Send CONTROL_ACK barrier (no new port, we reuse ctrl_sock).
        """
        if self.ctrl_sock is None:
            return
        ack = Message(
            msg_type=MessageType.CONTROL_ACK,
            payload={"cmd_type": cmd_type},
            step=step,
            gpu_id=self.gpu_id,
        )
        try:
            MessageProtocol.send_message(self.ctrl_sock, ack)
        except Exception:
            pass

    def _send_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Upload METRICS_STEP once to master_port+1.
        Non-fatal if it fails.
        """
        try:
            ms = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(ms)
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

    def _send_gradients(self, grads: Dict[str, torch.Tensor], step: int) -> None:
        """
        Upload GRADIENTS_UPLOAD once to master_port+2.
        If pipeline/model-parallel: ONLY final stage calls this.
        """
        try:
            gs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(gs)
            gs.connect(self.grad_sock_addr)

            msg = Message(
                msg_type=MessageType.GRADIENTS_UPLOAD,
                payload={"gradients": grads},
                gpu_id=self.gpu_id,
                step=step,
            )
            MessageProtocol.send_message(gs, msg, compress=True)
            gs.close()
        except Exception as e:
            print(f"[GPU {self.gpu_id}] gradient upload failed: {e}")

    def _send_checkpoint(self) -> None:
        """
        Send CHECKPOINT_SHARD_UPLOAD to master_port+7 at shutdown so
        orchestrator can persist our shard weights.
        """
        try:
            state = self.worker.get_model_state()
            payload = {
                "gpu_id": self.gpu_id,
                "filename": f"worker{self.gpu_id}_final.pt",
                "state_dict": state,
            }

            cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(cs)
            cs.connect(self.chkpt_sock_addr)

            msg = Message(
                msg_type=MessageType.CHECKPOINT_SHARD_UPLOAD,
                payload=payload,
                gpu_id=self.gpu_id,
            )
            MessageProtocol.send_message(cs, msg, compress=True)

            # read 1-byte OK back
            try:
                ok = cs.recv(1)
                if ok == b"\x01":
                    print(f"[GPU {self.gpu_id}] checkpoint uploaded OK")
                else:
                    print(f"[GPU {self.gpu_id}] checkpoint upload FAIL")
            except Exception:
                pass
            cs.close()
        except Exception as e:
            print(f"[GPU {self.gpu_id}] checkpoint exception: {e}")

    # -------------------------------------------------------------------------
    # main control loop
    # -------------------------------------------------------------------------
    def connect_and_train(self) -> None:
        """
        1. Connect / register.
        2. Loop: receive control Messages on ctrl_sock and handle them.
        3. On CONTROL_STOP, upload checkpoint + exit.
        """
        self._connect_ctrl()

        running = True
        while running:
            try:
                msg = MessageProtocol.receive_message(self.ctrl_sock, timeout=None)
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                break  # orchestrator went away
            if msg is None:
                break  # closed

            # dispatch by msg_type
            mtype = msg.msg_type

            if mtype == MessageType.CONTROL_START:
                # optional warmup/reset hook
                continue

            if mtype == MessageType.CONTROL_STOP:
                # graceful shutdown
                self._send_checkpoint()
                running = False
                continue

            if mtype == MessageType.CONTROL_DATA_PARALLEL_STEP:
                self._handle_train_data_parallel(msg)
                continue

            if mtype == MessageType.CONTROL_PIPELINE_PHASE1:
                self._handle_pipeline_phase1(msg)
                continue

            if mtype == MessageType.CONTROL_PIPELINE_PHASE2:
                self._handle_pipeline_phase2(msg)
                continue

            if mtype == MessageType.ACTIVATION_FRAME:
                self._handle_activation_frame(msg)
                continue

            # ignore unknown / unimplemented types for now

        # cleanup
        try:
            if self.ctrl_sock:
                self.ctrl_sock.close()
        except Exception:
            pass

        self.worker.cleanup()

    # -------------------------------------------------------------------------
    # DATA PARALLEL STEP HANDLER
    # -------------------------------------------------------------------------
    def _handle_train_data_parallel(self, msg: Message) -> None:
        """
        Data-parallel step:
          - Convert lists to tensors on this GPU.
          - Forward full model (or shard if using tensor_parallel to split weights locally).
          - If tensor_parallel enabled, gather logits across peers.
          - Compute CE, backward, clip.
          - If tensor_parallel enabled, all-reduce grads across peers.
          - Build grads dict {param_name: grad_cpu_tensor}, send to orchestrator.
          - Optimizer step locally.
          - Send metrics and CONTROL_ACK.
        """
        step = int(msg.step if msg.step is not None else -1)
        phase = msg.phase or "train"
        batch = msg.payload or {}
        ack_required = bool((msg.metadata or {}).get("ack_required", False))

        # to tensors
        input_ids = torch.tensor(
            batch["input_ids"],
            dtype=torch.long,
            device=self.worker.device,
        )
        labels_list = batch.get("labels")
        labels = (
            torch.tensor(labels_list, dtype=torch.long, device=self.worker.device)
            if labels_list is not None
            else None
        )

        self.worker.set_training_mode(phase == "train")

        # forward -> local logits
        logits = self.worker.forward(input_ids)

        # --- Tensor Parallel Gather (if enabled) ---
        if getattr(self.cfg, "enable_tensor_parallel", False):
            logits = self.tensor_gather(logits, step)

        # CE loss + backward
        loss_val = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            loss_val = float(loss.item())

            if phase == "train":
                loss.backward()

                # clip grads
                if getattr(self.cfg, "max_grad_norm", 0) and self.cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.worker.model.parameters(),
                        self.cfg.max_grad_norm,
                    )

                # --- Tensor Parallel Gradient Reduce ---
                if getattr(self.cfg, "enable_tensor_parallel", False):
                    for _, param in self.worker.model.named_parameters():
                        if param.grad is not None:
                            param.grad = self.tensor_reduce_grad(param.grad, step)

        # gather grads
        grads_list = self.worker.get_gradients()
        grad_dict: Dict[str, torch.Tensor] = {}
        for (pname, param), g in zip(self.worker.model.named_parameters(), grads_list):
            grad_dict[pname] = (g.detach().cpu() if g is not None
                                else torch.zeros_like(param).cpu())

        # upload grads to orchestrator (+2)
        self._send_gradients(grad_dict, step)

        # local optimizer step
        self.worker.update_weights()

        # send metrics
        metrics = {
            "gpu_id": self.gpu_id,
            "step": step,
            "phase": phase,
            "loss": loss_val,
            "timestamp": time.time(),
        }
        self._send_metrics(metrics)

        # CONTROL_ACK barrier
        if ack_required:
            self._send_command_ack("CONTROL_DATA_PARALLEL_STEP", step)

    # -------------------------------------------------------------------------
    # PIPELINE/MODEL-PARALLEL: PHASE 1
    # -------------------------------------------------------------------------
    def _handle_pipeline_phase1(self, msg: Message) -> None:
        """
        Phase1 (forward kickoff):
          - Stage0: receives input_ids, does forward(), produces activations.
          - Intermediate stages: may just ACK now and then wait for ACTIVATION_FRAME.
          - Any stage that *does* run forward and is NOT final shard
            should forward activations to next stage. (Future: via orchestrator
            ACTIVATION_FRAME relay; currently not fully implemented.)
          - Final shard can cache logits for CE in phase2.
        """
        step = int(msg.step if msg.step is not None else -1)
        phase = msg.phase or "train"
        batch = msg.payload or {}
        ack_required = bool((msg.metadata or {}).get("ack_required", False))

        self.worker.set_training_mode(phase == "train")

        # Only first shard actually gets tokens directly
        input_ids_list = batch.get("input_ids", None)

        activations = None
        if input_ids_list is not None:
            input_ids = torch.tensor(
                input_ids_list,
                dtype=torch.long,
                device=self.worker.device,
            )
            activations = self.worker.forward(input_ids)

            # If tensor-parallel & we are the last shard (produces logits), gather to full logits
            start_layer, end_layer = self._layer_assignment()
            last_end = self.cfg.n_layers
            if end_layer == last_end and getattr(self.cfg, "enable_tensor_parallel", False):
                activations = self.tensor_gather(activations, step)

        # cache logits if we're last shard
        start_layer, end_layer = self._layer_assignment()
        last_end = self.cfg.n_layers
        if activations is not None and end_layer == last_end:
            # final shard -> activations are final logits
            self._last_logits = activations

        # (Future) activation relay to next shard via ACTIVATION_FRAME.

        if ack_required:
            self._send_command_ack("CONTROL_PIPELINE_PHASE1", step)

    # -------------------------------------------------------------------------
    # PIPELINE/MODEL-PARALLEL: PHASE 2 (labels on LAST stage)
    # -------------------------------------------------------------------------
    def _handle_pipeline_phase2(self, msg: Message) -> None:
        """
        Phase2:
          - ONLY final shard cares about labels.
          - Use cached self._last_logits from phase1 or ACTIVATION_FRAME.
          - If tensor-parallel, ensure cached logits are gathered.
          - Compute CE, backward(), clip, TP-grad-reduce, prepare grad_dict.
          - Upload grads (GRADIENTS_UPLOAD) to orchestrator.
          - Optimizer step.
          - Metrics + CONTROL_ACK.
        """
        step = int(msg.step if msg.step is not None else -1)
        phase = msg.phase or "train"
        batch = msg.payload or {}
        ack_required = bool((msg.metadata or {}).get("ack_required", False))

        last_gpu = self.cfg.num_workers - 1
        if self.gpu_id != last_gpu:
            if ack_required:
                self._send_command_ack("CONTROL_PIPELINE_PHASE2", step)
            return

        self.worker.set_training_mode(phase == "train")

        labels_list = batch.get("labels", None)
        if labels_list is None:
            self._send_metrics(
                {"gpu_id": self.gpu_id, "step": step, "phase": phase, "loss": None, "timestamp": time.time()}
            )
            if ack_required:
                self._send_command_ack("CONTROL_PIPELINE_PHASE2", step)
            return

        if self._last_logits is None:
            if ack_required:
                self._send_command_ack("CONTROL_PIPELINE_PHASE2", step)
            return

        # If tensor-parallel was used, ensure logits are full-gathered (already done in phase1 if last shard)
        if getattr(self.cfg, "enable_tensor_parallel", False):
            # No-op if already gathered; safe to call.
            self._last_logits = self.tensor_gather(self._last_logits, step)

        labels = torch.tensor(
            labels_list,
            dtype=torch.long,
            device=self.worker.device,
        )

        logits = self._last_logits  # [B,T,V]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        if phase == "train":
            loss.backward()

            # clip grads
            if getattr(self.cfg, "max_grad_norm", 0) and self.cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.worker.model.parameters(),
                    self.cfg.max_grad_norm,
                )

            # Tensor-parallel grad all-reduce (per-parameter)
            if getattr(self.cfg, "enable_tensor_parallel", False):
                for _, param in self.worker.model.named_parameters():
                    if param.grad is not None:
                        param.grad = self.tensor_reduce_grad(param.grad, step)

            # build gradient dict
            grad_dict: Dict[str, torch.Tensor] = {}
            for pname, param in self.worker.model.named_parameters():
                grad_dict[pname] = (
                    param.grad.detach().cpu() if param.grad is not None
                    else torch.zeros_like(param).cpu()
                )

            # upload grads (only final stage uploads in pipeline-mode)
            self._send_gradients(grad_dict, step)

            # local optimizer step
            self.worker.update_weights()

        loss_val = float(loss.item())

        # metrics
        self._send_metrics(
            {"gpu_id": self.gpu_id, "step": step, "phase": phase, "loss": loss_val, "timestamp": time.time()}
        )

        if ack_required:
            self._send_command_ack("CONTROL_PIPELINE_PHASE2", step)

        # clear logits cache for next global step
        self._last_logits = None

    # -------------------------------------------------------------------------
    # ACTIVATION_FRAME (future pipeline relay)
    # -------------------------------------------------------------------------
    def _handle_activation_frame(self, msg: Message) -> None:
        """
        Placeholder for activation relay logic if orchestrator
        ever routes ACTIVATION_FRAME messages between shards.

        Expected msg.payload:
            {
              "from_gpu": int,
              "to_gpu":   int,
              "tensor_payload": { "tensor_np": ..., "shape": ..., "dtype": ... }
            }

        Steps:
          1. Rebuild activation tensor on this GPU.
          2. Run forward() through our shard.
          3. If not final shard -> forward again to next GPU (future).
          4. If final shard -> store logits in self._last_logits.
          5. Send CONTROL_ACK to orchestrator so it knows we consumed it.
        """
        step = int(msg.step if msg.step is not None else -1)
        payload = msg.payload or {}

        tensor_payload = payload.get("tensor_payload")
        if tensor_payload is None:
            self._send_command_ack("ACTIVATION_FRAME", step)
            return

        # rebuild activation on this worker's device
        act = MessageProtocol.unwrap_tensor_payload(
            tensor_payload,
            device=self.worker.device,
        )

        out = self.worker.forward(act)

        # final shard caches logits
        start_layer, end_layer = self._layer_assignment()
        last_end = self.cfg.n_layers
        if end_layer == last_end:
            self._last_logits = out

        # (future): forward 'out' again to downstream shard here...

        # ACK back so orchestrator knows we consumed the frame
        self._send_command_ack("ACTIVATION_FRAME", step)
