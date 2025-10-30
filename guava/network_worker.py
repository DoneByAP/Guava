"""
network_worker.py

NetworkWorker is the runtime that lives on each GPU box.

It does 3 jobs:
1. Connect to the orchestrator over all required sockets.
2. Receive step commands ("train_data_parallel", "train_phase1", etc.).
3. Drive a local worker (DataParallelWorker or ModelShardWorker) to actually run
   forward/backward/optim on this GPU.

This file glues your BaseWorker/DataParallelWorker/ModelShardWorker into the
wire protocol the Orchestrator speaks.
"""

import os, socket, struct, pickle, time, zlib, threading
from typing import Any, Dict, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DistributedConfig
from .socket_utils import optimize_socket_for_network
from .base_worker import DataParallelWorker, ModelShardWorker


class NetworkWorker:
    """
    High-level network runtime for ONE GPU.

    Typical launcher on a worker machine:

        cfg = DistributedConfig.from_env()
        nw = NetworkWorker(
            gpu_id=0,
            config=cfg,
            model_ctor=lambda: MyTransformer(cfg.vocab_size, cfg.d_model),
            master_ip=os.environ["MASTER_IP"],
            master_port=int(os.environ["MASTER_PORT"]),
        )
        nw.connect_and_train()

    Notes:
    - model_ctor MUST build the *full model* (we'll shard if needed).
    - For data_parallel: every GPU gets full model.
    - For model_parallel: we slice the layer range and wrap that slice in ModelShardWorker.
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
        self.model_ctor = model_ctor  # function that returns nn.Module

        # sockets
        self.ctrl_sock: Optional[socket.socket] = None        # +0 commands in/out
        self.metric_sock_addr: Tuple[str,int] = (master_ip, master_port + 1)
        self.grad_sock_addr:   Tuple[str,int] = (master_ip, master_port + 2)
        self.act_sock_addr:    Tuple[str,int] = (master_ip, master_port + 3)
        self.ack_sock_addr:    Tuple[str,int] = (master_ip, master_port + 4)
        self.resend_sock_addr: Tuple[str,int] = (master_ip, master_port + 5)
        self.cmd_ack_addr:     Tuple[str,int] = (master_ip, master_port + 6)
        self.chkpt_sock_addr:  Tuple[str,int] = (master_ip, master_port + 7)

        # which slice of layers does this gpu own?
        start_layer, end_layer = self._layer_assignment()

        # create the underlying compute worker
        full_model = self.model_ctor()

        if self.cfg.data_parallel and not self.cfg.model_parallel:
            self.worker = DataParallelWorker(self.gpu_id, self.cfg)
            self.worker.register_model(full_model)

        else:
            # MODEL / PIPELINE PARALLEL
            # we expect that the model's transformer stack is something like
            # model.transformer_blocks or model.layers that we can slice.
            shard_model = self._extract_shard(full_model, start_layer, end_layer)
            self.worker = ModelShardWorker(
                self.gpu_id,
                self.cfg,
                layer_start=start_layer,
                layer_end=end_layer,
            )
            self.worker.register_model(shard_model)

        self.worker.set_training_mode(True)

    # -------------------------------------------------------------------------
    # layer assignment and model slicing
    # -------------------------------------------------------------------------
    def _layer_assignment(self) -> Tuple[int, int]:
        """
        Derive [start_layer, end_layer) for this GPU.

        Data-parallel: full stack.
        Model-parallel: use cfg.layers_per_gpu.
        """
        if self.cfg.data_parallel and not self.cfg.model_parallel:
            return (0, self.cfg.n_layers)

        # model-parallel split:
        start = sum(self.cfg.layers_per_gpu[: self.gpu_id])
        end = start + self.cfg.layers_per_gpu[self.gpu_id]
        return (start, end)

    def _extract_shard(self, full_model: nn.Module, start: int, end: int) -> nn.Module:
        """
        Build a NEW nn.Module that only contains [start:end] transformer layers.

        Assumptions:
        - full_model has:
            full_model.embedding (token+pos embed)
            full_model.transformer_blocks (nn.ModuleList of blocks)
            full_model.ln_f / output head on last shard
        - We shard only transformer_blocks. The first shard keeps embeddings.
          The last shard keeps ln_f + lm_head.

        This matches the GPT-like example in your docs. If your actual model
        uses different field names, edit here and that's it.
        """
        class ShardModule(nn.Module):
            def __init__(self, parent, start_idx, end_idx, is_first, is_last):
                super().__init__()
                self.is_first = is_first
                self.is_last = is_last

                # we copy refs to submodules, not deep copy; parameters stay same
                # NOTE: in true production you'd deep-copy or move slices.
                if is_first:
                    self.embedding = parent.embedding
                    self.pos_embedding = getattr(parent, "pos_embedding", None)
                else:
                    self.embedding = None
                    self.pos_embedding = None

                # slice of transformer blocks
                blocks = parent.transformer_blocks[start_idx:end_idx]
                self.transformer_blocks = nn.ModuleList(blocks)

                if is_last:
                    self.ln_f = parent.ln_f
                    self.lm_head = parent.lm_head
                else:
                    self.ln_f = None
                    self.lm_head = None

            def forward(self, hidden_ids):
                """
                If first shard: hidden_ids is token IDs -> embed.
                Else: hidden_ids are already activations from prev shard.
                """
                x = hidden_ids
                if self.is_first:
                    # hidden_ids are token IDs here
                    input_ids = hidden_ids
                    seq_len = input_ids.size(1)
                    pos = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
                    tok = self.embedding(input_ids)
                    if self.pos_embedding is not None:
                        tok = tok + self.pos_embedding(pos)
                    x = tok

                # run our block slice
                for block in self.transformer_blocks:
                    x = block(x)

                if self.is_last:
                    x = self.ln_f(x)
                    x = self.lm_head(x)
                return x

        is_first = (start == 0)
        is_last = (end == self.cfg.n_layers)
        shard = ShardModule(
            parent=full_model,
            start_idx=start,
            end_idx=end,
            is_first=is_first,
            is_last=is_last,
        )
        return shard

    # -------------------------------------------------------------------------
    # socket helpers
    # -------------------------------------------------------------------------
    def _connect_ctrl(self) -> None:
        """
        Connect primary control socket to orchestrator (port +0).
        Send registration info.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        optimize_socket_for_network(s)
        s.connect((self.master_ip, self.master_port + 0))
        self.ctrl_sock = s

        start_layer, end_layer = self._layer_assignment()

        hello = {
            "gpu_id": self.gpu_id,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "hostname": socket.gethostname(),
        }
        data = pickle.dumps(hello)
        s.send(struct.pack("!I", len(data)))
        s.sendall(data)

        # wait for ack
        ack_size = struct.unpack("!I", self._recv_exact(s, 4))[0]
        ack_data = self._recv_exact(s, ack_size)
        ack = pickle.loads(ack_data)
        if ack.get("status") != "registered":
            raise RuntimeError("registration refused by orchestrator")

    def _send_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        One-shot connect to metrics (+1) and send pickled metrics.
        """
        try:
            ms = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(ms)
            ms.connect(self.metric_sock_addr)
            blob = pickle.dumps(metrics)
            ms.send(struct.pack("!I", len(blob)))
            ms.sendall(blob)
            ms.close()
        except Exception:
            # metrics failures are non-fatal
            pass

    def _send_gradients(self, grads: Dict[str, torch.Tensor]) -> None:
        """
        Upload gradient dict to orchestrator (+2).
        Layout matches orchestrator._gradient_loop() expectations.
        """
        try:
            gs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(gs)
            gs.connect(self.grad_sock_addr)

            msg = {
                "gpu_id": self.gpu_id,
                "gradients": grads,
            }
            raw = pickle.dumps(msg)
            comp = zlib.compress(raw)
            header = struct.pack("!Q?", len(comp), True)

            gs.send(header)
            gs.sendall(comp)
            gs.close()
        except Exception as e:
            # gradient failures are more serious but we still won't hard-crash;
            # orchestrator may log missing grads for that step.
            print(f"[GPU {self.gpu_id}] gradient upload failed: {e}")

    def _send_command_ack(self, cmd_type: str, step: int) -> None:
        """
        Send ACK for command barriers to orchestrator (+6).
        This unblocks orchestrator._wait_for_cmd_acks(...)
        """
        try:
            cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(cs)
            cs.connect(self.cmd_ack_addr)
            ack = {
                "gpu_id": self.gpu_id,
                "cmd_type": cmd_type,
                "step": step,
            }
            blob = pickle.dumps(ack)
            cs.send(struct.pack("!I", len(blob)))
            cs.sendall(blob)
            cs.close()
        except Exception:
            pass

    def _send_activation_ack(self, phase: str, step: int, dst: int) -> None:
        """
        Send pipeline activation ACK to orchestrator (+4).
        This unblocks orchestrator._wait_until_last_gpu_has(...)
        """
        try:
            asock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(asock)
            asock.connect(self.ack_sock_addr)
            ack = {
                "gpu_id": self.gpu_id,
                "dst": dst,
                "phase": phase,
                "step": step,
            }
            blob = pickle.dumps(ack)
            asock.send(struct.pack("!I", len(blob)))
            asock.sendall(blob)
            asock.close()
        except Exception:
            pass

    def _send_activation_frame_downstream(self, activation: torch.Tensor, phase: str, step: int, dst_gpu: int) -> None:
        """
        Send activations to orchestrator so it can forward to dst_gpu (+3).
        This matches orchestrator._activation_uplink_loop() contract.
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(s)
            s.connect(self.act_sock_addr)

            # Serialize activation tensor (cpu list for portability)
            act_payload = pickle.dumps({
                "tensor": activation.detach().cpu().numpy(),
                "dtype": str(activation.dtype),
                "shape": list(activation.shape),
            })
            # Build header:
            #   src:i16 dst:i16 phase_len:u16 phase:bytes step:i32 payload_len:u32 payload
            phase_b = phase.encode("utf-8")
            header = struct.pack("!hhH", self.gpu_id, dst_gpu, len(phase_b))
            step_hdr = struct.pack("!i", step)
            pay_hdr = struct.pack("!I", len(act_payload))

            s.sendall(header)
            s.sendall(phase_b)
            s.sendall(step_hdr)
            s.sendall(pay_hdr)
            s.sendall(act_payload)
            s.close()

            # tell orchestrator we delivered (the orchestrator will in turn relay and ACK later)
            self._send_activation_ack(phase, step, dst_gpu)

        except Exception as e:
            print(f"[GPU {self.gpu_id}] activation uplink failed: {e}")

    # -------------------------------------------------------------------------
    # main loop
    # -------------------------------------------------------------------------
    def connect_and_train(self) -> None:
        """
        Connect to orchestrator, then enter command loop:
          - read message
          - run step
        """
        self._connect_ctrl()

        while True:
            size_buf = self._recv_exact(self.ctrl_sock, 4)
            if not size_buf:
                break
            msg_size = struct.unpack("!I", size_buf)[0]
            raw = self._recv_exact(self.ctrl_sock, msg_size)
            if not raw:
                break

            cmd = pickle.loads(raw)
            ctype = cmd.get("command")

            if ctype == "start_training":
                # Optionally reset optim stats, etc.
                continue

            if ctype == "stop_training":
                # Save local checkpoint chunk on exit if you want
                self._final_checkpoint()
                break

            if ctype == "train_data_parallel":
                self._handle_train_data_parallel(cmd)

            elif ctype == "train_phase1":
                self._handle_train_phase1(cmd)

            elif ctype == "train_phase2_labels":
                self._handle_train_phase2_labels(cmd)

            elif ctype == "activation_frame":
                self._handle_activation_frame(cmd)

            else:
                # unknown or not implemented
                pass

        # cleanup
        try:
            self.ctrl_sock.close()
        except Exception:
            pass
        self.worker.cleanup()

    # -------------------------------------------------------------------------
    # DP path
    # -------------------------------------------------------------------------
    def _handle_train_data_parallel(self, cmd: Dict[str, Any]) -> None:
        """
        For data parallel:
          - batch['input_ids'], batch['labels'] are lists -> turn into tensors
          - forward full model
          - compute cross-entropy loss
          - backward()
          - clip grads
          - send gradients to orchestrator (+2)
          - optimizer.step()
          - ACK command (+6)
          - send metrics (+1)

        After orchestrator aggregates gradients, it *can* decide to broadcast
        updated weights back to all workers (not yet implemented here).
        """
        step = cmd["step"]
        phase = cmd.get("phase", "train")
        batch = cmd["batch"]

        # tensors on correct device
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=self.worker.device)
        labels = None
        if batch.get("labels") is not None:
            labels = torch.tensor(batch["labels"], dtype=torch.long, device=self.worker.device)

        self.worker.set_training_mode(phase == "train")

        # forward
        logits = self.worker.forward(input_ids)

        # compute CE and backward
        loss_val = None
        if labels is not None:
            loss_val = self.worker.compute_loss_and_backward(logits, labels)

        # gradient clipping already handled in compute_loss_and_backward()
        # gather grads for orchestrator to average
        grads_list = self.worker.get_gradients()

        # convert gradient list -> dict{name: tensor}
        # we need param names so orchestrator can stack/mean by key
        grad_dict = {}
        for (name, param), g in zip(
            self.worker.model.named_parameters(),
            grads_list
        ):
            if g is not None:
                grad_dict[name] = g.detach().cpu()
            else:
                grad_dict[name] = torch.zeros_like(param).cpu()

        # send gradients to +2 for aggregation across GPUs
        self._send_gradients(grad_dict)

        # after orchestrator averages & (eventually) pushes updated params
        # we still locally step so this worker makes progress immediately
        # NOTE: in strict data-parallel you'd wait for synced grads first.
        self.worker.update_weights()

        # send metrics
        metrics = {
            "gpu_id": self.gpu_id,
            "step": step,
            "phase": phase,
            "loss": float(loss_val) if loss_val is not None else None,
            "timestamp": time.time(),
        }
        self._send_metrics(metrics)

        # send command ACK so orchestrator can unblock
        if cmd.get("ack_required"):
            self._send_command_ack("train_data_parallel", step)

    # -------------------------------------------------------------------------
    # Model/Pipeline parallel PHASE 1
    # -------------------------------------------------------------------------
    def _handle_train_phase1(self, cmd: Dict[str, Any]) -> None:
        """
        phase1:
         - gpu0 receives input_ids, runs its shard forward → activations
         - intermediate GPUs receive NO input_ids (they just prep)
         - after forward, each shard sends activations downstream via orchestrator
           using _send_activation_frame_downstream()
         - each shard also ACKs the command to unblock orchestrator barrier
        """
        step = cmd["step"]
        phase = cmd.get("phase", "train")
        batch = cmd.get("batch", {})

        self.worker.set_training_mode(phase == "train")

        if "input_ids" in batch and batch["input_ids"] is not None:
            # We're the first shard in pipeline (gpu0)
            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=self.worker.device)
            activations = self.worker.forward(input_ids)
        else:
            # Not first shard. We'll wait for activation_frame later.
            activations = None

        # If we produced activations and we're NOT the last shard,
        # push downstream via orchestrator.
        start_layer, end_layer = self._layer_assignment()
        last_gpu = self.cfg.num_workers - 1
        if activations is not None and end_layer < self.cfg.n_layers:
            dst_gpu = self.gpu_id + 1
            self._send_activation_frame_downstream(
                activation=activations,
                phase=phase,
                step=step,
                dst_gpu=dst_gpu
            )

        # ACK this command so orchestrator knows phase1 was received
        if cmd.get("ack_required"):
            self._send_command_ack("train_phase1", step)

    # -------------------------------------------------------------------------
    # PHASE 2 (labels on last GPU)
    # -------------------------------------------------------------------------
    def _handle_train_phase2_labels(self, cmd: Dict[str, Any]) -> None:
        """
        Only the LAST GPU in the pipeline does this.
        It should now have final logits cached from _handle_activation_frame.
        We compute CE loss -> backward -> clip -> optimizer.step -> metrics.

        cmd['batch']['labels'] may be None for unsupervised.
        """
        step = cmd["step"]
        phase = cmd.get("phase", "train")
        batch = cmd["batch"]
        labels_list = batch.get("labels", None)

        # only last GPU cares. if we're not last, ignore.
        last_gpu = self.cfg.num_workers - 1
        if self.gpu_id != last_gpu:
            return

        if labels_list is None:
            # unsupervised case
            self.worker.set_training_mode(phase == "train")
            # no backward / no metrics to send except maybe step marker
            self._send_metrics({
                "gpu_id": self.gpu_id,
                "step": step,
                "phase": phase,
                "loss": None,
                "timestamp": time.time()
            })
            return

        # we expect we cached our final logits from activation_frame path
        if not hasattr(self, "_last_logits"):
            # we didn't receive activations? skip
            return

        labels = torch.tensor(labels_list, dtype=torch.long, device=self.worker.device)

        # Compute CE manually since DataParallelWorker.compute_loss_and_backward
        # assumes it owns full model. Here, shard is last stage -> we have logits.
        logits = self._last_logits  # [B, T, vocab]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        if phase == "train":
            loss.backward()
            # clip and step
            if self.cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.worker.model.parameters(),
                    self.cfg.max_grad_norm
                )
            self.worker.update_weights()

        loss_val = float(loss.item())
        self._send_metrics({
            "gpu_id": self.gpu_id,
            "step": step,
            "phase": phase,
            "loss": loss_val,
            "timestamp": time.time()
        })

    # -------------------------------------------------------------------------
    # ACTIVATION FRAME FROM PREV GPU
    # -------------------------------------------------------------------------
    def _handle_activation_frame(self, cmd: Dict[str, Any]) -> None:
        """
        Orchestrator relays activations here.
        cmd format:
          {
            "command": "activation_frame",
            "from": <src_gpu_id>,
            "phase": str,
            "step": int,
            "payload": <pickled {tensor: np.array, dtype:..., shape:[...]}>
          }

        We:
         - unpack the tensor
         - run self.worker.forward(activations)
         - if we're NOT last shard: send downstream
         - if we're last shard: store logits for phase2 loss calc
         - ACK to orchestrator (+4) so it unblocks its barrier
        """
        phase = cmd["phase"]
        step = cmd["step"]
        payload = pickle.loads(cmd["payload"])

        np_arr = payload["tensor"]
        act_tensor = torch.tensor(np_arr, device=self.worker.device)

        # run forward on our shard
        out = self.worker.forward(act_tensor)

        start_layer, end_layer = self._layer_assignment()
        last_layer_end = self.cfg.n_layers
        last_gpu = self.cfg.num_workers - 1

        if end_layer < last_layer_end:
            # not last shard, forward to next shard
            dst_gpu = self.gpu_id + 1
            self._send_activation_frame_downstream(
                activation=out,
                phase=phase,
                step=step,
                dst_gpu=dst_gpu
            )
        else:
            # last shard: 'out' are final logits
            self._last_logits = out

        # tell orchestrator we got it
        self._send_activation_ack(phase=phase, step=step, dst=self.gpu_id)

    # -------------------------------------------------------------------------
    # checkpoint upload on shutdown
    # -------------------------------------------------------------------------
    def _final_checkpoint(self) -> None:
        """
        Send our model weights (state_dict) to orchestrator (+7).
        This allows orchestrator to persist shard/replica weights.
        """
        try:
            state = self.worker.get_model_state()
            blob = pickle.dumps(state)
            comp = zlib.compress(blob)

            meta = {
                "gpu_id": self.gpu_id,
                "filename": f"worker{self.gpu_id}_final.pt",
                "original_size": len(blob),
                "compressed_size": len(comp),
                "hostname": socket.gethostname(),
            }

            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            optimize_socket_for_network(s)
            s.connect(self.chkpt_sock_addr)

            meta_bin = pickle.dumps(meta)
            s.send(struct.pack("!I", len(meta_bin)))
            s.sendall(meta_bin)

            s.send(struct.pack("!Q", len(comp)))
            s.sendall(comp)

            # read 1-byte success flag
            ok = s.recv(1)
            s.close()

            if ok == b"\x01":
                print(f"[GPU {self.gpu_id}] checkpoint uploaded OK")
            else:
                print(f"[GPU {self.gpu_id}] checkpoint upload FAIL")
        except Exception as e:
            print(f"[GPU {self.gpu_id}] checkpoint exception: {e}")

    # -------------------------------------------------------------------------
    # tiny helper
    # -------------------------------------------------------------------------
    def _recv_exact(self, conn: socket.socket, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf
