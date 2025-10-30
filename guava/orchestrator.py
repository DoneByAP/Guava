"""
orchestrator.py

Central coordinator ("master" / "brain").

Responsibilities:
- Listen for worker connections
- Send commands/batches to workers (data-parallel or pipeline-parallel)
- Receive gradients / metrics / checkpoints
- Run the top-level training loop
- Coordinate full multi-stage backward across all pipeline shards

Network contract (all TCP):
- Control / registration / ACKs / BACKWARD_READY : master_port+0 (long-lived socket per worker)
- Metrics upload:                                 master_port+1 (short-lived per send)
- Gradients upload:                               master_port+2 (short-lived per send)
- Activation uplink / relay:                      master_port+3 (reserved / future)
- Activation ACK / heartbeat:                     master_port+4 (reserved / future)
- Resend probe / heartbeat:                       master_port+5 (reserved / future)
- Command ACK legacy port:                        master_port+6 (unused now; ACKs come on +0)
- Checkpoints upload:                             master_port+7 (short-lived per send)

All control messages use MessageProtocol and Message/MessageType.
We run on CPU and act as the source of truth for weights.
"""

import os
import socket
import threading
import time
from typing import Any, Dict, Optional, List, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DistributedConfig
from .socket_utils import optimize_socket_for_network
from .protocol import MessageType, Message, MessageProtocol


class Orchestrator:
    """
    Orchestrator holds:
    - Authoritative full model weights
    - Optimizer
    - Training scheduler for data-parallel or pipeline-parallel
    - Control sockets for each worker
    - Gradient + metric buffers
    - BACKWARD_READY queues for upstream grad handoff between shards

    We now support TRUE multi-stage backward for pipeline/model-parallel:
    - Last shard computes CE loss, backward(), uploads grads, and sends BACKWARD_READY
      containing upstream_grad for previous shard.
    - Orchestrator walks upstream shard-by-shard with CONTROL_PIPELINE_BACKWARD,
      feeding that upstream_grad back, causing each shard to backward() and upload
      its own grads and produce the next upstream_grad.
    - After all shards have contributed grads, we average all and step once.
    """

    def __init__(self, config: DistributedConfig, mode: str = "orchestrator"):
        self.cfg = config
        self.mode = mode

        # Master model + optimizer
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Global training position
        self.global_step = 0
        self.epoch_idx = 0

        # Thread/shutdown control
        self._listener_threads: List[threading.Thread] = []
        self._shutdown_flag = threading.Event()

        # Worker registry:
        #   gpu_id -> { "gpu_id":..., "start_layer":..., "end_layer":..., "hostname":... }
        self.registered_workers: Dict[int, Dict[str, Any]] = {}

        # Long-lived control sockets:
        #   gpu_id -> socket
        self.ctrl_sockets: Dict[int, socket.socket] = {}

        # Gradient buffer (grad_server fills this, training loop drains it)
        #   gpu_id -> { param_name: grad_tensor_cpu }
        self.gradients_lock = threading.Lock()
        self.collected_gradients: Dict[int, Dict[str, torch.Tensor]] = {}

        # Metrics buffer (metrics_server fills this, training loop drains it)
        self.metrics_lock = threading.Lock()
        self.collected_metrics: List[Dict[str, Any]] = []

        # BACKWARD_READY queues:
        # Each worker sends BACKWARD_READY on its ctrl socket when it finishes backward().
        # We store those here so pipeline scheduler can pop them in order.
        #   gpu_id -> [Message, Message, ...]
        self._backward_lock = threading.Lock()
        self._backward_ready_queues: Dict[int, List[Message]] = {}

        # Bring up background listener servers (+0 control, +1 metrics, +2 grads, +7 checkpoints)
        self._start_listener_threads()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def register_model(self, model: nn.Module) -> None:
        """
        Attach the authoritative (full) model and create the optimizer.
        """
        self.model = model
        self.model.train(True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def wait_for_workers(
        self,
        expected: Optional[int] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1,
    ) -> None:
        """
        Block until N workers are registered and control sockets are live.

        Args:
            expected: how many workers we want (default cfg.num_workers)
            timeout: seconds, None = wait forever
            poll_interval: sleep step between polls

        Raises:
            TimeoutError if not enough workers register in time.
        """
        target = expected if expected is not None else self.cfg.num_workers
        start_ts = time.time()
        while len(self.registered_workers) < target:
            if timeout is not None and (time.time() - start_ts) > timeout:
                raise TimeoutError(
                    f"Timed out waiting for {target} workers; "
                    f"only {len(self.registered_workers)} registered"
                )
            time.sleep(poll_interval)

    def start_training(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        num_epochs: int = 1,
        val_interval: int = 100,
    ) -> None:
        """
        Main training driver.

        If cfg.pipeline_parallel or cfg.model_parallel:
            we use pipeline-style scheduler with multi-stage backward.

        Else:
            we use synchronous data-parallel (all GPUs have full model).
        """
        assert self.model is not None, "call register_model(model) first"
        assert self.optimizer is not None, "optimizer not created"

        print("======================================================================")
        print("GUAVA ORCHESTRATOR TRAIN LOOP")
        print(f"num_workers: {self.cfg.num_workers}")
        print(
            "parallelism: "
            f"data={self.cfg.data_parallel}, "
            f"model={self.cfg.model_parallel}, "
            f"pipeline={self.cfg.pipeline_parallel}"
        )
        print("======================================================================")

        if self.cfg.pipeline_parallel or self.cfg.model_parallel:
            self._start_training_pipeline_parallel(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                val_interval=val_interval,
            )
        else:
            self._start_training_data_parallel(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                val_interval=val_interval,
            )

        print("Training loop finished.")

    def save_checkpoint(self, path: str) -> None:
        """
        Save orchestrator's authoritative model weights to disk.
        """
        if self.model is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[orchestrator] checkpoint saved -> {path}")

    # ---------------------------------------------------------------------
    # Data-parallel training loop
    # ---------------------------------------------------------------------
    def _start_training_data_parallel(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable],
        num_epochs: int,
        val_interval: int,
    ) -> None:
        """
        Classic synchronous data-parallel:
          1. Broadcast batch to every worker via CONTROL_DATA_PARALLEL_STEP.
          2. Each worker runs forward+loss+backward locally.
          3. Each worker uploads grads via GRADIENTS_UPLOAD.
          4. We average and step on the master model.
        """
        for epoch in range(num_epochs):
            self.epoch_idx = epoch
            print(f"[Epoch {epoch+1}/{num_epochs}] (data-parallel)")

            for _, batch in enumerate(train_loader):
                self.global_step += 1
                input_ids, labels = batch

                # Send same batch to every worker
                step_msg = Message(
                    msg_type=MessageType.CONTROL_DATA_PARALLEL_STEP,
                    payload={
                        "input_ids": input_ids.tolist(),
                        "labels": labels.tolist(),
                    },
                    metadata={"ack_required": True},
                    step=self.global_step,
                    phase="train",
                )
                self._broadcast_ctrl(step_msg)

                # Wait for all workers to upload grads, then step
                self._aggregate_and_step(
                    pipeline_mode=False  # data-parallel: expect grads from all workers
                )

                # Drain and print worker metrics
                for m in self._drain_metrics():
                    if m.get("loss") is not None:
                        print(
                            f"[step {m['step']}] "
                            f"gpu{m['gpu_id']} loss={m['loss']:.4f}"
                        )

                # Periodic validation on orchestrator CPU model
                if val_loader is not None and (self.global_step % val_interval == 0):
                    val_loss = self._run_validation(val_loader)
                    print(f"[val @ step {self.global_step}] loss={val_loss:.4f}")

    # ---------------------------------------------------------------------
    # Pipeline/model-parallel training loop
    # ---------------------------------------------------------------------
    def _start_training_pipeline_parallel(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable],
        num_epochs: int,
        val_interval: int,
    ) -> None:
        """
        Full pipeline/model-parallel with true multi-stage backward.

        Per global step:
          1. CONTROL_PIPELINE_PHASE1 (broadcast):
               - stage0 ingest input_ids -> forward()
               - each shard runs its block forward
               - last shard caches logits
          2. CONTROL_PIPELINE_PHASE2 (send ONLY to last shard):
               - last shard computes CE loss w/ labels
               - last shard backward()
               - last shard uploads its grads
               - last shard sends BACKWARD_READY with upstream_grad
          3. Orchestrator walks upstream:
               for gid = last-1 down to 0:
                   send CONTROL_PIPELINE_BACKWARD with upstream_grad
                   that shard backward() + upload grads
                   that shard sends BACKWARD_READY with new upstream_grad
          4. Orchestrator aggregates all shard grads and steps optimizer.
        """
        for epoch in range(num_epochs):
            self.epoch_idx = epoch
            print(f"[Epoch {epoch+1}/{num_epochs}] (pipeline/model-parallel)")

            for _, batch in enumerate(train_loader):
                self.global_step += 1
                input_ids, labels = batch

                # Run ONE full pipeline step with backward chaining
                self._pipeline_single_step(
                    input_ids=input_ids,
                    labels=labels,
                    step=self.global_step,
                    phase="train",
                )

                # Metrics from all shards (loss only printed from last shard)
                for m in self._drain_metrics():
                    if m.get("loss") is not None:
                        print(
                            f"[step {m['step']}] "
                            f"gpu{m['gpu_id']} loss={m['loss']:.4f}"
                        )

                # Validation on orchestrator model every val_interval
                if val_loader is not None and (self.global_step % val_interval == 0):
                    val_loss = self._run_validation(val_loader)
                    print(f"[val @ step {self.global_step}] loss={val_loss:.4f}")

    def _pipeline_single_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        phase: str,
    ) -> None:
        """
        Execute ONE full pipeline step with upstream gradient chaining.

        Steps:
        (1) Broadcast CONTROL_PIPELINE_PHASE1 → all shards run forward slice.
        (2) Send CONTROL_PIPELINE_PHASE2 ONLY to LAST shard → CE + backward init.
        (3) Iteratively walk upstream with CONTROL_PIPELINE_BACKWARD for each
            previous shard using upstream_grad handed back in BACKWARD_READY.
        (4) After all shards uploaded GRADIENTS_UPLOAD, average & step.
        """
        last_gpu = self.cfg.num_workers - 1
        micro_batches = int(self.cfg.micro_batches)

        # ----------------- Phase 1: forward across shards -----------------
        phase1_msg = Message(
            msg_type=MessageType.CONTROL_PIPELINE_PHASE1,
            payload={
                # Only GPU0 (first shard) truly consumes these input_ids.
                # Others just ACK / prep forward cache.
                "input_ids": input_ids.tolist(),
            },
            metadata={
                "ack_required": True,
                "micro_batches": micro_batches,
            },
            step=step,
            phase=phase,
        )
        self._broadcast_ctrl(phase1_msg)

        # ----------------- Phase 2: start backward on LAST shard ----------
        # Only final shard needs labels to compute CE + backward().
        phase2_msg = Message(
            msg_type=MessageType.CONTROL_PIPELINE_PHASE2,
            payload={
                "labels": labels.tolist(),
            },
            metadata={
                "ack_required": True,
                "micro_batches": micro_batches,
            },
            step=step,
            phase=phase,
        )
        self._send_ctrl_to(last_gpu, phase2_msg)

        # Last shard will:
        #   - do CE + backward
        #   - clip grads
        #   - upload its grads via GRADIENTS_UPLOAD (+2)
        #   - send BACKWARD_READY over ctrl socket containing upstream_grad
        upstream_grad = self._wait_backward_ready(gpu_id=last_gpu, step=step)

        # ----------------- Backward chain: walk upstream ------------------
        # For gid = last-1 down to 0:
        # Each shard receives CONTROL_PIPELINE_BACKWARD with upstream_grad,
        # runs backward() through its cached activation,
        # uploads grads,
        # then returns BACKWARD_READY w/ grad for next earlier shard.
        for gid in range(last_gpu - 1, -1, -1):
            back_msg = Message(
                msg_type=MessageType.CONTROL_PIPELINE_BACKWARD,
                payload={
                    "upstream_grad": upstream_grad,
                },
                metadata={"ack_required": True},
                step=step,
                phase=phase,
                gpu_id=gid,
            )
            self._send_ctrl_to(gid, back_msg)

            # Wait for that shard to finish and give us the next upstream_grad
            upstream_grad = self._wait_backward_ready(gpu_id=gid, step=step)

        # At this point:
        #   - EVERY shard has uploaded its GRADIENTS_UPLOAD for this global step
        # Now we aggregate everyone's grads and take a single optimizer step.
        self._aggregate_and_step(
            pipeline_mode=True  # expect grads from ALL shards
        )

    # ---------------------------------------------------------------------
    # Validation (on orchestrator CPU copy)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _run_validation(self, val_loader: Iterable) -> float:
        """
        Simple validation on the orchestrator model.
        Computes average next-token cross-entropy.
        """
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for (inp_ids, lbls) in val_loader:
            logits = self.model(inp_ids)  # [B,T,V]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                lbls.view(-1),
            )
            n_tok = lbls.numel()
            total_loss += float(loss.item()) * n_tok
            total_tokens += n_tok

        self.model.train(True)

        return 0.0 if total_tokens == 0 else (total_loss / total_tokens)

    # ---------------------------------------------------------------------
    # Background listener servers
    # ---------------------------------------------------------------------
    def _start_listener_threads(self) -> None:
        """
        Spin up background servers:
          +0 : control/register/ACKs/BACKWARD_READY  (long-lived per worker)
          +1 : metrics uploads (METRICS_STEP)
          +2 : gradient uploads (GRADIENTS_UPLOAD)
          +7 : checkpoint uploads (CHECKPOINT_SHARD_UPLOAD)

        Ports +3,+4,+5,+6 are reserved (activation relay, heartbeats, etc.).
        """
        t0 = threading.Thread(target=self._ctrl_server, daemon=True)
        t1 = threading.Thread(target=self._metrics_server, daemon=True)
        t2 = threading.Thread(target=self._grad_server, daemon=True)
        t7 = threading.Thread(target=self._checkpoint_server, daemon=True)

        self._listener_threads = [t0, t1, t2, t7]
        for t in self._listener_threads:
            t.start()

    def _bind_listen(self, port_offset: int) -> socket.socket:
        """
        Bind and listen on (cfg.master_port + port_offset).
        """
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        optimize_socket_for_network(lsock)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((self.cfg.master_ip, self.cfg.master_port + port_offset))
        lsock.listen()
        return lsock

    # ---------------------------------------------------------------------
    # CONTROL / REGISTRATION SERVER (master_port+0)
    # ---------------------------------------------------------------------
    def _ctrl_server(self) -> None:
        """
        Accept worker control sockets on master_port+0.

        Per new worker connection:
          1. First message MUST be CONTROL_HELLO with:
             {gpu_id,start_layer,end_layer,hostname}
          2. We store socket in ctrl_sockets[gpu_id] and init BACKWARD_READY queue.
          3. We reply CONTROL_ACK(status="registered").
          4. We then poll forever for:
             - CONTROL_ACK   (barrier ACKs)
             - CONTROL_HEARTBEAT
             - BACKWARD_READY (contains upstream_grad after backward())
        """
        lsock = self._bind_listen(0)
        print(
            f"[orchestrator] ctrl_server listening on "
            f"{self.cfg.master_ip}:{self.cfg.master_port}+0"
        )

        while not self._shutdown_flag.is_set():
            try:
                conn, addr = lsock.accept()
            except OSError:
                break

            threading.Thread(
                target=self._handle_ctrl_connection,
                args=(conn, addr),
                daemon=True,
            ).start()

    def _handle_ctrl_connection(
        self,
        conn: socket.socket,
        addr: Tuple[str, int],
    ) -> None:
        """
        Manage a single worker's control socket for its lifetime.
        """
        optimize_socket_for_network(conn)

        try:
            # STEP 1: expect CONTROL_HELLO to register this worker.
            hello_msg = MessageProtocol.receive_message(conn, timeout=None)
            if hello_msg is None or hello_msg.msg_type != MessageType.CONTROL_HELLO:
                conn.close()
                return

            hello = hello_msg.payload  # {gpu_id,start_layer,end_layer,hostname}
            gpu_id = int(hello["gpu_id"])

            self.registered_workers[gpu_id] = hello
            self.ctrl_sockets[gpu_id] = conn

            # init BACKWARD_READY FIFO list for this worker
            with self._backward_lock:
                self._backward_ready_queues[gpu_id] = []

            print(f"[orchestrator] registered worker GPU {gpu_id} from {addr}")

            # STEP 2: send CONTROL_ACK(status='registered')
            ack_msg = Message(
                msg_type=MessageType.CONTROL_ACK,
                payload={"status": "registered", "gpu_id": gpu_id},
                gpu_id=gpu_id,
            )
            MessageProtocol.send_message(conn, ack_msg)

            # STEP 3: poll loop for control messages from worker
            conn.settimeout(0.1)
            while not self._shutdown_flag.is_set():
                try:
                    in_msg = MessageProtocol.receive_message(conn, timeout=0.1)

                    if in_msg is None:
                        # worker closed cleanly
                        break

                    # Barrier ACK after we broadcast a control command
                    if in_msg.msg_type == MessageType.CONTROL_ACK:
                        step = in_msg.step
                        cmd_type = (
                            in_msg.payload.get("cmd_type", "unknown")
                            if isinstance(in_msg.payload, dict)
                            else "unknown"
                        )
                        print(
                            f"[orchestrator] ACK from GPU{gpu_id} "
                            f"cmd={cmd_type} step={step}"
                        )

                    # Heartbeat (liveness ping)
                    elif in_msg.msg_type == MessageType.CONTROL_HEARTBEAT:
                        # could mark alive/last_seen here
                        pass

                    # BACKWARD_READY:
                    # Worker finished backward(), uploaded grads, and is
                    # telling us "here's upstream_grad for the previous shard".
                    elif in_msg.msg_type == MessageType.BACKWARD_READY:
                        with self._backward_lock:
                            self._backward_ready_queues[gpu_id].append(in_msg)

                    else:
                        # future message types land here
                        pass

                except socket.timeout:
                    continue
                except (ConnectionResetError, ConnectionAbortedError):
                    break

        finally:
            # We DO NOT close conn here on purpose.
            # We keep the socket as long as it's alive in ctrl_sockets[gpu_id].
            pass

    # ---------------------------------------------------------------------
    # METRICS SERVER (+1)
    # ---------------------------------------------------------------------
    def _metrics_server(self) -> None:
        """
        Workers connect, send one METRICS_STEP Message, then close.
        """
        lsock = self._bind_listen(1)
        print(
            f"[orchestrator] metrics_server listening on "
            f"{self.cfg.master_ip}:{self.cfg.master_port}+1"
        )

        while not self._shutdown_flag.is_set():
            try:
                conn, _ = lsock.accept()
            except OSError:
                break

            threading.Thread(
                target=self._handle_metrics_conn,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_metrics_conn(self, conn: socket.socket) -> None:
        try:
            msg = MessageProtocol.receive_message(conn, timeout=None)
            if msg is not None and msg.msg_type == MessageType.METRICS_STEP:
                with self.metrics_lock:
                    self.collected_metrics.append(msg.payload)
        finally:
            conn.close()

    # ---------------------------------------------------------------------
    # GRADIENT SERVER (+2)
    # ---------------------------------------------------------------------
    def _grad_server(self) -> None:
        """
        Workers connect, send one GRADIENTS_UPLOAD Message, then close.

        Message payload:
            {
                "gradients": {
                    "<param_name>": torch.Tensor (CPU),
                    ...
                }
            }
        Message.gpu_id tells us which worker sent them.
        """
        lsock = self._bind_listen(2)
        print(
            f"[orchestrator] grad_server listening on "
            f"{self.cfg.master_ip}:{self.cfg.master_port}+2"
        )

        while not self._shutdown_flag.is_set():
            try:
                conn, _ = lsock.accept()
            except OSError:
                break

            threading.Thread(
                target=self._handle_grad_conn,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_grad_conn(self, conn: socket.socket) -> None:
        try:
            msg = MessageProtocol.receive_message(conn, timeout=None)
            if msg is None or msg.msg_type != MessageType.GRADIENTS_UPLOAD:
                conn.close()
                return

            gpu_id = int(msg.gpu_id if msg.gpu_id is not None else -1)
            grads_dict = msg.payload.get("gradients", {})

            with self.gradients_lock:
                self.collected_gradients[gpu_id] = grads_dict
        finally:
            conn.close()

    # ---------------------------------------------------------------------
    # CHECKPOINT SERVER (+7)
    # ---------------------------------------------------------------------
    def _checkpoint_server(self) -> None:
        """
        Workers connect to +7 at shutdown and send CHECKPOINT_SHARD_UPLOAD.
        Orchestrator persists shard weights to disk.
        """
        lsock = self._bind_listen(7)
        print(
            f"[orchestrator] checkpoint_server listening on "
            f"{self.cfg.master_ip}:{self.cfg.master_port}+7"
        )

        while not self._shutdown_flag.is_set():
            try:
                conn, _ = lsock.accept()
            except OSError:
                break

            threading.Thread(
                target=self._handle_checkpoint_conn,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_checkpoint_conn(self, conn: socket.socket) -> None:
        try:
            msg = MessageProtocol.receive_message(conn, timeout=None)
            if msg is None or msg.msg_type != MessageType.CHECKPOINT_SHARD_UPLOAD:
                conn.close()
                return

            payload = msg.payload or {}
            gpu_id = payload.get("gpu_id", msg.gpu_id)
            filename = payload.get("filename", f"worker{gpu_id}_final.pt")
            state_dict = payload.get("state_dict")

            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
            out_path = os.path.join(self.cfg.checkpoint_dir, filename)
            torch.save(state_dict, out_path)
            print(f"[orchestrator] got checkpoint from GPU{gpu_id} -> {out_path}")

            # Best-effort single-byte "OK" so worker can log it
            try:
                conn.sendall(b"\x01")
            except Exception:
                pass

        finally:
            conn.close()

    # ---------------------------------------------------------------------
    # Training helpers
    # ---------------------------------------------------------------------
    def _broadcast_ctrl(self, msg: Message) -> None:
        """
        Send the same Message to EVERY registered worker via its control socket.
        If a socket is dead, drop that worker.
        """
        dead = []
        for gpu_id, sock in list(self.ctrl_sockets.items()):
            try:
                MessageProtocol.send_message(sock, msg, compress=True)
            except Exception as e:
                print(f"[orchestrator] failed to send cmd to GPU{gpu_id}: {e}")
                dead.append(gpu_id)

        for gpu_id in dead:
            self.ctrl_sockets.pop(gpu_id, None)
            self.registered_workers.pop(gpu_id, None)

    def _send_ctrl_to(self, gpu_id: int, msg: Message) -> None:
        """
        Send one Message to one specific worker (used for:
        - PHASE2 to last shard
        - CONTROL_PIPELINE_BACKWARD during upstream walk).
        """
        sock = self.ctrl_sockets.get(gpu_id)
        if sock is None:
            raise RuntimeError(f"No control socket for GPU{gpu_id}")
        try:
            MessageProtocol.send_message(sock, msg, compress=True)
        except Exception as e:
            raise RuntimeError(f"Failed to send ctrl msg to GPU{gpu_id}: {e}")

    def _wait_backward_ready(self, gpu_id: int, step: int) -> Any:
        """
        Block until worker `gpu_id` reports BACKWARD_READY for this `step`.

        Worker contract for BACKWARD_READY:
            msg.payload = {
                "upstream_grad": <picklable array-like or None>
            }

        Returns:
            upstream_grad to feed into the PREVIOUS shard. None for shard0.
        """
        while True:
            with self._backward_lock:
                q = self._backward_ready_queues.get(gpu_id, [])
                # look for matching global step
                for i, m in enumerate(q):
                    if m.step == step:
                        msg = q.pop(i)
                        upstream_grad = None
                        if isinstance(msg.payload, dict):
                            upstream_grad = msg.payload.get("upstream_grad", None)
                        return upstream_grad
            time.sleep(0.01)

    def _expected_grad_senders(self, pipeline_mode: bool) -> int:
        """
        How many workers should upload gradients for this global step?

        - Data-parallel: all workers upload grads.
        - Pipeline/model-parallel with full backward: EVERY shard uploads
          its local param grads after its backward() call.

        In both cases now, it's num_workers.
        """
        return max(self.cfg.num_workers, 1)

    def _aggregate_and_step(self, pipeline_mode: bool = False) -> None:
        """
        Gradient sync + optimizer step:
          1. Wait for expected GRADIENTS_UPLOAD messages.
          2. Average grads across workers per param name.
          3. Load averaged grads into master model.
          4. optimizer.step() on orchestrator.
        """
        assert self.model is not None
        assert self.optimizer is not None

        expected_workers = self._expected_grad_senders(pipeline_mode)

        # Wait until we have ALL expected gradient uploads
        while True:
            with self.gradients_lock:
                ready = len(self.collected_gradients) >= expected_workers
            if ready:
                break
            time.sleep(0.01)

        # Snapshot + clear buffer
        with self.gradients_lock:
            per_gpu_grads = self.collected_gradients
            self.collected_gradients = {}

        # Group grads by param name
        stacked: Dict[str, List[torch.Tensor]] = {}
        for _, grad_dict in per_gpu_grads.items():
            for pname, g in grad_dict.items():
                stacked.setdefault(pname, []).append(g)

        # Average each param's grad
        avg_grads: Dict[str, torch.Tensor] = {}
        for pname, lst in stacked.items():
            avg_grads[pname] = torch.stack(lst).mean(dim=0)

        # Load averaged grads into master model + step optimizer
        self.optimizer.zero_grad(set_to_none=True)
        for (pname, param) in self.model.named_parameters():
            if pname in avg_grads:
                param.grad = avg_grads[pname].to(param.device).clone()
        self.optimizer.step()

    def _drain_metrics(self) -> List[Dict[str, Any]]:
        """
        Return and clear buffered metrics from workers.
        """
        with self.metrics_lock:
            out = self.collected_metrics
            self.collected_metrics = []
        return out



# =====================================================================
# Tensor-Parallel Collectives (Megatron-style intra-layer synchronization)
# =====================================================================

import pickle
import zlib
from .socket_utils import send_with_size, recv_with_size
from .protocol import MessageType


    # -----------------------------------------------------------------
    # Tensor-parallel helper methods
    # -----------------------------------------------------------------

def get_tensor_peers(self, gpu_id: int) -> list[int]:
        """
        Return list of peer GPU IDs that share the same tensor-parallel group.
        Example: tensor_parallel_size = 2 → groups [0,1], [2,3], ...
        """
        tp = getattr(self.cfg, "tensor_parallel_size", 1)
        if tp <= 1:
            return [gpu_id]
        base = gpu_id - (gpu_id % tp)
        return [base + i for i in range(tp)]

def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        """CPU→bytes (compressed)"""
        return zlib.compress(pickle.dumps(tensor.cpu(), protocol=4))

def _deserialize_tensor(self, data: bytes) -> torch.Tensor:
        """bytes→Tensor (GPU if available)"""
        t = pickle.loads(zlib.decompress(data))
        return t.cuda() if torch.cuda.is_available() else t

    # -----------------------------------------------------------------
    # Collective operations (pure socket, no torch.distributed)
    # -----------------------------------------------------------------

def collect_tensor(self, gpu_id: int, tensor: torch.Tensor) -> list[torch.Tensor]:
        """
        All-gather partial outputs from all peers of this layer group.
        Each peer computes Y_chunk = X @ W_chunk.
        We collect all Y_chunk → full Y.
        """
        peers = self.get_tensor_peers(gpu_id)
        if len(peers) == 1:
            return [tensor]

        payload = self._serialize_tensor(tensor)
        results: list[torch.Tensor] = []

        for peer in peers:
            sock = self.ctrl_sockets.get(peer)
            if sock is None:
                continue
            msg = Message(
                msg_type=MessageType.TENSOR_FORWARD_GATHER,
                gpu_id=gpu_id,
                step=self.global_step,
            )
            # send header first
            MessageProtocol.send_message(sock, msg)
            send_with_size(sock, payload)
            # wait for peer's response tensor
            data = recv_with_size(sock)
            results.append(self._deserialize_tensor(data))

        return results

def reduce_tensor(self, gpu_id: int, grad: torch.Tensor) -> torch.Tensor:
        """
        All-reduce (average) gradients across tensor-parallel peers.
        Each peer computes its local dX_chunk;
        we average them to build full dLoss/dX.
        """
        peers = self.get_tensor_peers(gpu_id)
        if len(peers) == 1:
            return grad

        payload = self._serialize_tensor(grad)
        collected = [grad]

        for peer in peers:
            sock = self.ctrl_sockets.get(peer)
            if sock is None:
                continue
            msg = Message(
                msg_type=MessageType.TENSOR_BACKWARD_REDUCE,
                gpu_id=gpu_id,
                step=self.global_step,
            )
            MessageProtocol.send_message(sock, msg)
            send_with_size(sock, payload)
            data = recv_with_size(sock)
            collected.append(self._deserialize_tensor(data))

        # average
        return sum(collected) / len(collected)

def tensor_sync_barrier(self, gpu_id: int) -> None:
        """
        Simple synchronization barrier across tensor-parallel peers.
        Ensures all peers reached the same step boundary before continuing.
        """
        peers = self.get_tensor_peers(gpu_id)
        if len(peers) == 1:
            return
        for peer in peers:
            sock = self.ctrl_sockets.get(peer)
            if sock is None:
                continue
            msg = Message(
                msg_type=MessageType.TENSOR_SYNC_BARRIER,
                gpu_id=gpu_id,
                step=self.global_step,
            )
            MessageProtocol.send_message(sock, msg)
            try:
                sock.recv(2)  # expect short 'OK'
            except Exception:
                pass
