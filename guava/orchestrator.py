"""
orchestrator.py

Central coordinator (the "master" / "brain").
- Listens for worker connections
- Sends training commands / batches to workers
- Receives gradients / metrics / checkpoints
- Runs the top-level training loop

This version is intentionally minimal so we can run end-to-end without
circular imports. It's not the final high-performance orchestrator,
but it matches the public API that orchestrator_train.py expects.
"""

import os
import socket
import struct
import pickle
import threading
import time
from typing import Any, Dict, Optional, List, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DistributedConfig
from .socket_utils import optimize_socket_for_network, find_available_port
from .protocol import MessageType, Message, MessageProtocol


class Orchestrator:
    """
    The Orchestrator owns:
    - the reference copy of the model (or model shards if MP later)
    - the optimizer
    - the training loop (epochs / dataloader iteration)
    - sockets workers connect to

    Public methods you actually call from outside:
      - __init__(config, mode="orchestrator")
      - register_model(model)
      - start_training(train_loader, val_loader, num_epochs=1, val_interval=100)
      - save_checkpoint(path)

    Workers talk to us using the multi-port scheme that NetworkWorker expects:
      master_port + 0 : control / commands / worker registration
      master_port + 1 : metrics upload
      master_port + 2 : gradients upload
      master_port + 3 : activation uplink
      master_port + 4 : activation ACK
      master_port + 5 : resend probe / heartbeat (not fully wired here)
      master_port + 6 : command ACKs
      master_port + 7 : checkpoints from workers
    """

    def __init__(self, config: DistributedConfig, mode: str = "orchestrator"):
        self.cfg = config
        self.mode = mode

        # we store the global model here
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # step counters
        self.global_step = 0
        self.epoch_idx = 0

        # sockets for incoming worker stuff
        self._listener_threads: List[threading.Thread] = []
        self._shutdown_flag = threading.Event()

        # bookkeeping about workers
        # key: gpu_id -> dict(meta info like layer range)
        self.registered_workers: Dict[int, Dict[str, Any]] = {}

        # mutex for gradients/metrics/etc.
        self.gradients_lock = threading.Lock()
        self.collected_gradients: Dict[int, Dict[str, torch.Tensor]] = {}
        self.metrics_lock = threading.Lock()
        self.collected_metrics: List[Dict[str, Any]] = []

        # spawn all server listeners
        self._start_listener_threads()

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def register_model(self, model: nn.Module) -> None:
        """
        Attach the reference model that represents "truth" weights.

        We also build the optimizer here.
        """
        self.model = model
        self.model.train(True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def start_training(
        self,
        train_loader: Iterable,
        val_loader: Optional[Iterable] = None,
        num_epochs: int = 1,
        val_interval: int = 100,
    ) -> None:
        """
        Top-level loop:
        - iterate epochs
        - for each batch:
            1. build a command for workers (data parallel path only here)
            2. send batch to all workers
            3. wait for metrics / gradients
            4. step optimizer with averaged grads
            5. log
            6. periodically run validation (optional)

        NOTE: this "demo orchestrator" assumes DATA PARALLEL mode across N workers.
        We'll extend later for pipeline/model parallel.
        """
        assert self.model is not None, "call register_model(model) first"
        assert self.optimizer is not None, "optimizer not created"

        print("======================================================================")
        print("GUAVA ORCHESTRATOR TRAIN LOOP")
        print(f"num_workers: {self.cfg.num_workers}")
        print(f"parallelism: data_parallel={self.cfg.data_parallel}, "
              f"model_parallel={self.cfg.model_parallel}")
        print("======================================================================")

        # Main training epochs
        for epoch in range(num_epochs):
            self.epoch_idx = epoch
            print(f"[Epoch {epoch+1}/{num_epochs}]")

            for batch_idx, batch in enumerate(train_loader):
                self.global_step += 1

                # batch is (input_ids, labels)
                input_ids, labels = batch
                # convert to python lists so we can pickle -> send
                cmd = {
                    "command": "train_data_parallel",
                    "step": self.global_step,
                    "phase": "train",
                    "ack_required": True,
                    "batch": {
                        "input_ids": input_ids.tolist(),
                        "labels": labels.tolist(),
                    },
                }

                # tell all workers: do forward+backward locally, then upload grads/metrics
                self._broadcast_ctrl(cmd)

                # wait until we've received gradients from everyone,
                # then average them into self.model, then step optimizer
                self._aggregate_and_step()

                # pull metrics that workers pushed, show loss
                metrics = self._drain_metrics()
                for m in metrics:
                    if m.get("loss") is not None:
                        print(f"[step {m['step']}] gpu{m['gpu_id']} loss={m['loss']:.4f}")

                # simple val loop every val_interval steps
                if val_loader is not None and (self.global_step % val_interval == 0):
                    val_loss = self._run_validation(val_loader)
                    print(f"[val @ step {self.global_step}] loss={val_loss:.4f}")

        print("Training loop finished.")

    def wait_for_workers(
        self,
        expected: Optional[int] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1,
    ) -> None:
        """
        Block until the orchestrator has registered N workers.

        Args:
            expected: how many workers we want. Defaults to self.cfg.num_workers.
            timeout: max seconds to wait (None = wait forever).
            poll_interval: sleep interval between checks.

        Raises:
            TimeoutError: if timeout is set and not enough workers connect.
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


    def save_checkpoint(self, path: str) -> None:
        """
        Save orchestrator's version of the model.
        """
        if self.model is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[orchestrator] checkpoint saved -> {path}")

    # ---------------------------------------------------------------------
    # validation
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _run_validation(self, val_loader: Iterable) -> float:
        """
        Very basic validation: run model locally on CPU.
        This is just for sanity check in this demo.
        """
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for (input_ids, labels) in val_loader:
            logits = self.model(input_ids)  # [B,T,V]
            # CE loss next-token
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            num_tok = labels.numel()
            total_loss += float(loss.item()) * num_tok
            total_tokens += num_tok

        self.model.train(True)

        if total_tokens == 0:
            return 0.0
        return total_loss / total_tokens

    # ---------------------------------------------------------------------
    # sockets: server side
    # ---------------------------------------------------------------------
    def _start_listener_threads(self) -> None:
        """
        Spin up background servers for:
          master_port+0 : control/register + incoming ACKs-on-same-socket
          master_port+1 : metrics
          master_port+2 : gradients
          master_port+7 : checkpoints (optional right now)

        For now we don't fully implement +3,+4,+5,+6 logic here,
        but we reserve those ports for pipeline/etc.
        """

        # control / registration / cmds come from us outwards,
        # but we ALSO accept initial HELLO from workers on +0.
        t0 = threading.Thread(target=self._ctrl_server, daemon=True)
        t1 = threading.Thread(target=self._metrics_server, daemon=True)
        t2 = threading.Thread(target=self._grad_server, daemon=True)
        t7 = threading.Thread(target=self._checkpoint_server, daemon=True)

        self._listener_threads = [t0, t1, t2, t7]
        for t in self._listener_threads:
            t.start()

    def _bind_listen(self, port_offset: int) -> socket.socket:
        """
        Bind and listen on master_port + offset.
        Returns the listening socket.
        """
        lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        optimize_socket_for_network(lsock)
        lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        lsock.bind((self.cfg.master_ip, self.cfg.master_port + port_offset))
        lsock.listen()
        return lsock

    def _ctrl_server(self) -> None:
        """
        Accept worker registration and then stay open to read ACKs/etc.
        Protocol:
          worker connects ->
          worker sends: [4-byte len][pickle{gpu_id,start_layer,end_layer,hostname}]
          we store it and respond with ack
        After that, we keep the connection around to send commands.
        For simplicity we just keep each accepted socket in a dict.
        """
        self.ctrl_sockets: Dict[int, socket.socket] = {}  # gpu_id -> socket

        lsock = self._bind_listen(0)
        print(f"[orchestrator] ctrl_server listening on {self.cfg.master_ip}:{self.cfg.master_port}+0")

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

    def _handle_ctrl_connection(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        """
        Read initial HELLO from this worker and stash the socket.
        Then loop receiving ACKs (optional).
        """
        try:
            # read HELLO
            header = self._recv_exact(conn, 4)
            if not header:
                conn.close()
                return
            size = struct.unpack("!I", header)[0]
            blob = self._recv_exact(conn, size)
            hello = pickle.loads(blob)

            gpu_id = hello["gpu_id"]
            self.registered_workers[gpu_id] = hello
            self.ctrl_sockets[gpu_id] = conn

            # send ack
            ack = {"status": "registered", "gpu_id": gpu_id}
            ack_blob = pickle.dumps(ack)
            conn.send(struct.pack("!I", len(ack_blob)))
            conn.sendall(ack_blob)

            print(f"[orchestrator] registered worker GPU {gpu_id} from {addr}")

            # now just sit in a loop reading ACKs, but non-block training must continue,
            # so we use non-blocking recv with small timeout.
            conn.settimeout(0.1)
            while not self._shutdown_flag.is_set():
                try:
                    hdr = conn.recv(4, socket.MSG_PEEK)
                    if not hdr:
                        break

                    # We see a header? then consume it fully.
                    header2 = self._recv_exact(conn, 4)
                    if not header2:
                        break
                    size2 = struct.unpack("!I", header2)[0]
                    payload2 = self._recv_exact(conn, size2)
                    ack_msg = pickle.loads(payload2)

                    # Right now we just print the ACK.
                    ctype = ack_msg.get("cmd_type", "unknown")
                    step = ack_msg.get("step", -1)
                    gid = ack_msg.get("gpu_id", -1)
                    print(f"[orchestrator] ACK from GPU{gid} cmd={ctype} step={step}")

                except socket.timeout:
                    # nothing pending -> loop again
                    continue
                except ConnectionResetError:
                    break

        finally:
            # we do NOT close conn here because we keep it live to send commands.
            # if it dies, future send will fail and we can remove it.
            pass

    def _metrics_server(self) -> None:
        """
        Workers connect here (master_port+1) and push metrics:
          send: [4-byte len][pickle{gpu_id, step, loss, ...}]
        We stash them in self.collected_metrics.
        """
        lsock = self._bind_listen(1)
        print(f"[orchestrator] metrics_server listening on {self.cfg.master_ip}:{self.cfg.master_port}+1")

        while not self._shutdown_flag.is_set():
            try:
                conn, addr = lsock.accept()
            except OSError:
                break

            threading.Thread(
                target=self._handle_metrics_conn,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_metrics_conn(self, conn: socket.socket) -> None:
        try:
            header = self._recv_exact(conn, 4)
            if not header:
                conn.close()
                return
            size = struct.unpack("!I", header)[0]
            blob = self._recv_exact(conn, size)
            metrics = pickle.loads(blob)

            with self.metrics_lock:
                self.collected_metrics.append(metrics)

        finally:
            conn.close()

    def _grad_server(self) -> None:
        """
        Workers connect here (master_port+2) with gradients.
        Protocol:
          header: struct.pack("!Q?", len(zlib_bytes), True)
          body:   zlib(compress(pickle{ 'gpu_id':int, 'gradients':{param_name:tensor,...} }))
        We store per-gpu gradients in self.collected_gradients.
        """
        lsock = self._bind_listen(2)
        print(f"[orchestrator] grad_server listening on {self.cfg.master_ip}:{self.cfg.master_port}+2")

        while not self._shutdown_flag.is_set():
            try:
                conn, addr = lsock.accept()
            except OSError:
                break

            threading.Thread(
                target=self._handle_grad_conn,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_grad_conn(self, conn: socket.socket) -> None:
        import zlib
        try:
            hdr = self._recv_exact(conn, 9)
            if not hdr:
                conn.close()
                return
            comp_len, _ = struct.unpack("!Q?", hdr)

            comp_blob = self._recv_exact(conn, comp_len)
            raw = zlib.decompress(comp_blob)
            grad_msg = pickle.loads(raw)

            gpu_id = grad_msg["gpu_id"]
            grads_dict = grad_msg["gradients"]

            # store
            with self.gradients_lock:
                self.collected_gradients[gpu_id] = grads_dict

        finally:
            conn.close()

    def _checkpoint_server(self) -> None:
        """
        Workers connect here (master_port+7) when shutting down
        to upload their final state_dict shard.
        We just read it and write it to disk.
        """
        lsock = self._bind_listen(7)
        print(f"[orchestrator] checkpoint_server listening on {self.cfg.master_ip}:{self.cfg.master_port}+7")

        while not self._shutdown_flag.is_set():
            try:
                conn, addr = lsock.accept()
            except OSError:
                break

            threading.Thread(
                target=self._handle_checkpoint_conn,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_checkpoint_conn(self, conn: socket.socket) -> None:
        import zlib
        try:
            # first read meta
            meta_len_hdr = self._recv_exact(conn, 4)
            if not meta_len_hdr:
                conn.close()
                return
            meta_len = struct.unpack("!I", meta_len_hdr)[0]
            meta_blob = self._recv_exact(conn, meta_len)
            meta = pickle.loads(meta_blob)

            # then read compressed state_dict
            comp_len_hdr = self._recv_exact(conn, 8)
            comp_len = struct.unpack("!Q", comp_len_hdr)[0]
            comp_blob = self._recv_exact(conn, comp_len)
            raw_blob = zlib.decompress(comp_blob)
            state_dict = pickle.loads(raw_blob)

            # write shard to disk
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
            out_path = os.path.join(self.cfg.checkpoint_dir, meta["filename"])
            torch.save(state_dict, out_path)

            # send 1-byte "OK"
            conn.sendall(b"\x01")

            print(f"[orchestrator] got checkpoint from GPU{meta['gpu_id']} -> {out_path}")
        finally:
            conn.close()

    # ---------------------------------------------------------------------
    # training helpers
    # ---------------------------------------------------------------------
    def _broadcast_ctrl(self, cmd: Dict[str, Any]) -> None:
        """
        Send a command dict to EVERY currently registered worker
        over their control socket.
        """
        dead = []
        blob = pickle.dumps(cmd)
        header = struct.pack("!I", len(blob))

        for gpu_id, sock in self.ctrl_sockets.items():
            try:
                sock.sendall(header)
                sock.sendall(blob)
            except Exception as e:
                print(f"[orchestrator] failed to send cmd to GPU{gpu_id}: {e}")
                dead.append(gpu_id)

        for gpu_id in dead:
            self.ctrl_sockets.pop(gpu_id, None)
            self.registered_workers.pop(gpu_id, None)

    def _aggregate_and_step(self) -> None:
        """
        Wait until we have gradients from ALL expected workers,
        average them, apply to our master model, step optimizer.

        Sync-style data parallel.
        """
        assert self.model is not None
        assert self.optimizer is not None

        expected_workers = max(self.cfg.num_workers, 1)

        # wait until we've heard from everyone
        while True:
            with self.gradients_lock:
                ready = len(self.collected_gradients) >= expected_workers
            if ready:
                break
            time.sleep(0.01)

        # grab and clear
        with self.gradients_lock:
            per_gpu_grads = self.collected_gradients
            self.collected_gradients = {}

        # group grads by param name
        stacked: Dict[str, List[torch.Tensor]] = {}
        for gpu_id, grad_dict in per_gpu_grads.items():
            for name, g in grad_dict.items():
                stacked.setdefault(name, []).append(g)

        # average per param
        avg_grads: Dict[str, torch.Tensor] = {}
        for name, lst in stacked.items():
            avg_grads[name] = torch.stack(lst).mean(dim=0)

        # load averaged grads into master model and step
        self.optimizer.zero_grad(set_to_none=True)
        for (name, param) in self.model.named_parameters():
            if name in avg_grads:
                param.grad = avg_grads[name].to(param.device).clone()

        self.optimizer.step()


    def _drain_metrics(self) -> List[Dict[str, Any]]:
        """Return and clear collected metrics."""
        with self.metrics_lock:
            out = self.collected_metrics
            self.collected_metrics = []
        return out

    # ---------------------------------------------------------------------
    # util
    # ---------------------------------------------------------------------
    def _recv_exact(self, conn: socket.socket, n: int) -> bytes:
        """
        Read exactly n bytes or return b"" on disconnect.
        """
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                return b""
            buf += chunk
        return buf
