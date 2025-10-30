"""
protocol.py

Canonical message / wire contract for GUAVA distributed training.

Why this exists:
- Orchestrator <-> Workers talk over multiple TCP ports.
- We want a consistent schema for:
    * control commands (start step, phase1, phase2, stop, etc.)
    * metrics (loss, step)
    * gradients upload
    * activation frames (pipeline forward handoff)
    * tensor-parallel collectives (forward gather / backward reduce)

All control/telemetry goes through a single framing:
MessageProtocol (length-prefixed, optional zlib, pickle payload).
"""

import struct
import pickle
import zlib
import socket
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass

import torch  # used by tensor (un)wrapping helpers


class MessageType(Enum):
    """
    Unified message taxonomy for orchestrator <-> worker communication.

    Socket mapping:
        +0 : Control / lifecycle (long-lived per worker)
        +1 : Metrics (short-lived per send)
        +2 : Gradients (short-lived per send)
        +7 : Checkpoints (short-lived per send)
    """

    # ---------------- Control & Lifecycle ----------------
    CONTROL_START = "CONTROL_START"                     # orch -> workers (optional warmup/reset)
    CONTROL_STOP = "CONTROL_STOP"                       # orch -> workers (graceful shutdown)
    CONTROL_HELLO = "CONTROL_HELLO"                     # worker -> orch (register)
    CONTROL_ACK = "CONTROL_ACK"                         # worker <-> orch (barrier / status)
    CONTROL_HEARTBEAT = "CONTROL_HEARTBEAT"             # worker -> orch (liveness)

    # ---------------- Step / Training Control ----------------
    CONTROL_DATA_PARALLEL_STEP = "CONTROL_DATA_PARALLEL_STEP"   # orch -> workers
    CONTROL_PIPELINE_PHASE1 = "CONTROL_PIPELINE_PHASE1"         # orch -> all workers (forward start)
    CONTROL_PIPELINE_PHASE2 = "CONTROL_PIPELINE_PHASE2"         # orch -> last worker only (labels/loss/backward start)
    CONTROL_PIPELINE_BACKWARD = "CONTROL_PIPELINE_BACKWARD"     # orch -> mid/earlier shard (propagate upstream grad)

    # ---------------- Activation Relay (optional future path) ----------------
    ACTIVATION_FRAME = "ACTIVATION_FRAME"               # orch -> worker (relay activations to next stage)

    # ---------------- Gradient Flow ----------------
    BACKWARD_READY = "BACKWARD_READY"                   # worker -> orch (done backward, upstream grad ready)

    # ---------------- Metrics / Gradients / Checkpoints ----------------
    METRICS_STEP = "METRICS_STEP"                       # worker -> orch (+1 socket)
    GRADIENTS_UPLOAD = "GRADIENTS_UPLOAD"               # worker -> orch (+2 socket)
    CHECKPOINT_SHARD_UPLOAD = "CHECKPOINT_SHARD_UPLOAD" # worker -> orch (+7 socket)

    # ---------------- Tensor-Parallel Collectives ----------------
    TENSOR_FORWARD_GATHER = "TENSOR_FORWARD_GATHER"     # worker <-> orch (gather partial Y along model dim)
    TENSOR_BACKWARD_REDUCE = "TENSOR_BACKWARD_REDUCE"   # worker <-> orch (all-reduce/avg dLoss/dX)
    TENSOR_SYNC_BARRIER = "TENSOR_SYNC_BARRIER"         # worker <-> orch (simple barrier if needed)


@dataclass
class Message:
    """
    Canonical message container.

    Fields:
        msg_type: MessageType enum
        payload:  Picklable body (dict / list / numpy array / etc.)
        metadata: Routing/context like {"layer_start":0,"layer_end":6}
        step:     Global training step this refers to
        gpu_id:   Sender GPU ID (worker local index)
        phase:    "train", "val", "phase1", etc.
        micro_batch_idx: which micro-batch in the pipeline (for overlap)
        num_micro_batches: total micro-batches in this step
    """

    msg_type: MessageType
    payload: Any = None
    metadata: Optional[Dict] = None
    step: Optional[int] = None
    gpu_id: Optional[int] = None
    phase: Optional[str] = None
    micro_batch_idx: Optional[int] = None
    num_micro_batches: Optional[int] = None

    def to_dict(self) -> dict:
        """
        Make a plain dict for pickling or JSON. This is what hits the wire.
        """
        return {
            "msg_type": (
                self.msg_type.value
                if isinstance(self.msg_type, MessageType)
                else self.msg_type
            ),
            "payload": self.payload,
            "metadata": self.metadata,
            "step": self.step,
            "gpu_id": self.gpu_id,
            "phase": self.phase,
            "micro_batch_idx": self.micro_batch_idx,
            "num_micro_batches": self.num_micro_batches,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """
        Rebuild a Message from a dict we unpickled.
        """
        msg_type = data["msg_type"]
        if isinstance(msg_type, str):
            msg_type = MessageType(msg_type)

        return cls(
            msg_type=msg_type,
            payload=data.get("payload"),
            metadata=data.get("metadata"),
            step=data.get("step"),
            gpu_id=data.get("gpu_id"),
            phase=data.get("phase"),
            micro_batch_idx=data.get("micro_batch_idx"),
            num_micro_batches=data.get("num_micro_batches"),
        )


class MessageProtocol:
    """
    Length-prefixed, optional-zlib, pickle-based framing.

    Wire frame:
        [4 bytes big-endian uint32: body_len]
        [body_len bytes: (maybe-compressed) pickle(Message.to_dict())]
    """

    # ------------------------------------------------------------------
    # Core (de)serialization
    # ------------------------------------------------------------------
    @staticmethod
    def serialize(message: Message, compress: bool = True) -> bytes:
        """
        Message -> bytes
        """
        raw = pickle.dumps(message.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)
        return zlib.compress(raw, level=6) if compress else raw

    @staticmethod
    def deserialize(data: bytes, decompress: bool = True) -> Message:
        """
        bytes -> Message
        """
        if decompress:
            data = zlib.decompress(data)
        msg_dict = pickle.loads(data)
        return Message.from_dict(msg_dict)

    # ------------------------------------------------------------------
    # Socket helpers
    # ------------------------------------------------------------------
    @staticmethod
    def pack_for_send(message: Message, compress: bool = True) -> bytes:
        """
        Build header+payload as one contiguous buffer for sendall().
        """
        body = MessageProtocol.serialize(message, compress=compress)
        header = struct.pack("!I", len(body))
        return header + body

    @staticmethod
    def send_message(sock: socket.socket, message: Message, compress: bool = True) -> None:
        """
        Safe 2-part send: header then payload.
        """
        body = MessageProtocol.serialize(message, compress=compress)
        header = struct.pack("!I", len(body))
        sock.sendall(header)
        sock.sendall(body)

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        """
        Read exactly n bytes or return None if peer closed.
        """
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    @staticmethod
    def receive_message(
        sock: socket.socket,
        *,
        decompress: bool = True,
        timeout: Optional[float] = None,
        max_len_bytes: int = 100 * 1024 * 1024,  # 100 MB sanity cap
    ) -> Optional[Message]:
        """
        Blocking receive of a single framed Message.

        Returns:
            Message, or None if the peer closed cleanly.
        Raises:
            TimeoutError, ConnectionError, ValueError
        """
        prev_timeout = sock.gettimeout()
        try:
            sock.settimeout(timeout if timeout is not None else None)

            header = MessageProtocol._recv_exact(sock, 4)
            if header is None:
                return None
            (length,) = struct.unpack("!I", header)

            if length > max_len_bytes:
                raise ValueError(f"Message too large: {length} bytes")

            body = MessageProtocol._recv_exact(sock, length)
            if body is None:
                return None

            return MessageProtocol.deserialize(body, decompress=decompress)

        except socket.timeout as e:
            raise TimeoutError(f"Receive timeout: {e}")
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise ConnectionError(f"Socket receive failed: {e}")
        finally:
            sock.settimeout(prev_timeout)

    # ------------------------------------------------------------------
    # Tensor helpers (activations, gradients, etc.) — optional path
    # ------------------------------------------------------------------
    @staticmethod
    def wrap_tensor_payload(
        tensor: torch.Tensor,
        *,
        include_grad: bool = False,
    ) -> Dict[str, Any]:
        """
        Turn a torch.Tensor into a picklable dict payload.
        Used for ACTIVATION_FRAME and similar messages.
        """
        cpu_t = tensor.detach().cpu()
        return {
            "tensor_np": cpu_t.numpy(),          # numpy array
            "shape": tuple(cpu_t.shape),
            "dtype": str(cpu_t.dtype),
            "requires_grad": bool(
                tensor.requires_grad if include_grad else False
            ),
        }

    @staticmethod
    def unwrap_tensor_payload(
        payload: Dict[str, Any],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Rebuild a tensor from wrap_tensor_payload() dict.
        Returned tensor is placed on `device`.
        """
        t = torch.from_numpy(payload["tensor_np"]).to(device)
        return t
