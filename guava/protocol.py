"""
Communication protocol for distributed training.

This defines:
- MessageType: enum of all valid packet "kinds"
- Message: lightweight container (serializable dict)
- MessageProtocol: helpers for length-prefixed, compressed, pickle-based socket I/O
  plus tensor send/receive utilities.

All network traffic between orchestrator and workers should go through this.
"""

import struct
import pickle
import zlib
import socket
from enum import Enum
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass


class MessageType(Enum):
    """Message types for distributed training protocol."""

    # --- Control / lifecycle -------------------------------------------------
    HELLO = "HELLO"                 # worker -> orchestrator: "I'm alive"
    READY = "READY"                 # worker -> orchestrator: "ready for work"
    START_TRAINING = "START_TRAINING"
    STOP_TRAINING = "STOP_TRAINING"
    SHUTDOWN = "SHUTDOWN"
    HEARTBEAT = "HEARTBEAT"
    ACK = "ACK"

    # --- Training data / tensors --------------------------------------------
    BATCH_DATA = "BATCH_DATA"       # orchestrator -> worker: inputs batch
    ACTIVATIONS = "ACTIVATIONS"     # shard -> next shard: fwd activations
    GRADIENTS = "GRADIENTS"         # shard -> prev shard: bwd gradients
    LABELS = "LABELS"               # orchestrator -> worker: ground-truth labels

    # --- Model state / weights ----------------------------------------------
    MODEL_CONFIG = "MODEL_CONFIG"   # orchestrator -> worker: hyperparams / structural info
    MODEL_WEIGHTS = "MODEL_WEIGHTS" # orchestrator <-> worker: full/broadcasted state_dict
    MODEL_UPDATE = "MODEL_UPDATE"   # orchestrator -> worker: apply these grads / sync step

    # --- Metrics / telemetry -------------------------------------------------
    LOSS = "LOSS"                   # worker -> orchestrator: scalar loss
    METRICS = "METRICS"             # worker -> orchestrator: throughput, ppl, etc.

    # --- Pipeline phase coordination ----------------------------------------
    PHASE1 = "PHASE1"               # "forward phase" broadcast / step barrier
    PHASE2 = "PHASE2"               # "backward/optim phase" / label dispatch

    # --- Error / recovery ----------------------------------------------------
    ERROR = "ERROR"                 # worker/orchestrator -> peer: error description
    RETRY = "RETRY"                 # orchestrator -> worker: resend request, re-run step


@dataclass
class Message:
    """
    Structured message used by MessageProtocol.

    msg_type: MessageType enum identifying the packet role.
    payload: Any picklable data (dicts, lists, scalars, numpy arrays, state_dict chunks, etc.).
    metadata: Small dict for routing/extra keys: {"layer_start":6,"layer_end":12,"timestamp":...}
    step: Training step this message is about.
    gpu_id: Which GPU index (local index on that machine) sent this, if relevant.
    phase: "train" or "val" or "phase1"/"phase2", etc.
    """

    msg_type: MessageType
    payload: Any
    metadata: Optional[Dict] = None
    step: Optional[int] = None
    gpu_id: Optional[int] = None
    phase: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert message into a plain dict that can be pickled or JSON'd."""
        return {
            "msg_type": self.msg_type.value if isinstance(self.msg_type, MessageType) else self.msg_type,
            "payload": self.payload,
            "metadata": self.metadata,
            "step": self.step,
            "gpu_id": self.gpu_id,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Inverse of to_dict()."""
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
        )


class MessageProtocol:
    """
    Length-prefixed, zlib-compressed, pickle-based messaging layer.

    Wire format for each frame:
        [4 bytes: big-endian uint32 payload_len]
        [payload_len bytes: pickled (and optionally compressed) dict]

    NOTE:
    - We always pickle a dict (Message.to_dict()).
    - We never pickle raw torch.Tensors directly on the main channel unless wrapped.
    """

    # ---------------------------
    # Core (de)serialization
    # ---------------------------

    @staticmethod
    def serialize(message: Message, compress: bool = True) -> bytes:
        """
        Turn a Message into bytes (optionally compressed).
        """
        raw = pickle.dumps(message.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)
        if compress:
            raw = zlib.compress(raw, level=6)
        return raw

    @staticmethod
    def deserialize(data: bytes, decompress: bool = True) -> Message:
        """
        Bytes -> Message.
        """
        if decompress:
            data = zlib.decompress(data)
        msg_dict = pickle.loads(data)
        return Message.from_dict(msg_dict)

    # ---------------------------
    # Socket send/recv helpers
    # ---------------------------

    @staticmethod
    def pack_for_send(message: Message, compress: bool = True) -> bytes:
        """
        Convenience helper: return header+payload bytes in one buffer.
        Lets callers do: sock.sendall(MessageProtocol.pack_for_send(msg)).
        """
        body = MessageProtocol.serialize(message, compress=compress)
        header = struct.pack("!I", len(body))
        return header + body

    @staticmethod
    def send_message(sock: socket.socket, message: Message, compress: bool = True) -> None:
        """
        High-level safe send. Uses sendall() twice (header then body).
        If you need a single sendall(), use pack_for_send().
        """
        try:
            body = MessageProtocol.serialize(message, compress=compress)
            header = struct.pack("!I", len(body))
            sock.sendall(header)
            sock.sendall(body)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise ConnectionError(f"Failed to send message: {e}")

    @staticmethod
    def _recv_exact(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
        """
        Read exactly num_bytes from the socket, unless connection closes.
        Returns None if EOF.
        """
        buf = b""
        while len(buf) < num_bytes:
            chunk = sock.recv(num_bytes - len(buf))
            if not chunk:
                return None  # peer closed
            buf += chunk
        return buf

    @staticmethod
    def receive_message(
        sock: socket.socket,
        decompress: bool = True,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Blocking receive of a single Message frame.

        timeout:
            None  -> blocking
            float -> set a temporary socket timeout in seconds

        Returns:
            Message or None if peer closed cleanly.

        Raises:
            TimeoutError if we hit timeout
            ConnectionError for socket breakage
            ValueError if payload length is insane
        """
        prev_timeout = sock.gettimeout()
        try:
            # apply temporary timeout policy
            sock.settimeout(timeout if timeout is not None else None)

            # read header
            header = MessageProtocol._recv_exact(sock, 4)
            if header is None:
                return None  # remote closed
            (length,) = struct.unpack("!I", header)

            # sanity limit
            if length > 100 * 1024 * 1024:  # 100 MB safety cap
                raise ValueError(f"Message too large: {length} bytes")

            # read payload
            body = MessageProtocol._recv_exact(sock, length)
            if body is None:
                return None  # remote closed mid-frame

            # decode
            msg = MessageProtocol.deserialize(body, decompress=decompress)
            return msg

        except socket.timeout as e:
            raise TimeoutError(f"Receive timeout: {e}")
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            raise ConnectionError(f"Failed to receive message: {e}")
        finally:
            # restore original timeout
            sock.settimeout(prev_timeout)

    # ---------------------------
    # Tensor helpers
    # ---------------------------

    @staticmethod
    def send_tensor(
        sock: socket.socket,
        tensor,
        metadata: Optional[Dict] = None,
        *,
        msg_type: MessageType = MessageType.ACTIVATIONS,
        step: Optional[int] = None,
        gpu_id: Optional[int] = None,
        phase: Optional[str] = None,
        compress: bool = False,
    ) -> None:
        """
        Send a torch.Tensor over the wire by wrapping it in a Message.

        We:
        - move tensor to CPU
        - convert to numpy
        - include shape, dtype, device, requires_grad, etc.
        - attach optional routing info (step, gpu_id, phase)

        NOTE: msg_type defaults to ACTIVATIONS but can be GRADIENTS, LABELS, etc.
        """
        import torch
        import numpy as np  # noqa: F401 (pickle handles numpy arrays)

        cpu_t = tensor.detach().cpu()
        tensor_np = cpu_t.numpy()

        payload = {
            "data": tensor_np,                          # numpy array (picklable)
            "shape": tuple(cpu_t.shape),                # (B, T, C) etc.
            "dtype": str(cpu_t.dtype),                  # 'torch.float32', etc.
            "requires_grad": bool(tensor.requires_grad),
            "device_str": str(tensor.device),
        }

        full_meta = dict(metadata or {})
        if step is not None:
            full_meta["step"] = step
        if gpu_id is not None:
            full_meta["gpu_id"] = gpu_id
        if phase is not None:
            full_meta["phase"] = phase

        message = Message(
            msg_type=msg_type,
            payload=payload,
            metadata=full_meta if full_meta else None,
            step=step,
            gpu_id=gpu_id,
            phase=phase,
        )

        MessageProtocol.send_message(sock, message, compress=compress)

    @staticmethod
    def receive_tensor(
        sock: socket.socket,
        device: str = "cpu",
        timeout: Optional[float] = None,
    ):
        """
        Receive a Message that wraps a tensor, rebuild the torch.Tensor,
        and move it to the requested device.

        Returns:
            (tensor, info) where:
                tensor = reconstructed torch.Tensor on `device`
                info   = dict with metadata/attrs like 'requires_grad', 'gpu_id', etc.
        """
        import torch

        msg = MessageProtocol.receive_message(sock, timeout=timeout)
        if msg is None:
            return None, None

        payload = msg.payload
        tensor_np = payload["data"]  # numpy array that we pickled
        t = torch.from_numpy(tensor_np).to(device)

        # we don't reapply requires_grad here automatically.
        # upstream code can call t.requires_grad_(True) if it needs to backprop.
        info = {
            "shape": payload.get("shape"),
            "dtype": payload.get("dtype"),
            "requires_grad": payload.get("requires_grad", False),
            "device_str": payload.get("device_str"),
            "metadata": msg.metadata,
            "step": msg.step,
            "gpu_id": msg.gpu_id,
            "phase": msg.phase,
            "msg_type": msg.msg_type,
        }

        return t, info
