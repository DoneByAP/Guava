"""
protocol.py

Canonical message / wire contract for GUAVA distributed training.
"""

import struct
import pickle
import zlib
import socket
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass
import time 
import torch  # used by tensor (un)wrapping helpers
from .energy_monitor import get_error_tracker, ErrorSeverity, track_errors


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
    CONTROL_START = "CONTROL_START"
    CONTROL_STOP = "CONTROL_STOP"
    CONTROL_HELLO = "CONTROL_HELLO"
    CONTROL_GOODBYE = "CONTROL_GOODBYE"
    CONTROL_ACK = "CONTROL_ACK"
    CONTROL_HEARTBEAT = "CONTROL_HEARTBEAT"
    CONTROL_RESEND_REQUEST = "CONTROL_RESEND_REQUEST"  # ✅ NEW: Worker requests batch resend due to corruption

    # ---------------- Step / Training Control ----------------
    CONTROL_DATA_PARALLEL_STEP = "CONTROL_DATA_PARALLEL_STEP"
    CONTROL_PIPELINE_PHASE1 = "CONTROL_PIPELINE_PHASE1"
    CONTROL_PIPELINE_PHASE2 = "CONTROL_PIPELINE_PHASE2"
    CONTROL_PIPELINE_BACKWARD = "CONTROL_PIPELINE_BACKWARD"

    # ---------------- Activation Relay ----------------
    ACTIVATION_FRAME = "ACTIVATION_FRAME"

    # ---------------- Gradient Flow ----------------
    BACKWARD_READY = "BACKWARD_READY"

    # ---------------- Metrics / Gradients / Checkpoints ----------------
    METRICS_STEP = "METRICS_STEP"
    GRADIENTS_UPLOAD = "GRADIENTS_UPLOAD"
    CHECKPOINT_SHARD_UPLOAD = "CHECKPOINT_SHARD_UPLOAD"

    # ---------------- Tensor-Parallel Collectives ----------------
    TENSOR_FORWARD_GATHER = "TENSOR_FORWARD_GATHER"
    TENSOR_BACKWARD_REDUCE = "TENSOR_BACKWARD_REDUCE"
    TENSOR_SYNC_BARRIER = "TENSOR_SYNC_BARRIER"


# ✅ NEW: Custom exception for message corruption
class MessageCorruptedError(Exception):
    """Raised when message data is corrupt and needs to be resent by orchestrator"""
    pass


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
    Length-prefixed, optional-zlib, pickle-based framing with MAGIC HEADER sync.

    Wire frame:
        [4 bytes: MAGIC_HEADER (0xDEADBEEF)]
        [4 bytes big-endian uint32: body_len]
        [body_len bytes: (maybe-compressed) pickle(Message.to_dict())]
    """
    
    # ✅ MAGIC HEADER for frame synchronization
    MAGIC_HEADER = b'\xDE\xAD\xBE\xEF'

    # ------------------------------------------------------------------
    # Core (de)serialization
    # ------------------------------------------------------------------
    @staticmethod
    def serialize(message: Message, compress: bool = True) -> bytes:
        """Message -> bytes"""
        raw = pickle.dumps(message.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)
        return zlib.compress(raw, level=6) if compress else raw

    @staticmethod
    def deserialize(data: bytes, decompress: bool = True) -> Message:
        """bytes -> Message"""
        if decompress:
            data = zlib.decompress(data)
        msg_dict = pickle.loads(data)
        return Message.from_dict(msg_dict)

    # ------------------------------------------------------------------
    # Socket helpers
    # ------------------------------------------------------------------

    @staticmethod
    def send_message(sock: socket.socket, message: Message, compress: bool = True) -> None:
        """
        Safe send with MAGIC HEADER for sync recovery.
        Frame: [MAGIC][LENGTH][BODY]
        """
        body = MessageProtocol.serialize(message, compress=compress)
        
        # ✅ Build frame: magic + length + body
        frame = (
            MessageProtocol.MAGIC_HEADER +
            struct.pack("!I", len(body)) +
            body
        )
        
        sock.sendall(frame)

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
        """Read exactly n bytes or return None if peer closed."""
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
        timeout: Optional[float] = None,
        max_len_bytes: Optional[int] = None,
        max_retries: int = 10,
        retry_interval: float = 6.0,
        channel_name: str = "unknown",
    ) -> Optional[Message]:
        """
        Blocking receive with MAGIC HEADER sync recovery and corruption detection.
        Now reports all errors to EnergyMonitor telemetry!
        """
        
        if max_len_bytes is None:
            max_len_bytes = 1_000_000_000

        prev_timeout = sock.gettimeout()
        
        for attempt in range(max_retries):
            try:
                sock.settimeout(timeout if timeout is not None else None)

                # ✅ STEP 1: Find MAGIC HEADER (with sync recovery)
                sync_attempts = 0
                max_sync_attempts = 100
                
                while sync_attempts < max_sync_attempts:
                    magic = MessageProtocol._recv_exact(sock, 4)
                    if magic is None:
                        return None
                    
                    if magic == MessageProtocol.MAGIC_HEADER:
                        # ✅ Found sync point!
                        if sync_attempts > 0:
                            # ✅ LOG SUCCESSFUL RESYNC
                            print(f"[protocol:{channel_name}] ✅ Resynchronized after {sync_attempts} attempts")
                            
                            try:
                                get_error_tracker().report_error(
                                    error=RuntimeError(f"Socket desync recovered after {sync_attempts} attempts"),
                                    context=f"protocol.receive_message.{channel_name}",
                                    severity=ErrorSeverity.WARNING,
                                    include_traceback=False,
                                    metadata={
                                        "channel": channel_name,
                                        "sync_attempts": sync_attempts,
                                        "expected_magic": MessageProtocol.MAGIC_HEADER.hex(),
                                        "recovery_success": True,
                                    },
                                    recovery_attempted=True,
                                    recovery_success=True,
                                )
                            except Exception:
                                pass
                        
                        break
                    else:
                        # ❌ Out of sync!
                        if sync_attempts == 0:
                            print(f"[protocol:{channel_name}] ⚠️ OUT OF SYNC! Searching for magic header...")
                            print(f"[protocol:{channel_name}] Expected: {MessageProtocol.MAGIC_HEADER.hex()}, got: {magic.hex()}")
                            
                            # ✅ LOG DESYNC EVENT
                            try:
                                get_error_tracker().report_error(
                                    error=RuntimeError("Socket desynchronization detected"),
                                    context=f"protocol.receive_message.{channel_name}",
                                    severity=ErrorSeverity.ERROR,
                                    include_traceback=False,
                                    metadata={
                                        "channel": channel_name,
                                        "expected_magic": MessageProtocol.MAGIC_HEADER.hex(),
                                        "actual_bytes": magic.hex(),
                                        "attempt": attempt + 1,
                                    },
                                    recovery_attempted=True,
                                    recovery_success=None,  # Don't know yet
                                )
                            except Exception:
                                pass
                        
                        sync_attempts += 1
                        continue
                
                if sync_attempts >= max_sync_attempts:
                    print(f"[protocol:{channel_name}] ❌ Failed to find sync point after {max_sync_attempts} attempts")
                    
                    # ✅ LOG PERMANENT DESYNC
                    error = MessageCorruptedError("Cannot find magic header - socket permanently out of sync")
                    try:
                        get_error_tracker().report_error(
                            error=error,
                            context=f"protocol.receive_message.{channel_name}",
                            severity=ErrorSeverity.CRITICAL,
                            include_traceback=False,
                            metadata={
                                "channel": channel_name,
                                "sync_attempts": sync_attempts,
                                "max_sync_attempts": max_sync_attempts,
                            },
                            recovery_attempted=True,
                            recovery_success=False,
                        )
                    except Exception:
                        pass
                    
                    raise error

                # ✅ STEP 2: Read length
                length_bytes = MessageProtocol._recv_exact(sock, 4)
                if length_bytes is None:
                    return None
                
                (length,) = struct.unpack("!I", length_bytes)
                
                # ✅ STEP 3: Validate length
                if length == 0:
                    print(f"[protocol:{channel_name}] ⚠️ Zero-length message, skipping...")
                    
                    # ✅ LOG ZERO-LENGTH MESSAGE
                    try:
                        get_error_tracker().report_error(
                            error=ValueError("Zero-length message received"),
                            context=f"protocol.receive_message.{channel_name}",
                            severity=ErrorSeverity.WARNING,
                            include_traceback=False,
                            metadata={"channel": channel_name, "attempt": attempt + 1},
                        )
                    except Exception:
                        pass
                    
                    continue
                
                if length > max_len_bytes:
                    print(f"[protocol:{channel_name}] ❌ Invalid length: {length:,} bytes (max: {max_len_bytes:,})")
                    print(f"[protocol:{channel_name}] Length bytes: {length_bytes.hex()}")
                    
                    # ✅ LOG INVALID LENGTH
                    try:
                        get_error_tracker().report_error(
                            error=ValueError(f"Message too large: {length} bytes"),
                            context=f"protocol.receive_message.{channel_name}",
                            severity=ErrorSeverity.ERROR,
                            include_traceback=False,
                            metadata={
                                "channel": channel_name,
                                "length": length,
                                "max_length": max_len_bytes,
                                "length_bytes": length_bytes.hex(),
                                "attempt": attempt + 1,
                            },
                        )
                    except Exception:
                        pass
                    
                    if attempt < max_retries - 1:
                        print(f"[protocol:{channel_name}] 🔄 Resyncing...")
                        continue
                    raise ValueError(f"Message too large: {length} bytes")

                # ✅ STEP 4: Read body
                body = MessageProtocol._recv_exact(sock, length)
                if body is None:
                    return None

                # ✅ STEP 5: Deserialize
                for decompress_flag in [True, False]:
                    try:
                        msg = MessageProtocol.deserialize(body, decompress=decompress_flag)
                        return msg
                    except (zlib.error, pickle.UnpicklingError, EOFError, ValueError) as e:
                        last_error = e
                        continue
                
                # ❌ All deserialization strategies failed
                print(f"[protocol:{channel_name}] ❌ Attempt {attempt+1}/{max_retries}: Deserialization failed")
                print(f"[protocol:{channel_name}] Body length: {len(body)}, first 32 bytes: {body[:32].hex() if len(body) >= 32 else body.hex()}")
                
                # ✅ LOG DESERIALIZATION FAILURE
                try:
                    get_error_tracker().report_error(
                        error=last_error,
                        context=f"protocol.receive_message.{channel_name}",
                        severity=ErrorSeverity.ERROR,
                        include_traceback=False,
                        metadata={
                            "channel": channel_name,
                            "body_length": len(body),
                            "first_32_bytes": body[:32].hex() if len(body) >= 32 else body.hex(),
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "error_type": type(last_error).__name__,
                        },
                    )
                except Exception:
                    pass
                
                if attempt < max_retries - 1:
                    print(f"[protocol:{channel_name}] 🔄 Corrupt message discarded, waiting {retry_interval}s for resend...")
                    time.sleep(retry_interval)
                    continue
                else:
                    raise MessageCorruptedError(f"Deserialization failed after {max_retries} attempts")

            except socket.timeout:
                if attempt < max_retries - 1:
                    print(f"[protocol:{channel_name}] ⏱️ Attempt {attempt+1}/{max_retries}: Timeout, waiting {retry_interval}s...")
                    
                    # ✅ LOG TIMEOUT (only on first attempt)
                    if attempt == 0:
                        try:
                            get_error_tracker().report_error(
                                error=TimeoutError(f"Socket timeout on {channel_name}"),
                                context=f"protocol.receive_message.{channel_name}",
                                severity=ErrorSeverity.WARNING,
                                include_traceback=False,
                                metadata={
                                    "channel": channel_name,
                                    "timeout": timeout,
                                    "attempt": attempt + 1,
                                },
                            )
                        except Exception:
                            pass
                    
                    time.sleep(retry_interval)
                    continue
                else:
                    raise TimeoutError(f"[{channel_name}] Receive timeout after {max_retries} attempts")
                    
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                # ✅ LOG CONNECTION ERROR
                try:
                    get_error_tracker().report_error(
                        error=e,
                        context=f"protocol.receive_message.{channel_name}",
                        severity=ErrorSeverity.CRITICAL,
                        include_traceback=True,
                        metadata={
                            "channel": channel_name,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                        },
                    )
                except Exception:
                    pass
                
                if attempt < max_retries - 1:
                    print(f"[protocol:{channel_name}] 🔌 Attempt {attempt+1}/{max_retries}: Connection error ({e}), retrying...")
                    time.sleep(retry_interval)
                    continue
                else:
                    raise ConnectionError(f"[{channel_name}] Socket failed after {max_retries} attempts: {e}")
            
            finally:
                sock.settimeout(prev_timeout)
        
        return None

# ✅ NEW: Custom exception so worker knows to request resend
class MessageCorruptedError(Exception):
    """Raised when message data is corrupt and needs to be resent"""
    pass
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
            "tensor_np": cpu_t.numpy(),
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