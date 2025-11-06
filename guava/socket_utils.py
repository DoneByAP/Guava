"""
socket_utils.py

Low-level socket helpers for distributed training.
"""

import socket
import platform
import time
import struct
from typing import Optional, Tuple


def optimize_socket_for_network(sock: socket.socket, buf_bytes: Optional[int] = None) -> None:
    """
    Tune a TCP socket for Guava traffic.
    """
    if buf_bytes is None:
        # Set to maximum reasonable buffer size
        # Note: OS will clamp to kernel maximum anyway
        buf_bytes = 10 * 1024 * 1024 * 1024  # 10 GB

    # Clamp to maximum int32 value to avoid overflow (2^31 - 1 = 2,147,483,647 bytes ~= 2 GB)
    # This is the maximum that setsockopt can handle on most systems
    max_socket_buf = 2147483647  # 2 GB - max signed 32-bit integer
    buf_bytes = min(buf_bytes, max_socket_buf)

    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buf_bytes)
    except (OSError, OverflowError, TypeError):
        pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buf_bytes)
    except (OSError, OverflowError, TypeError):
        pass

    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass

    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        pass

    sock.settimeout(None)


def set_socket_timeout(sock: socket.socket, seconds: Optional[float]) -> None:
    """seconds=None => blocking; seconds=float => per-call timeout."""
    sock.settimeout(seconds if seconds is not None else None)


def create_optimized_socket(buf_bytes: Optional[int] = None) -> socket.socket:
    """
    Create a TCP socket and immediately tune it with optimize_socket_for_network().
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    optimize_socket_for_network(sock, buf_bytes=buf_bytes)
    return sock


def connect_with_retry(
    master_ip: str,
    master_port: int,
    retry_interval: float = 2.0,
    buf_bytes: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> socket.socket:
    """
    Try to connect to (master_ip, master_port) with automatic retry.
    """
    attempts = 0
    while True:
        try:
            sock = create_optimized_socket(buf_bytes=buf_bytes)
            sock.connect((master_ip, master_port))
            return sock
        except OSError:
            attempts += 1
            if max_retries is not None and attempts >= max_retries:
                raise RuntimeError(
                    f"Failed to connect to {master_ip}:{master_port} "
                    f"after {attempts} attempts"
                )
            time.sleep(retry_interval)


def listen_and_accept(
    host: str,
    port: int,
    backlog: int = 64,
    buf_bytes: Optional[int] = None,
) -> Tuple[socket.socket, socket.socket, Tuple[str, int]]:
    """
    Bind, listen, and accept exactly one incoming TCP connection.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(backlog)

    client_sock, addr = server_sock.accept()
    optimize_socket_for_network(client_sock, buf_bytes=buf_bytes)
    return server_sock, client_sock, addr


def get_local_ip() -> str:
    """
    Best-effort guess at this machine's LAN IP.
    """
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        probe.close()
        return ip
    except Exception:
        return "127.0.0.1"


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """
    Check whether we can bind host:port right now (True = free / False = busy).
    """
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_sock.bind((host, port))
        test_sock.close()
        return True
    except OSError:
        return False


def find_available_port(
    start_port: int = 29500,
    max_attempts: int = 100,
) -> int:
    """
    Find the first available TCP port in [start_port, start_port+max_attempts).
    """
    for p in range(start_port, start_port + max_attempts):
        if is_port_available(p):
            return p
    raise RuntimeError(
        f"No available port found in range {start_port}-{start_port + max_attempts}"
    )


def safe_socket_close(sock: socket.socket) -> None:
    """
    Gracefully shutdown and close a socket.
    """
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sized send/recv helpers (used by tensor-parallel collectives and uploads)
# ---------------------------------------------------------------------------

def _recvall(sock: socket.socket, n: int) -> bytes:
    """
    Read exactly n bytes from a blocking socket.
    """
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed during recv")
        buf += chunk
    return buf


def send_with_size(sock: socket.socket, data: bytes) -> None:
    """
    Send a 4-byte big-endian length header followed by raw bytes.
    """
    header = struct.pack("!I", len(data))
    sock.sendall(header)
    sock.sendall(data)


def recv_with_size(sock: socket.socket, *, max_len_bytes: Optional[int] = None) -> bytes:
    """
    Receive a 4-byte length header, then that many bytes.

    Args:
        sock: TCP socket in blocking mode.
        max_len_bytes: safety cap. We throw if payload is absurdly large.

    Returns:
        payload bytes.

    Raises:
        ValueError if declared length > max_len_bytes
        ConnectionError if socket closes early.
    """
    # Use 10 GB default if not specified
    if max_len_bytes is None:
        max_len_bytes = 10 * 1024 * 1024 * 1024

    header = _recvall(sock, 4)
    (length,) = struct.unpack("!I", header)
    if length > max_len_bytes:
        raise ValueError(f"Incoming payload too large: {length} bytes (limit: {max_len_bytes} bytes)")
    return _recvall(sock, length)