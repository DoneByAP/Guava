"""
socket_utils.py

Low-level socket helpers for distributed training.

We do a few important things here:
- Disable Nagle (TCP_NODELAY) for low-latency command / activation hops.
- Turn on KEEPALIVE so we eventually notice dead peers.
- Grow SO_RCVBUF / SO_SNDBUF so tensor- or pickle-sized messages don't choke.
- Keep sockets in blocking mode by default. We rely on MessageProtocol timeouts.

Also provides generic sized-send helpers (send_with_size/recv_with_size)
used by tensor-parallel collectives to exchange raw serialized tensors.
"""

import socket
import platform
import time
import struct
from typing import Optional, Tuple


def optimize_socket_for_network(sock: socket.socket, buffer_size: Optional[int] = None) -> None:
    """
    Tune a TCP socket for our training traffic.

    We apply:
    - TCP_NODELAY      -> lower latency for step commands / ACKs
    - SO_KEEPALIVE     -> helps detect dead peers eventually
    - SO_RCVBUF/SNDBUF -> large buffers for chunky payloads (pickled tensors, etc.)
    - settimeout(None) -> blocking by default; higher layers set per-call timeouts
    """
    if buffer_size is None:
        if platform.system() == "Darwin":
            buffer_size = 8 * 1024 * 1024    # 8 MB
        else:
            buffer_size = 16 * 1024 * 1024   # 16 MB

    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)
    sock.settimeout(None)


def set_socket_timeout(sock: socket.socket, seconds: Optional[float]) -> None:
    """seconds=None => blocking; seconds=float => per-call timeout."""
    sock.settimeout(seconds if seconds is not None else None)


def create_optimized_socket(buffer_size: Optional[int] = None) -> socket.socket:
    """Create and optimize a TCP socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    optimize_socket_for_network(sock, buffer_size)
    return sock


def connect_with_retry(
    master_ip: str,
    master_port: int,
    retry_interval: float = 2.0,
    buffer_size: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> socket.socket:
    """
    Connect to (master_ip, master_port) with automatic retry.
    """
    attempts = 0
    while True:
        try:
            sock = create_optimized_socket(buffer_size)
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
    buffer_size: Optional[int] = None,
) -> Tuple[socket.socket, socket.socket, Tuple[str, int]]:
    """
    Bind, listen, and accept exactly one incoming TCP connection.
    (Mostly useful for single-connection tests or simple setups.)
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(backlog)

    client_sock, addr = server_sock.accept()
    optimize_socket_for_network(client_sock, buffer_size)
    return server_sock, client_sock, addr


def get_local_ip() -> str:
    """Best-effort guess at this machine's LAN IP."""
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        probe.close()
        return ip
    except Exception:
        return "127.0.0.1"


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """True if we can bind 'host:port' right now, False if in use."""
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
    """Find the first available TCP port in [start_port, start_port+max_attempts)."""
    for p in range(start_port, start_port + max_attempts):
        if is_port_available(p):
            return p
    raise RuntimeError(
        f"No available port found in range {start_port}-{start_port + max_attempts}"
    )


def safe_socket_close(sock: socket.socket) -> None:
    """Gracefully shutdown and close a socket. Ignore errors."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sized send/recv helpers (used by tensor-parallel collectives)
# ---------------------------------------------------------------------------

def _recvall(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise ConnectionError if closed early."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed during recv")
        buf += chunk
    return buf


def send_with_size(sock: socket.socket, data: bytes) -> None:
    """
    Send a 4-byte big-endian length header followed by the raw bytes.
    """
    header = struct.pack("!I", len(data))
    sock.sendall(header)
    sock.sendall(data)


def recv_with_size(sock: socket.socket, *, max_len_bytes: int = 256 * 1024 * 1024) -> bytes:
    """
    Receive a 4-byte length header then that many bytes.
    Raises ValueError if length exceeds max_len_bytes.
    """
    header = _recvall(sock, 4)
    (length,) = struct.unpack("!I", header)
    if length > max_len_bytes:
        raise ValueError(f"Incoming payload too large: {length} bytes")
    return _recvall(sock, length)
