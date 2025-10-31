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

NEW:
- You can now control socket buffer size per connection (in MB) from
  DistributedConfig.socket_buffer_mb. The worker/orchestrator will
  pass that down as buf_bytes to let Windows/Linux ask for 32MB, 64MB, etc.
  macOS will clamp lower and that's fine.
"""

import socket
import platform
import time
import struct
from typing import Optional, Tuple


def optimize_socket_for_network(sock: socket.socket, buf_bytes: Optional[int] = None) -> None:
    """
    Tune a TCP socket for Guava traffic.

    Args:
        sock: the TCP socket we are about to use.
        buf_bytes: desired SO_SNDBUF / SO_RCVBUF in bytes.
                   If None, we pick a platform-aware default:
                       - macOS: 8 MB
                       - everything else: 16 MB
                   (Windows can happily go higher, e.g. 64 MB, and we'll try.)

    What we do:
    - TCP_NODELAY      => lower latency for command/ACK bursts
    - SO_KEEPALIVE     => helps detect dead peers over time
    - SO_SNDBUF/RCVBUF => bigger buffers for fat tensor payloads
    - settimeout(None) => leave the socket in blocking mode by default
    """
    if buf_bytes is None:
        if platform.system() == "Darwin":
            buf_bytes = 8 * 1024 * 1024      # ~8 MB default on macOS
        else:
            buf_bytes = 16 * 1024 * 1024     # ~16 MB default on Windows/Linux

    # Ask OS for bigger TCP buffers. OS may clamp lower (esp. macOS).
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buf_bytes)
    except OSError:
        pass
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buf_bytes)
    except OSError:
        pass

    # Turn off Nagle so tiny control messages don't batch.
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass

    # Keepalive so a totally-dead peer eventually gets noticed.
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        pass

    # Blocking mode. Higher layers decide per-call timeout.
    sock.settimeout(None)


def set_socket_timeout(sock: socket.socket, seconds: Optional[float]) -> None:
    """seconds=None => blocking; seconds=float => per-call timeout."""
    sock.settimeout(seconds if seconds is not None else None)


def create_optimized_socket(buf_bytes: Optional[int] = None) -> socket.socket:
    """
    Create a TCP socket and immediately tune it with optimize_socket_for_network().

    buf_bytes:
        Desired buffer size in bytes (see optimize_socket_for_network()).
        Example: 64 * 1024 * 1024 for ~64MB buffers on Windows/Linux.
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

    We'll keep retrying forever unless max_retries is provided.
    Each attempt creates a new optimized socket with the requested buf_bytes.

    Returns:
        A connected, tuned socket.
    Raises:
        RuntimeError if we hit max_retries without success.
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

    Returns:
        (server_sock, client_sock, addr)

    Notes:
    - We tune ONLY the accepted client_sock with optimize_socket_for_network(),
      not the listening socket.
    - This is mainly for simple single-shot exchanges (tests, checkpoints, etc.).
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
    We'll try to open a dummy UDP socket to 8.8.8.8 and read the chosen iface.
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
    Used by orchestrator when auto-picking control/grad/metrics sockets.
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

    We walk forward linearly and return the first one we can bind.
    If none are available in that range, raise.
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
    Ignore any errors because at shutdown the peer might already be gone.
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
    Raises ConnectionError if the peer closes early.
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
    This lets the receiver know exactly how many bytes to read.
    """
    header = struct.pack("!I", len(data))
    sock.sendall(header)
    sock.sendall(data)


def recv_with_size(sock: socket.socket, *, max_len_bytes: int = 256 * 1024 * 1024) -> bytes:
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
    header = _recvall(sock, 4)
    (length,) = struct.unpack("!I", header)
    if length > max_len_bytes:
        raise ValueError(f"Incoming payload too large: {length} bytes")
    return _recvall(sock, length)
