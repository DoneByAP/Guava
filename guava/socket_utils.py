"""
Socket utilities for optimized network communication.
"""

import socket
import platform
import time
from typing import Optional, Tuple


def optimize_socket_for_network(sock: socket.socket, buffer_size: Optional[int] = None) -> None:
    """
    Optimize socket for distributed training network communication.

    Applies:
    - TCP_NODELAY: disable Nagle for low latency
    - SO_KEEPALIVE: detect dead peers
    - SO_RCVBUF / SO_SNDBUF: huge buffers for throughput
    - Blocking mode: no timeout by default (we do per-call timeouts elsewhere)

    Args:
        sock: Socket to optimize
        buffer_size: Buffer size in bytes (default: platform-specific)
    """
    # Pick a sane default buffer size per platform
    if buffer_size is None:
        if platform.system() == 'Darwin':  # macOS limit is lower
            buffer_size = 8 * 1024 * 1024   # 8 MB
        else:
            buffer_size = 16 * 1024 * 1024  # 16 MB for Linux/Windows

    # Disable Nagle's algorithm for latency-sensitive hop-to-hop transfers
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Enable TCP keepalive so we eventually notice dead peers
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

    # Enlarge buffers for high-throughput tensor streaming
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)

    # Default to blocking mode (infinite wait). We'll override ad hoc in protocol.
    sock.settimeout(None)


def create_optimized_socket(buffer_size: Optional[int] = None) -> socket.socket:
    """
    Create and optimize a new TCP socket.

    Args:
        buffer_size: Optional custom buffer size in bytes.

    Returns:
        Optimized, blocking-mode TCP socket.
    """
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
    Connect to orchestrator with automatic retry.
    Used by remote workers.

    Args:
        master_ip: Orchestrator IP or hostname
        master_port: Orchestrator port
        retry_interval: Seconds to sleep between attempts
        buffer_size: Passed to optimize_socket_for_network
        max_retries: If None, retry forever. Otherwise stop after this many attempts.

    Returns:
        Connected + optimized socket
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
                    f"Failed to connect to {master_ip}:{master_port} after {attempts} attempts"
                )
            time.sleep(retry_interval)


def listen_and_accept(
    host: str,
    port: int,
    backlog: int = 64,
    buffer_size: Optional[int] = None
) -> Tuple[socket.socket, socket.socket, Tuple[str, int]]:
    """
    Bind, listen, and accept a single incoming connection from a worker.
    Used by the orchestrator.

    This returns:
    - server_sock (listening socket, still open so you can accept more)
    - client_sock (the accepted/connected worker socket, already optimized)
    - addr (client addr tuple)

    You (the orchestrator) typically:
        server_sock, client_sock, addr = listen_and_accept(...)
        # use client_sock for MessageProtocol traffic
        # keep server_sock around to accept next worker
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # allow quick reuse if orchestrator restarts
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(backlog)

    client_sock, addr = server_sock.accept()
    optimize_socket_for_network(client_sock, buffer_size)

    return server_sock, client_sock, addr


def get_local_ip() -> str:
    """
    Try to guess the outward-facing local IP of this machine.

    Returns:
        IP string like '192.168.0.12' or '127.0.0.1' fallback.
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
    Check if a TCP port can be bound.

    Args:
        port: Port num to test
        host: Bind host

    Returns:
        True if bind works, False if in use.
    """
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_sock.bind((host, port))
        test_sock.close()
        return True
    except OSError:
        return False


def find_available_port(start_port: int = 29500, max_attempts: int = 100) -> int:
    """
    Find a free port in a range (useful for orchestrator auto-bind).

    Args:
        start_port: First port to try
        max_attempts: How many consecutive ports to check

    Returns:
        An available port number

    Raises:
        RuntimeError if no available port found
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    raise RuntimeError(
        f"No available port found in range {start_port}-{start_port + max_attempts}"
    )


def safe_socket_close(sock: socket.socket) -> None:
    """
    Shut down and close a socket quietly.

    Args:
        sock: Socket to close
    """
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass
