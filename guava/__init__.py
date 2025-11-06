"""
Guava — Distributed Neural Network Training + Energy Telemetry

A modular framework for orchestrating distributed training across
multiple GPUs and machines.

Supports:
- Data Parallelism: Different batches across GPUs
- Model Parallelism: Split model layers across GPUs
- Pipeline Parallelism: Micro-batch pipelining
- Tensor Parallelism: Intra-layer tensor slicing
- Hybrid parallelism: Combinations allowed

✨ Energy-Aware ML
- Real-time cross-platform hardware telemetry
- GPU/CPU/RAM/IO/Network joule accounting
- DARPA-style first-order energy semantics output
- Training phase tagging (forward/backward/sync/idle)

Example Usage:

    from guava import DistributedConfig, Orchestrator, NetworkWorker
    
    cfg = DistributedConfig(data_parallel=True, num_workers=2, batch_size=32)

    # Orchestrator node
    orch = Orchestrator(cfg)
    orch.register_model(your_model)
    orch.start_training()

    # Worker node
    worker = NetworkWorker(gpu_id=0, master_ip="192.168.1.100")
    worker.connect_and_train()
"""

__version__ = "0.7.3"  # 🚀 energy-aware upgrade

from .config import DistributedConfig, ParallelismStrategy
from .orchestrator import Orchestrator
from .network_worker import NetworkWorker
from .energy_monitor import EnergyMonitor
from .protocol import MessageType, MessageProtocol
from .socket_utils import optimize_socket_for_network

__all__ = [
    "DistributedConfig",
    "ParallelismStrategy",
    "Orchestrator",
    "NetworkWorker",
    "EnergyMonitor",
    "MessageType",
    "MessageProtocol",
    "optimize_socket_for_network",
    "__version__",
]

# -------------------------------------------------------------------
# 🪟 Windows OHM Auto-Init (silent + safe)
# -------------------------------------------------------------------
import platform
if platform.system() == "Windows":
    try:
        from .post_install import run as _guava_post_install
        _guava_post_install()
    except Exception:
        # Fail silently — no breakage if OHM missing
        pass
