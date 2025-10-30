"""
distrib-train-net: Distributed Neural Network Training Over Network

A modular framework for distributed training across multiple GPUs and machines
with support for:
- Data Parallelism: Different batches across GPUs
- Model Parallelism: Layers split across GPUs  
- Pipeline Parallelism: Micro-batch pipelining
- Tensor Parallelism: Split tensors within layers

Example Usage:
    ```python
    from distrib_train_net import DistributedConfig, Orchestrator, NetworkWorker
    
    # Configure distributed training
    config = DistributedConfig(
        data_parallel=True,
        num_workers=2,
        batch_size=32
    )
    
    # On orchestrator node
    orchestrator = Orchestrator(config)
    orchestrator.register_model(your_model)
    orchestrator.start_training()
    
    # On worker nodes
    worker = NetworkWorker(gpu_id=0, master_ip="192.168.1.100")
    worker.connect_and_train()
    ```
"""

__version__ = "0.1.1"

from .config import DistributedConfig, ParallelismStrategy
from .orchestrator import Orchestrator
from .network_worker import NetworkWorker
from .protocol import MessageType, MessageProtocol
from .socket_utils import optimize_socket_for_network

__all__ = [
    "DistributedConfig",
    "ParallelismStrategy",
    "Orchestrator",
    "NetworkWorker",
    "MessageType",
    "MessageProtocol",
    "optimize_socket_for_network",
    "__version__",
]
