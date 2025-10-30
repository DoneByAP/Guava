# Guava

**Distributed Neural Network Training Over Network**

A modular, pip-installable framework for distributed deep learning training across multiple GPUs and machines. Built for flexibility, performance, and ease of use.

## 🚀 Features

- **Multiple Parallelism Strategies**
  - **Data Parallelism**: Train on different batches across GPUs
  - **Model Parallelism**: Split model layers across GPUs
  - **Pipeline Parallelism**: Micro-batch pipelining for efficiency
  - **Tensor Parallelism**: Split tensors within layers (single-node)
  - **Hybrid Parallelism**: Combine strategies for maximum scalability

- **Network-Optimized Communication**
  - Optimized socket configurations for low latency and high throughput
  - Automatic compression and serialization
  - Robust error handling and recovery
  - TCP keepalive and buffer tuning

- **Flexible Architecture**
  - Easy integration with any PyTorch model
  - Support for custom data loaders
  - Checkpoint and resume capabilities
  - Comprehensive logging and metrics

- **Production Ready**
  - Automatic reconnection on failures
  - CUDA error recovery
  - Memory management
  - Progress tracking with tqdm

## 📦 Installation

```bash
pip install distrib-train-net
```

Or install from source:

```bash
git clone https://github.com/yourusername/distrib-train-net.git
cd distrib-train-net
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPUs (for GPU training)

## 🎯 Quick Start

### 1. Single Machine, Multiple GPUs (Data Parallel)

```python
from distrib_train_net import DistributedConfig, Orchestrator
import torch.nn as nn

# Your model
class MyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=8),
            num_layers=6
        )
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

# Configure distributed training
config = DistributedConfig(
    vocab_size=50000,
    d_model=512,
    batch_size=32,
    data_parallel=True,  # Use data parallelism
    num_workers=torch.cuda.device_count()
)

# Create orchestrator
orchestrator = Orchestrator(config)

# Register your model
model = MyTransformer(config.vocab_size, config.d_model)
orchestrator.register_model(model)

# Start training
orchestrator.start_training(your_dataloader, num_epochs=10)
```

### 2. Multi-Machine Training (Distributed)

**On Orchestrator Node:**
```python
from distrib_train_net import DistributedConfig, Orchestrator

config = DistributedConfig(
    vocab_size=50000,
    d_model=512,
    num_workers=4,  # Total GPUs across all machines
    master_ip="192.168.1.100",
    master_port=29500,
    data_parallel=True
)

orchestrator = Orchestrator(config, mode='orchestrator')
orchestrator.register_model(model)
orchestrator.start_training(dataloader, num_epochs=10)
```

**On Worker Nodes:**
```python
from distrib_train_net import NetworkWorker, DistributedConfig

config = DistributedConfig.from_env()  # Load from environment

# Start worker process
worker = NetworkWorker(
    gpu_id=0,  # Local GPU ID
    config=config,
    master_ip="192.168.1.100",
    master_port=29500
)

worker.connect_and_train()  # Auto-reconnects on failures
```

### 3. Model Parallelism (Large Models)

```python
config = DistributedConfig(
    vocab_size=50000,
    d_model=4096,  # Large model
    n_layers=48,
    num_workers=4,
    model_parallel=True,  # Split layers across GPUs
    pipeline_parallel=True,  # Enable pipelining
    micro_batches=4  # Pipeline micro-batches
)

orchestrator = Orchestrator(config)
orchestrator.register_model(large_model)
orchestrator.start_training(dataloader, num_epochs=10)
```

## 🔧 Configuration

### Environment Variables

Control behavior via environment variables:

```bash
# Parallelism Strategy
export DATA_PARALLEL=1          # Enable data parallelism (default: 1)
export MODEL_PARALLEL=0         # Enable model parallelism (default: 0)
export PIPELINE_PARALLEL=0      # Enable pipeline parallelism (default: 0)
export TENSOR_PARALLEL=0        # Enable tensor parallelism (default: 0)

# Pipeline Configuration
export MICRO_BATCHES=4          # Number of micro-batches (default: 4)
export TENSOR_PARALLEL_SIZE=2   # GPUs for tensor parallel group (default: 2)

# Communication Timeouts
export ACT_TIMEOUT_SEC=60       # Activation timeout (0 = no timeout)
export ACK_TIMEOUT_SEC=30       # Acknowledgment timeout
export RESENDS_MAX=3            # Max resend attempts
export RESEND_PROBE_SEC=5       # Resend probe interval

# Optimization
export ALLOW_ACT_REUSE=0        # Cache activation reuse (default: 0)
export ACT_CACHE_STEPS=256      # Activation cache size

# Logging
export COMPACT_LOG=1            # Compact logging (default: 1)
export LOG_STEP_EVERY=100       # Log interval
export LOG_ACT=0                # Log activations (default: 0)
```

### Configuration Object

```python
from distrib_train_net import DistributedConfig

config = DistributedConfig(
    # Model architecture
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=1024,
    dropout=0.1,
    
    # Training hyperparameters
    batch_size=16,
    learning_rate=3e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    use_amp=False,
    
    # Distributed setup
    num_workers=4,
    master_ip="localhost",
    master_port=29500,
    
    # Parallelism strategy
    data_parallel=True,
    model_parallel=False,
    pipeline_parallel=False,
    tensor_parallel=False,
    
    # Checkpointing
    checkpoint_dir="./checkpoints",
    save_interval=1000,
)

# Auto-adapt to available GPUs
config.adapt_to_gpus(torch.cuda.device_count())
```

## 📚 Architecture

### Component Overview

```
distrib-train-net/
├── core/
│   ├── config.py           # Configuration management
│   └── __init__.py
├── communication/
│   ├── protocol.py         # Message protocol and serialization
│   └── __init__.py
├── orchestration/
│   ├── orchestrator.py     # Central coordinator
│   └── __init__.py
├── workers/
│   ├── base_worker.py      # Worker base classes
│   ├── network_worker.py   # Network-connected worker
│   └── __init__.py
├── utils/
│   ├── socket_utils.py     # Socket optimization utilities
│   ├── checkpoint.py       # Checkpoint management
│   └── __init__.py
└── examples/
    ├── transformer.py      # Transformer training example
    ├── multi_node.py       # Multi-node setup example
    └── custom_model.py     # Custom model integration
```

### Communication Protocol

The framework uses a length-prefixed message protocol:

```
[4 bytes: length] [N bytes: compressed pickled data]
```

**Message Types:**
- Control: `HELLO`, `READY`, `START_TRAINING`, `STOP_TRAINING`, `HEARTBEAT`
- Data: `BATCH_DATA`, `ACTIVATIONS`, `GRADIENTS`, `LABELS`
- Model: `MODEL_CONFIG`, `MODEL_WEIGHTS`, `MODEL_UPDATE`
- Metrics: `LOSS`, `METRICS`

### Socket Optimizations

- **TCP_NODELAY**: Disabled Nagle's algorithm for low latency
- **SO_KEEPALIVE**: TCP keepalive for connection monitoring
- **Buffer Tuning**: 16MB buffers (8MB on macOS) for high throughput
- **Compression**: zlib compression for large payloads

## 🎓 Examples

### Custom Data Loader Integration

```python
from torch.utils.data import DataLoader

# Your custom dataset
dataset = YourDataset(...)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use with orchestrator
orchestrator = Orchestrator(config)
orchestrator.register_model(model)
orchestrator.start_training(dataloader, num_epochs=10)
```

### Custom Model Integration

```python
from distrib_train_net import DistributedConfig, Orchestrator

class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture
        
    def forward(self, x):
        # Your forward pass
        return output

# Register with framework
model = MyCustomModel()
config = DistributedConfig(...)
orchestrator = Orchestrator(config)
orchestrator.register_model(model)
```

### Multi-Node Cluster Setup

**Setup script for worker nodes:**
```bash
#!/bin/bash
# launch_worker.sh

export CUDA_VISIBLE_DEVICES=0,1  # GPUs to use
export MASTER_IP=192.168.1.100
export MASTER_PORT=29500

python -m distrib_train_net.workers.network_worker \
    --gpu-id 0 \
    --master-ip $MASTER_IP \
    --master-port $MASTER_PORT
```

## 📊 Performance Tips

### 1. Choose the Right Parallelism Strategy

- **Small models, multiple GPUs**: Data parallelism
- **Large models, limited memory**: Model parallelism
- **Very large models**: Pipeline + model parallelism
- **Within-node optimization**: Tensor parallelism

### 2. Network Optimization

- Use high-bandwidth interconnect (InfiniBand, 10GbE+)
- Place orchestrator on fastest node
- Minimize cross-datacenter training
- Use compression for slow networks

### 3. Memory Management

```python
# Enable gradient checkpointing for large models
config = DistributedConfig(
    use_amp=True,  # Mixed precision training
    max_grad_norm=1.0,  # Gradient clipping
)

# Monitor memory usage
orchestrator.enable_memory_profiling()
```

### 4. Batch Size Tuning

```python
# Effective batch size in data parallelism
effective_batch_size = config.batch_size * config.num_workers

# For pipeline parallelism
effective_batch_size = config.batch_size * config.micro_batches
```

## 🐛 Troubleshooting

### Connection Issues

```python
# Increase timeouts
config.activation_timeout = 120.0  # 2 minutes
config.ack_timeout = 60.0

# Enable retries
config.max_resends = 5
config.resend_probe_interval = 10.0
```

### CUDA Out of Memory

```python
# Reduce batch size
config.batch_size = 8

# Enable gradient checkpointing
# (implement in your model)

# Use mixed precision
config.use_amp = True
```

### Network Bottlenecks

```bash
# Monitor network usage
nload  # or
iftop

# Check socket buffer settings
sysctl net.core.rmem_max
sysctl net.core.wmem_max
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built on top of:
- PyTorch for deep learning primitives
- Modern distributed training best practices
- Production-tested networking patterns

## 📮 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/distrib-train-net/issues)
- **Documentation**: [Read the Docs](https://distrib-train-net.readthedocs.io)
- **Discord**: [Join our community](https://discord.gg/yourdiscord)

---

**Made with ❤️ for the ML community**
