# Guava

**Distributed Neural Network Training Over Network**

A modular, pip-installable framework for distributed deep learning training across multiple GPUs and machines. Built for flexibility, performance, and ease of use.

---

## 🚀 Features

### Multiple Parallelism Strategies
- **Data Parallelism:** Train on different batches across GPUs  
- **Model Parallelism:** Split model layers across GPUs  
- **Pipeline Parallelism:** Micro-batch pipelining for efficiency  
- **Tensor Parallelism:** Split tensors within layers (single-node)  
- **Hybrid Parallelism:** Combine strategies for maximum scalability  

### Network-Optimized Communication
- Optimized socket configurations for low latency and high throughput  
- Automatic compression and serialization  
- Robust error handling and recovery  
- TCP keepalive and buffer tuning  

### Flexible Architecture
- Easy integration with any PyTorch model  
- Support for custom data loaders  
- Checkpoint and resume capabilities  
- Comprehensive logging and metrics  

### Runtime Behavior
- Automatic reconnection on failures  
- CUDA error recovery  
- Memory management  
- Progress tracking with tqdm  

---

## 📦 Installation

```bash
pip install guava
```

Or install from source:

```bash
git clone https://github.com/yourusername/guava.git
cd guava
pip install -e .
```

### Requirements
- Python >= 3.8  
- PyTorch >= 2.0.0  
- CUDA-capable GPUs (for GPU training)

---

## 🎯 Quick Start

### 1. Single Machine, Multiple GPUs (Data Parallel)
```python
from guava import DistributedConfig, Orchestrator
import torch
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
    data_parallel=True, # Use data parallelism
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
from guava import DistributedConfig, Orchestrator

config = DistributedConfig(
    vocab_size=50000,
    d_model=512,
    num_workers=4, # Total GPUs across all machines
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
from guava import NetworkWorker, DistributedConfig

config = DistributedConfig.from_env() # Load from environment

worker = NetworkWorker(
    gpu_id=0,
    config=config,
    model_ctor=lambda: MyTransformer(config.vocab_size, config.d_model),
    master_ip="192.168.1.100",
    master_port=29500
)
worker.connect_and_train()
```

### 3. Model Parallelism (Large Models)
```python
from guava import DistributedConfig, Orchestrator

config = DistributedConfig(
    vocab_size=50000,
    d_model=4096,
    n_layers=48,
    num_workers=4,
    model_parallel=True,
    pipeline_parallel=True,
    micro_batches=4
)

orchestrator = Orchestrator(config)
orchestrator.register_model(large_model)
orchestrator.start_training(dataloader, num_epochs=10)
```

---

## 🔧 Configuration

### Environment Variables
```bash
# Parallelism Strategy
export DATA_PARALLEL=1
export MODEL_PARALLEL=0
export PIPELINE_PARALLEL=0
export TENSOR_PARALLEL=0

# Pipeline Configuration
export MICRO_BATCHES=4
export TENSOR_PARALLEL_SIZE=2

# Communication Timeouts
export ACT_TIMEOUT_SEC=60
export ACK_TIMEOUT_SEC=30
export RESENDS_MAX=3
export RESEND_PROBE_SEC=5

# Optimization
export ALLOW_ACT_REUSE=0
export ACT_CACHE_STEPS=256

# Logging
export COMPACT_LOG=1
export LOG_STEP_EVERY=100
export LOG_ACT=0
```

### Configuration Object
```python
from guava import DistributedConfig
import torch

config = DistributedConfig(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=1024,
    dropout=0.1,
   
    batch_size=16,
    learning_rate=3e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    use_amp=False,
   
    num_workers=4,
    master_ip="localhost",
    master_port=29500,
   
    data_parallel=True,
    model_parallel=False,
    pipeline_parallel=False,
    tensor_parallel=False,
   
    checkpoint_dir="./checkpoints",
    save_interval=1000,
)

config.adapt_to_gpus(torch.cuda.device_count())
```

---

## 📚 Architecture

### Component Overview
```
guava/
├── config.py
├── protocol.py
├── orchestrator.py
├── base_worker.py
├── network_worker.py
├── socket_utils.py
└── __init__.py
```

### Communication Protocol
The framework uses a length-prefixed message protocol:
```
[4 bytes: length] [N bytes: compressed pickled data]
```

**Message Types:**
- Control: HELLO, READY, START_TRAINING, STOP_TRAINING, HEARTBEAT  
- Data: BATCH_DATA, ACTIVATIONS, GRADIENTS, LABELS  
- Model: MODEL_CONFIG, MODEL_WEIGHTS, MODEL_UPDATE  
- Metrics: LOSS, METRICS  

**Socket Optimizations:**
- TCP_NODELAY  
- SO_KEEPALIVE  
- 16MB buffers (8MB on macOS)  
- zlib compression  

---

## 🎓 Examples

### Custom Data Loader Integration
```python
from torch.utils.data import DataLoader
from guava import Orchestrator, DistributedConfig
import torch.nn as nn

dataset = YourDataset(...)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

orchestrator = Orchestrator(config)
orchestrator.register_model(model)
orchestrator.start_training(dataloader, num_epochs=10)
```

### Custom Model Integration
```python
from guava import DistributedConfig, Orchestrator
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture
       
    def forward(self, x):
        return output

model = MyCustomModel()
config = DistributedConfig(...)
orchestrator = Orchestrator(config)
orchestrator.register_model(model)
```

### Multi-Node Cluster Setup
```bash
#!/bin/bash
# launch_worker.sh
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_IP=192.168.1.100
export MASTER_PORT=29500
python -m guava.network_worker \
    --gpu-id 0 \
    --master-ip $MASTER_IP \
    --master-port $MASTER_PORT
```

---

## 📊 Performance Tips

### Choose the Right Parallelism Strategy
- Small models → **Data parallelism**  
- Large models → **Model parallelism**  
- Very large models → **Pipeline + model parallelism**  
- Within-node → **Tensor parallelism**

### Network Optimization
- Use InfiniBand / 10GbE+  
- Keep orchestrator near workers  
- Use compression for slower networks  

### Memory Management
```python
config = DistributedConfig(
    use_amp=True,
    max_grad_norm=1.0,
)
```
*(Optional)*: `orchestrator.enable_memory_profiling()`

### Batch Size Tuning
```python
effective_batch_size = config.batch_size * config.num_workers
# or with pipeline parallelism
effective_batch_size = config.batch_size * config.micro_batches
```

---

## 🐛 Troubleshooting

### ⚠️ A Devs Note on Errors
<div style="background: #fff8dc; border-left: 5px solid #ffcc00; padding: 12px 16px; border-radius: 6px; box-shadow: 0 0 12px rgba(255, 204, 0, 0.4);">

<b>Honestly, GPU error codes can get a little funky in my opinion.</b><br><br>
In your development, consider that a lot of the time during training (and sometimes inference),  
a simple restart is all that’s needed and you’ll progress until the next error.<br><br>
Obscure computational or memory errors lurk — you don’t always need to hunt them down  
until you absolutely have to.
</div>

### Connection Issues
```python
config.activation_timeout = 120.0
config.ack_timeout = 60.0
config.max_resends = 5
config.resend_probe_interval = 10.0
```

### CUDA Out of Memory
```python
config.batch_size = 8
config.use_amp = True
```

### Network Bottlenecks
```bash
nload
iftop
sysctl net.core.rmem_max
sysctl net.core.wmem_max
```

---

## 🤝 Contributing
1. Fork the repository  
2. Create a feature branch  
3. Add tests  
4. Submit a pull request  

---

## 📄 License
MIT License — see `LICENSE` file for details.

---

## 🙏 Acknowledgments
Built on top of:
- PyTorch  
- Modern distributed training best practices  
- Practical networking patterns  

---

## 📮 Support
- **Issues:** GitHub Issues  
- **Documentation:** coming soon  
- **Discord:** coming soon  

**Made with ❤️ for the ML community**
