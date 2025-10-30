# Guava

**Distributed Neural Network Training Over Network**

Guava is a modular framework for orchestrating distributed PyTorch training across multiple GPUs and machines. It ships with ready-to-run orchestration and worker scripts that make it easy to spin up a tiny end-to-end demo or adapt the infrastructure to your own model.

---

## 🚀 Features

### Multiple Parallelism Strategies
- **Data Parallelism:** Train on different batches across GPUs
- **Model Parallelism:** Split model layers across GPUs
- **Pipeline Parallelism:** Micro-batch pipelining for efficiency
- **Tensor Parallelism:** Split tensors within layers (single-node)
- **Hybrid Parallelism:** Combine strategies for maximum scalability

### Network-Optimized Communication
- Tuned socket options for low latency and high throughput
- Automatic compression and serialization
- Robust error handling and reconnection logic
- TCP keepalive and buffer tuning

### Flexible Architecture
- Works with any PyTorch `nn.Module`
- Supports custom data loaders and datasets
- Checkpoint and resume capabilities
- Comprehensive logging and metrics hooks

### Runtime Behavior
- Automatic reconnection on failures
- CUDA error recovery helpers
- Graceful shutdown handling
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

## 🖥 GPU Worker Setup (CUDA on Linux)

Every GPU worker box needs:
1. An NVIDIA GPU
2. NVIDIA drivers for that GPU
3. CUDA toolkit (gives you `nvcc`, CUDA runtime, etc.)
4. PyTorch built/installed with CUDA support

You can download CUDA directly from NVIDIA here:
https://developer.nvidia.com/cuda-downloads

Below is a typical Ubuntu install flow for CUDA Toolkit using NVIDIA's apt repo.  
(This is an example; adjust the Ubuntu version in the URL if you're not on this release.)

```bash
# 1. (Recommended) Update packages
sudo apt-get update
sudo apt-get install -y wget gnupg

# 2. Add NVIDIA's CUDA repo keyring (example for Ubuntu 22.04 / x86_64)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 3. Install CUDA toolkit
sudo apt-get -y install cuda-toolkit

# 4. (Optional) Add CUDA to PATH for this shell
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 5. Verify (should print CUDA compiler version)
nvcc --version
```

Notes:
- You don't have to install the full CUDA *driver* from this script if your driver is already installed via your distro or vendor.
- `cuda-toolkit` gives you user-space CUDA libs that PyTorch will use.
- All worker nodes must be on compatible driver / CUDA so math kernels behave the same.

---

## 🎯 Quick Start (TinyToyModel Demo)

Guava includes two launch scripts that demonstrate orchestrated data-parallel training with a minimal `TinyToyModel`. Launch the orchestrator on the control node (CPU is fine), then connect one or more GPU workers.

### 1. Launch the Orchestrator

```bash
python orchestrator_train.py     --master-ip 0.0.0.0     --master-port 29500     --gpus 2     --train-batches 100     --val-interval 20
```

Key behaviors:
- Builds the same `TinyToyModel` used on the workers
- Constructs a `DistributedConfig` mirroring worker hyperparameters
- Generates deterministic toy batches and labels
- Waits for workers to register before starting the training loop
- Periodically runs validation if `--val-interval` > 0
- Saves a checkpoint to `--checkpoint-dir` when training completes

### 2. Launch GPU Workers

Run this once per machine that contributes GPUs. The script spawns a `NetworkWorker` thread for each GPU ID you provide.

```bash
python guava_worker.py     --gpu-ids 0,1     --master-ip 192.168.0.177     --master-port 29500     --world-size 2
```

What happens on each worker host:
- Parses the comma-separated `--gpu-ids` list (e.g. "0,1")
- Builds a `DistributedConfig` that mirrors the orchestrator
- Constructs a per-thread `TinyToyModel` factory to keep parameters aligned
- Starts a `NetworkWorker` thread per GPU that calls `connect_and_train()`
- Blocks the main process until all worker threads exit

> **Config parity is critical** — `batch_size`, `vocab_size`, `d_model`, `n_layers`, and `n_heads` must match across the orchestrator and all workers.

### 3. Monitor Training

Both scripts print detailed startup banners so you can confirm:
- The orchestrator is bound to the intended IP/port
- Each worker reports the correct CUDA device and world size
- Workers successfully register before the orchestrator begins training

When training finishes, the orchestrator stores `orchestrator_final.pt` in the configured checkpoint directory and all worker threads shut down cleanly.

---

## 🧩 Script Reference

### `orchestrator_train.py`

```python
orch = Orchestrator(cfg)
model = TinyToyModel(vocab_size=cfg.vocab_size, d_model=cfg.d_model)

orch.register_model(model)
orch.wait_for_workers(timeout=args.worker_timeout)

orch.start_training(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=args.epochs,
    val_interval=max(args.val_interval, 1),
)

orch.save_checkpoint(f"{cfg.checkpoint_dir}/orchestrator_final.pt}")
```

CLI flags you'll care about:
- `--gpus`: total worker count the orchestrator expects
- `--train-batches`: number of toy batches per epoch
- `--val-interval`: run validation every N steps (0 disables validation)
- `--worker-timeout`: optional timeout while waiting for workers to register

### `guava_worker.py`

```python
runtime = NetworkWorker(
    gpu_id=gpu_id,
    config=cfg,
    model_ctor=model_ctor,
    master_ip=master_ip,
    master_port=master_port,
)
runtime.connect_and_train()
```

CLI flags you'll care about:
- `--gpu-ids`: comma-separated list of local CUDA device indices
- `--world-size`: total number of workers across all machines
- `--batch-size`, `--vocab-size`, `--d-model`, `--n-layers`, `--n-heads`: must match the orchestrator

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

cfg = DistributedConfig()
cfg.master_ip = args.master_ip
cfg.master_port = args.master_port
cfg.num_workers = args.gpus
cfg.batch_size = args.batch_size
cfg.vocab_size = args.vocab_size
cfg.d_model = args.d_model
cfg.n_layers = args.n_layers
cfg.n_heads = args.n_heads
cfg.data_parallel = True
```

---

## 📚 Architecture

### Component Overview

```text
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

```text
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

## 📊 Performance Tips

### Choose the Right Parallelism Strategy
- Small models → **Data parallelism**
- Large models → **Model parallelism**
- Very large models → **Pipeline + model parallelism**
- Within-node → **Tensor parallelism**

### Network Optimization
- Use InfiniBand / 10GbE+
- Keep the orchestrator near workers
- Use compression for slower networks

### Memory Management

```python
config = DistributedConfig(
    use_amp=True,
    max_grad_norm=1.0,
)
```

*Optional*: `orchestrator.enable_memory_profiling()`

### Batch Size Tuning

```python
effective_batch_size = cfg.batch_size * cfg.num_workers
```

---

## 🐛 Troubleshooting

> **A Dev's Note on Errors**
>
> GPU error codes can get funky. During training (and sometimes inference),
> a simple restart often clears transient issues. Don't hunt down obscure
> computational or memory errors until you absolutely have to.

### Connection Issues

```python
cfg.activation_timeout = 120.0
cfg.ack_timeout = 60.0
cfg.max_resends = 5
cfg.resend_probe_interval = 10.0
```

### CUDA Out of Memory

```python
cfg.batch_size = 8
cfg.use_amp = True
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

## 🪪 License

Guava is licensed under a dual license model:

- **Community Edition:** Apache License 2.0. You can use, modify, and distribute it for personal, research, or non‑commercial work.
- **Commercial Edition:** A commercial license is required for organizations or products that integrate Guava into a paid offering or provide Guava-based services (managed GPU clusters, training-as-a-service, etc.).

For commercial licensing inquiries, contact:  
📧 azanipeterking@gmail.com

---

## 📮 Support

- **Issues:** GitHub Issues  
- **Documentation:** coming soon  
- **Discord:** coming soon  

**Made with ❤️ for the ML community**
