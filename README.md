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

## 🎯 Quick Start (TinyToyModel Demo)

Guava includes two launch scripts that demonstrate orchestrated data-parallel training with a minimal `TinyToyModel`. Launch the orchestrator on the control node (CPU is fine), then connect one or more GPU workers.

### 1. Launch the Orchestrator

```bash
python orchestrator_train.py     --master-ip 0.0.0.0     --master-port 29500     --gpus 2     --train-batches 100     --val-interval 20
```

What this does:
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

Each worker host will:
- Parse the comma-separated `--gpu-ids` list (e.g. `"0,1"`)
- Build a `DistributedConfig` that mirrors the orchestrator
- Construct a per-thread `TinyToyModel` factory to keep parameters aligned
- Start a `NetworkWorker` thread per GPU that calls `connect_and_train()`
- Block until all threads exit

⚠ Important: `batch_size`, `vocab_size`, `d_model`, `n_layers`, and `n_heads` must match across orchestrator and workers. You are responsible for keeping them aligned.

### 3. Monitor Training

Both scripts print detailed startup banners so you can confirm:
- The orchestrator is bound to the intended IP/port
- Each worker reports the correct CUDA device and world size
- Workers successfully register before the orchestrator begins training

When training finishes:
- The orchestrator writes `orchestrator_final.pt` to the checkpoint dir
- Workers shut down cleanly

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

orch.save_checkpoint(f"{cfg.checkpoint_dir}/orchestrator_final.pt")
```

CLI flags (orchestrator):
- `--gpus`: how many total workers you expect
- `--train-batches`: number of fake batches per epoch
- `--val-interval`: run validation every N steps (0 turns off val)
- `--worker-timeout`: how long to wait for workers to show up

---

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

CLI flags (worker):
- `--gpu-ids`: comma-separated list of local CUDA device indices
- `--world-size`: total number of workers across all machines
- `--batch-size`, `--vocab-size`, `--d-model`, `--n-layers`, `--n-heads`: must match orchestrator

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

Message framing:
```text
[4 bytes: length] [N bytes: compressed pickled data]
```

**Message Types:**
- Control: `HELLO`, `READY`, `START_TRAINING`, `STOP_TRAINING`, `HEARTBEAT`
- Data: `BATCH_DATA`, `ACTIVATIONS`, `GRADIENTS`, `LABELS`
- Model: `MODEL_CONFIG`, `MODEL_WEIGHTS`, `MODEL_UPDATE`
- Metrics: `LOSS`, `METRICS`

**Socket tuning:**
- `TCP_NODELAY`
- `SO_KEEPALIVE`
- Large socket buffers (16MB, 8MB on macOS)
- `zlib` compression for payloads

---

## 📊 Performance Tips

### Choose the Right Parallelism Strategy
- Small models → **Data parallelism**
- Large models → **Model parallelism**
- Very large models → **Pipeline + model parallelism**
- Within-node → **Tensor parallelism**

### Network Optimization
- Use InfiniBand / 10GbE+
- Keep the orchestrator near the workers
- Use compression for slower networks

### Memory Management

```python
config = DistributedConfig(
    use_amp=True,
    max_grad_norm=1.0,
)
```

*Optional:* `orchestrator.enable_memory_profiling()`

### Batch Size Tuning

```python
effective_batch_size = cfg.batch_size * cfg.num_workers
```

---

## 🐛 Troubleshooting

> **A Dev's Note on Errors**
>
> GPU error codes can get weird. During training (and sometimes inference), a simple restart often clears transient issues. Don't waste a night chasing ghosts unless it keeps happening.

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

1. Fork the repo  
2. Create a feature branch  
3. Add tests  
4. Open a PR  

---

## 🪪 License

Guava is licensed under a dual model:

- **Community Edition:** Apache 2.0.  
  You can use, modify, and ship it for personal, research, or non-commercial work.

- **Commercial Edition:** A commercial license is required for companies or products that integrate Guava into something paid (for example: managed GPU training, orchestration as a service, etc).

For commercial licensing inquiries:
📧 azanipeterking@gmail.com

---

## 📮 Support

- Issues: GitHub Issues  
- Docs: coming soon  
- Discord: coming soon  

**Made with ❤️ for the ML community**
