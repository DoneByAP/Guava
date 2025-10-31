# Guava

**Distributed Neural Network Training Over Network**

Guava is a modular, socket-driven framework for orchestrating distributed PyTorch training across multiple GPUs and machines. It gives you:

- an **Orchestrator** (the "brain") that lives on CPU and coordinates training
- one or more **NetworkWorker** processes (one per GPU) that actually run the model/shard
- message-level parallelism primitives (data / model / pipeline / tensor) without requiring `torch.distributed`

This README reflects the current pip package layout:

- `config.py`
- `base_worker.py`
- `orchestrator.py`
- `network_worker.py`
- `protocol.py`
- `socket_utils.py`
- `__init__.py`
- plus the example launch scripts (`orchestrator_train.py`, `guava_worker.py`)

---

## Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Core Modules](#-core-modules)
- [Quick Start](#-quick-start-cluster-layout)
- [Distributed Execution Model](#-distributed-execution-model)
- [Parallelism Modes](#-parallelism-modes)
- [Configuration](#-configuration)
- [Networking / Sockets](#-networking--sockets)
- [Tensor Parallel](#-tensor-parallel)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Support](#-support)

---

## 🚀 Features

### Multiple Parallelism Strategies
- **Data Parallelism**  
  Each GPU holds a full copy of the model and trains on its own batch. Grads are averaged.
- **Model Parallelism**  
  The model is split by layers across GPUs. Each worker holds only a slice of the transformer.
- **Pipeline Parallelism**  
  Micro-batch style: activations flow forward shard → shard, gradients flow backward shard ← shard.
- **Tensor Parallelism**  
  We shard *inside* a layer (e.g. split weight matrices across GPUs and gather results).
- **Hybrid**  
  You can enable multiple flags and Guava will treat the system as hybrid.

### Orchestrator-Driven Training
- Orchestrator (`Orchestrator`) runs on CPU.
- Workers (`NetworkWorker`) run on GPUs.
- The orchestrator:
  - dispatches training steps to workers
  - receives gradients and metrics
  - averages grads
  - applies the optimizer step on the authoritative "master" model
  - handles multi-stage backward across pipeline shards

### Socket-Level Control Plane
- No NCCL requirement in the core loop.
- Each worker registers with `CONTROL_HELLO`.
- All messages use a length-prefixed pickle+zlib protocol (`MessageProtocol`) with strongly-typed `MessageType`s.
- Multiple TCP ports are reserved for different traffic classes:
  - `+0` control / ACK / backward-ready
  - `+1` metrics
  - `+2` gradients
  - `+7` checkpoints
  - (others reserved for activation relay, heartbeat, tensor collectives)

### Reliability / Safety Knobs
- Explicit ACK barriers between orchestrator and workers.
- Infinite/blocking waits by default on control sockets.
- Keepalive, large send/recv buffers, `TCP_NODELAY`.
- Optional compression for large tensors.

### Checkpointing
- Each worker can upload its shard weights to the orchestrator over the checkpoint channel.
- Orchestrator can save the full ("authoritative") model state dict.

---

## 📦 Installation

You need two things:
1. **Correct NVIDIA stack on your GPU boxes** (CUDA driver + CUDA toolkit that matches what PyTorch expects).
2. **PyTorch built for that stack**, then Guava.

### Step 0. (GPU worker machines ) Install NVIDIA driver + CUDA toolkit FIRST
Workers that run on GPUs *must* have a CUDA runtime/toolkit + driver that match the wheel you're about to install.  
Get it from NVIDIA:

👉 https://developer.nvidia.com/cuda-downloads

Pick the version that matches your GPU, OS, and what PyTorch will expect (for example CUDA 12.6).  
After install, you should be able to run:

```powershell
nvidia-smi
```

and see something like:

```text
CUDA Version: 12.6
Driver Version: 560.xx
```

Your orchestrator (CPU head node) does **not** need CUDA. Your workers **do**.

### Step 1. Install PyTorch for *your* machine
Go to the official PyTorch selector and copy/paste the command it gives you:

👉 https://pytorch.org/get-started/locally/

Why: different GPUs / CUDA driver versions / OSes / Python versions need different wheels.  
Example (Windows, CUDA 12.6) might look like:

```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Your exact command may use:
- `cu126`, `cu124`, etc. for NVIDIA CUDA builds
- or `cpu` wheels if you're just running the orchestrator on a CPU box

Then sanity check:

```powershell
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('gpu_count=', torch.cuda.device_count())"
```

If `cuda_available=True` and `gpu_count>0`, that machine is ready to act as a GPU worker.
If it's `False`, you can still run the orchestrator there (CPU-only is fine).

### Step 2. Install Guava
Once PyTorch is working in that venv:

**Normal / public (when published):**
```bash
pip install guava
```

**During development (TestPyPI sandbox):**
```bash
pip install --extra-index-url https://test.pypi.org/simple/ guava
```

Now you can import the primitives:

```python
from guava import DistributedConfig, Orchestrator, NetworkWorker
```

---

## 🧠 Core Modules

### `DistributedConfig` (`config.py`)
Central config object shared by orchestrator + all workers.

It defines:
- model shape (`vocab_size`, `d_model`, `n_layers`, etc.)
- training hyperparams (`batch_size`, `learning_rate`, etc.)
- cluster layout (`num_workers`, `master_ip`, `master_port`)
- parallelism mode (`data_parallel`, `model_parallel`, `pipeline_parallel`, `tensor_parallel`)
- tensor-parallel group size (`tensor_parallel_size`)
- pipeline micro-batching (`micro_batches`)
- socket tuning / retry / timeout policy
- logging cadence
- checkpoint directory

Useful helpers:
- `adapt_to_gpus(num_gpus)`
- `get_layers_per_gpu()`
- `tensor_parallel_groups()`
- `get_parallelism_strategy()`
- `from_env()`, `to_dict()`, `from_dict()`

### `BaseWorker`, `DataParallelWorker`, `ModelShardWorker` (`base_worker.py`)
Abstract/derived worker classes that actually run compute on a specific GPU:
- hold full model (data parallel) **or** a layer slice (`[layer_start:layer_end)`)
- run forward/backward locally
- capture and send gradients
- apply optimizer steps when appropriate
- cooperate in pipeline backward and tensor-parallel collectives

### `Orchestrator` (`orchestrator.py`)
Runs on CPU and:
- listens for workers to `CONTROL_HELLO`
- dispatches steps (`CONTROL_DATA_PARALLEL_STEP`, `CONTROL_PIPELINE_PHASE1`, etc.)
- waits for gradients via `GRADIENTS_UPLOAD`
- averages / applies grads to the master copy of the model
- performs validation
- saves checkpoints

### `NetworkWorker` (`network_worker.py`)
Runs on each GPU box:
- builds or slices the model for that GPU
- opens a long-lived control socket to the orchestrator
- executes forward/backward on demand
- uploads gradients and metrics
- handles checkpoint shard upload on shutdown

### `protocol.py`
Defines:
- `MessageType` enum for every command / upload / ACK
- `Message` dataclass that moves over the wire
- `MessageProtocol` helpers for length-prefixed, zlib-compressed pickle blobs

### `socket_utils.py`
Low-level TCP tuning and helpers:
- `TCP_NODELAY`, `SO_KEEPALIVE`
- big socket buffers (MBs)
- blocking reads with `[size][payload]` framing
- retry helpers

---

## 🚦 Quick Start (Cluster Layout)

Below are the actual example scripts you ship: `orchestrator_train.py` (head node) and `guava_worker.py` (GPU node).

### 1. On the Orchestrator Node (CPU / head box)

```bash
python orchestrator_train.py ^
  --master-ip 0.0.0.0 ^
  --master-port 29500 ^
  --gpus 2 ^
  --train-batches 100 ^
  --val-interval 20
```

What `orchestrator_train.py` does (simplified):
```python
import torch
from guava import DistributedConfig, Orchestrator

# TinyToyModel is defined INSIDE orchestrator_train.py (and must match the worker's version)
class TinyToyModel(torch.nn.Module):
    def __init__(self, vocab_size=100, d_model=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.linear = torch.nn.Linear(d_model, vocab_size)
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        return x

cfg = DistributedConfig(
    master_ip="0.0.0.0",
    master_port=29500,
    num_workers=2,          # --gpus
    batch_size=2,
    vocab_size=100,
    d_model=32,
    n_layers=2,
    n_heads=4,
    checkpoint_dir="./checkpoints",
    data_parallel=True,
    model_parallel=False,
    pipeline_parallel=False,
    tensor_parallel=False,
)

orch = Orchestrator(cfg)
model = TinyToyModel(vocab_size=cfg.vocab_size, d_model=cfg.d_model)
orch.register_model(model)

# build toy train/val loaders (random token data)
# ...
orch.wait_for_workers(timeout=None)
orch.start_training(train_loader, val_loader, num_epochs=1, val_interval=25)
orch.save_checkpoint(f"{cfg.checkpoint_dir}/orchestrator_final.pt")
```

### 2. On the GPU Worker Node(s)

```bash
python guava_worker.py ^
  --gpu-ids 0,1 ^
  --master-ip 192.168.0.177 ^
  --master-port 29500 ^
  --world-size 2 ^
  --batch-size 2 ^
  --vocab-size 100 ^
  --d-model 32 ^
  --n-layers 2 ^
  --n-heads 4
```

What `guava_worker.py` does (simplified):
```python
import torch
from guava import DistributedConfig, NetworkWorker

# TinyToyModel MUST MATCH orchestrator_train.py
class TinyToyModel(torch.nn.Module):
    def __init__(self, vocab_size=100, d_model=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.linear = torch.nn.Linear(d_model, vocab_size)
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        return x

cfg = DistributedConfig(
    master_ip="192.168.0.177",
    master_port=29500,
    num_workers=2,          # MUST == orchestrator --gpus
    batch_size=2,
    vocab_size=100,
    d_model=32,
    n_layers=2,
    n_heads=4,
    data_parallel=True,
    model_parallel=False,
    pipeline_parallel=False,
    tensor_parallel=False,
)

def model_ctor():
    return TinyToyModel(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
    )

# For each GPU ID passed via --gpu-ids we spin up a NetworkWorker thread
# Each NetworkWorker:
#   - connects to orchestrator_ip:master_port+0
#   - trains forever (blocking)
worker = NetworkWorker(
    gpu_id=0,
    config=cfg,
    model_ctor=model_ctor,
    master_ip=cfg.master_ip,
    master_port=cfg.master_port,
)
worker.connect_and_train()
```

Key rules:
- `num_workers` on BOTH sides must match total GPUs participating across the cluster.
- Model definition MUST match 1:1 between orchestrator and worker (same layer shapes, same vocab size, etc.).
- You can run multiple `gpu_id`s on the same physical box; each becomes one logical worker.

---

## 🧱 Distributed Execution Model

### Control Plane
- Orchestrator opens listening sockets on `master_port + {0..7}`.
- Each GPU worker:
  - connects to `master_port + 0`
  - sends `CONTROL_HELLO`
  - gets `CONTROL_ACK`
  - keeps that socket open for commands.

### Data Parallel Step Flow
1. Orchestrator → all workers: `CONTROL_DATA_PARALLEL_STEP`
   (payload: `input_ids`, `labels`).
2. Worker:
   - forward
   - CE loss
   - backward()
   - clip grads
   - upload grads via `GRADIENTS_UPLOAD` (port `+2`)
   - upload metrics via `METRICS_STEP` (port `+1`)
   - ACK back on control socket
3. Orchestrator:
   - waits for grads from all workers
   - averages per-parameter
   - optimizer.step() on the master model

### Pipeline / Model Parallel Step Flow
Pipeline mode uses multi-phase control messages:
- `CONTROL_PIPELINE_PHASE1`
- `CONTROL_PIPELINE_PHASE2`
- `CONTROL_PIPELINE_BACKWARD`
The orchestrator walks the chain forward, then backward shard-by-shard using `BACKWARD_READY` to pass upstream gradients.

---

## ⚙ Parallelism Modes

Controlled by `DistributedConfig` flags:

```python
cfg.data_parallel = True
cfg.model_parallel = False
cfg.pipeline_parallel = False
cfg.tensor_parallel = False
```

- **Pure Data Parallel**
  - Every GPU gets full model
  - Orchestrator averages grads

- **Pipeline / Model Parallel**
  - Layers are split across GPUs
  - Orchestrator coordinates staged forward and chained backward

- **Tensor Parallel**
  - A single "layer" is split across a group of GPUs
  - Uses message types like `TENSOR_FORWARD_GATHER` and `TENSOR_BACKWARD_REDUCE`
  - Can be combined with pipeline/model parallel for hybrid setups

`cfg.get_parallelism_strategy()` returns:
`DATA_PARALLEL`, `MODEL_PARALLEL`, `PIPELINE_PARALLEL`, `TENSOR_PARALLEL`, or `HYBRID`.

---

## 🛠 Configuration

### Programmatic

```python
from guava import DistributedConfig

cfg = DistributedConfig(
    vocab_size=50257,
    d_model=768,
    n_heads=12,
    n_layers=12,
    d_ff=3072,
    max_seq_len=1024,
    dropout=0.1,

    batch_size=8,
    learning_rate=3e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    use_amp=False,

    master_ip="0.0.0.0",
    master_port=29500,
    num_workers=2,

    data_parallel=True,
    model_parallel=False,
    pipeline_parallel=False,
    tensor_parallel=False,
    micro_batches=4,
    tensor_parallel_size=2,

    activation_timeout=0.0,   # 0.0 == "wait forever"
    ack_timeout=0.0,          # 0.0 == "wait forever"
    max_resends=0,
    resend_probe_interval=5.0,

    checkpoint_dir="./model/checkpoints",
    save_interval=1000,
)
```

### From Environment

```powershell
setx MASTER_IP 192.168.0.177
setx MASTER_PORT 29500
setx NUM_WORKERS 2

setx DATA_PARALLEL 1
setx MODEL_PARALLEL 0
setx PIPELINE_PARALLEL 0
setx TENSOR_PARALLEL 0

setx MICRO_BATCHES 4
setx TENSOR_PARALLEL_SIZE 2

setx COMPACT_LOG 1
setx LOG_STEP_EVERY 100
setx LOG_ACT 0

setx ACT_TIMEOUT_SEC 0
setx ACK_TIMEOUT_SEC 0
setx RESENDS_MAX 0
setx RESEND_PROBE_SEC 5
```

Then in Python:

```python
cfg = DistributedConfig.from_env()
print(cfg.layers_per_gpu)  # auto-filled mapping per GPU
```

---

## 🌐 Networking / Sockets

All communication is plain TCP. We tune sockets with `socket_utils.optimize_socket_for_network()`:
- `TCP_NODELAY` → low latency for control / ACK messages.
- `SO_KEEPALIVE` → eventually detect dead peers.
- Large `SO_SNDBUF` / `SO_RCVBUF` → let us push fat tensors without stalling.
- Blocking reads with `[4-byte length][payload]` framing.
- Payload is zlib-compressed pickle.

### Buffer sizing

By default we request:
- ~16MB send/recv buffers on Windows and Linux
- ~8MB on macOS (macOS tends to clamp aggressively)

But you can push it higher on Windows/Linux.

Cranking buffers:
- 32MB, 64MB, even 128MB **per socket** can work on Windows/Linux for a *small number* of long-lived sockets.  
  That’s literally our pattern: a handful of fat pipes between `orchestrator <-> worker` for control, gradients, checkpoints. It’s not crazy.

Why you *might* want giant buffers:
- You're shipping big activation tensors / gradient blobs
- You’ve got high-bandwidth links (10GbE, 25GbE, InfiniBand, etc.)
- The RTT between orchestrator and worker is not trivial (not on the same PCIe switch / same host loopback).  
  Example: different physical machines across the rack on 25GbE.

In that situation, large socket buffers let the sender stay ahead and keep the pipe full without constantly stalling on `send()` waits.

Where huge buffers can bite you:
1. **Many workers per box**  
   If you spin up hundreds or thousands of workers all pointing at one orchestrator, those 64MB+ buffers multiply and can starve kernel memory.
2. **Local security inspection / firewall hooks**  
   On Windows, Defender / firewall inspection sometimes copies payloads before release. Giant buffers can actually introduce jitter and weird latency spikes under heavy load.

Rule of thumb:
- Single orchestrator ⇄ a few powerful GPU workers over real network fabric (10GbE+): go ahead and use 32MB–64MB+ buffers.
- One machine, localhost / loopback / same motherboard: bigger than ~16MB usually doesn’t help because latency is already microseconds.

Guava exposes this through `DistributedConfig.socket_buffer_mb`. That value (in MB) is converted to bytes and passed down to `optimize_socket_for_network()`, which requests that buffer size on every training socket. The OS may clamp it lower, and that’s fine.

### Port Layout (relative to `master_port`)

Guava reserves a small "port block" starting at `master_port`:

- `+0` Control plane  
  Long-lived connection. Handles:
  - worker registration (`CONTROL_HELLO`)
  - training step commands (`CONTROL_DATA_PARALLEL_STEP`, `CONTROL_PIPELINE_PHASE1`, etc.)
  - per-step ACK barriers (`CONTROL_ACK`)
  - backward coordination (`BACKWARD_READY`)
  - tensor-parallel collectives

- `+1` Metrics upload  
  Short-lived connection. Worker connects, sends `METRICS_STEP`, disconnects.

- `+2` Gradient upload  
  Short-lived connection. Worker connects, sends `GRADIENTS_UPLOAD` for that step, disconnects.

- `+7` Checkpoint upload  
  Short-lived connection. Worker uploads `CHECKPOINT_SHARD_UPLOAD` on shutdown or save.

- `+3,+4,+5,+6` Reserved  
  Activation relay, heartbeat, possible future explicit tensor-parallel rendezvous channels.

**Pattern:**
- Each worker holds exactly one long-lived control socket to `master_port + 0`.  
- Everything heavy (gradients, checkpoints) uses dedicated short-lived sockets on `+1`, `+2`, `+7`.  
  This lets us crank per-socket buffer sizes for those fat transfers without stalling the control channel.

---

## 🧮 Tensor Parallel

Tensor Parallel (TP) is optional and can stack with data/model/pipeline parallel.  
Goal: split big matmuls across multiple GPUs *inside* a single layer.

When TP is on:
- each GPU computes a slice of the layer
- results are gathered (`tensor_gather`)
- grads are reduced/averaged (`tensor_reduce_grad`)

`DistributedConfig.tensor_parallel_groups()` produces peer groups like:
```python
[[0,1], [2,3], ...]  # for tensor_parallel_size=2
```

Workers coordinate via messages like:
- `TENSOR_FORWARD_GATHER`
- `TENSOR_BACKWARD_REDUCE`
- `TENSOR_SYNC_BARRIER`

---

## 🐛 Troubleshooting

>  **A Dev's Note on Errors**
> 
> Honestly, GPU error codes can get a little funky in my opinion.  
> In your development, consider that a lot of the time during training (and sometimes inference),  
> a simple restart is all that’s needed and you’ll progress until the next error.
>   
### Worker never registers
- Confirm the worker box can reach `master_ip:master_port+0`.
- Check firewall / Windows Defender inbound rules.
- Make sure `--world-size` on workers equals `--gpus` on orchestrator.

### Stalled waiting on gradients
- The orchestrator will not step until *all* expected workers upload grads for that step.
- If one worker crashed mid-step, training pauses.
- Look at that worker's console for OOM or device lost.

### CUDA OOM
Try:
```python
cfg.batch_size = 4
cfg.use_amp = True
cfg.max_grad_norm = 1.0
```

### Connection resets under load
- Prefer same-rack networking / low latency between orchestrator and workers.
- 10GbE+ helps.
- Watch MTU / jumbo frame config for huge tensor payloads.

---

## 🪪 License

Guava is offered under a **dual license**:

- **Community Edition (Apache 2.0)**  
  You can use, modify, and redistribute for personal, research, or non-commercial work.

- **Commercial Edition (Proprietary)**  
  Required if:
  - You integrate Guava into a paid product or service,
  - You offer managed training / managed inference built on Guava,
  - You sell Guava-powered compute access.

For commercial licensing, contact:  
📧 azanipeterking@gmail.com

---

## 📮 Support

- **Issues:** GitHub Issues (open an issue with logs and config details)
- **Docs / Examples:** more launch recipes coming
- **Discord / Community:** coming

**Made with ❤️ for the ML community**
