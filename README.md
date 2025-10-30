# Guava

**Distributed Neural Network Training Over Network**

Guava is a modular, socket-driven framework for orchestrating distributed PyTorch training across multiple GPUs and machines. It gives you:

- an **Orchestrator** (the "brain") that lives on CPU and coordinates training  
- one or more **NetworkWorker** processes (one per GPU) that actually run the model/shard  
- message-level parallelism primitives (data / model / pipeline / tensor) without requiring `torch.distributed`

This README describes the current state of the pip module you just saw:

- `config.py`
- `base_worker.py`
- `orchestrator.py`
- `network_worker.py`
- `protocol.py`
- `socket_utils.py`
- `__init__.py`
- `orchestrator_train.py` (CLI entrypoint for the orchestrator)
- `guava_worker.py` (CLI entrypoint for GPU workers)

---

## Table of Contents

- [🚀 Features](#-features)
- [📦 Installation](#-installation)
- [🧠 Core Modules](#-core-modules)
- [⚡ Quick Start (Real Commands You Run)](#-quick-start-real-commands-you-run)
- [🧱 Distributed Execution Model](#-distributed-execution-model)
- [⚙ Parallelism Modes](#-parallelism-modes)
- [🛠 Configuration](#-configuration)
- [🌐 Networking / Sockets](#-networking--sockets)
- [🧮 Tensor Parallel](#-tensor-parallel)
- [🐛 Troubleshooting](#-troubleshooting)
- [🪪 License](#-license)
- [📮 Support](#-support)

---

## 🚀 Features

### Multiple Parallelism Strategies

- **Data Parallelism**  
  Each GPU holds a full copy of the model and trains on its own batch. Gradients are averaged.
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

```bash
# Recommended (your users):
pip install guava

# Or from TestPyPI while developing:
pip install --extra-index-url https://test.pypi.org/simple/ guava
```

**Requirements**

- Python ≥ 3.8  
- PyTorch ≥ 2.0  
- At least one CUDA-capable GPU for actual training (workers)  
- CPU-only is fine for the orchestrator

After install:

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

Key helpers:

- `adapt_to_gpus(num_gpus)`  
  Mutates the config to make sane choices based on how many GPUs are actually in play.

- `get_layers_per_gpu()`  
  Tells each worker how many transformer layers it "owns".  
  - Pure data parallel or pure tensor parallel (no model/pipeline sharding): every GPU gets the full stack.  
  - Otherwise: layers are split across workers like `[6,6]`, `[5,4,4]`, etc.

- `tensor_parallel_groups()`  
  Returns groups like `[[0,1],[2,3],...]` if `tensor_parallel_size=2`.

- `get_parallelism_strategy()`  
  Returns `DATA_PARALLEL`, `MODEL_PARALLEL`, `PIPELINE_PARALLEL`, `TENSOR_PARALLEL`, or `HYBRID`.

- `from_env()` / `to_dict()` / `from_dict()`  
  Lets you bootstrap configs from environment variables (e.g. in Docker/SSH launch scripts).

### `BaseWorker`, `DataParallelWorker`, `ModelShardWorker` (`base_worker.py`)

Abstractions for the actual compute running on a GPU.

All workers:

- live on a **specific GPU**
- hold either:
  - the full model (data parallel), or
  - only a slice of layers (model/pipeline parallel)
- expose:
  - `forward(...)`
  - `backward(...)`
  - `update_weights()` (optimizer step)
  - gradient capture / sync helpers
  - cleanup logic for CUDA memory

`DataParallelWorker`:

- full model replica
- runs forward + CE loss + backward locally
- returns gradients for every parameter so the orchestrator can average them
- can also receive a gradient-from-master mode if you want external loss computation

`ModelShardWorker`:

- only owns `[layer_start:layer_end)` of the transformer
- caches activations so it can:
  - receive upstream grads from the *next* shard
  - run backward on its local slice
  - produce the upstream grad for the *previous* shard

Both inherit `TensorParallelMixin`, which exposes:

- `tensor_split(...)`  
  Return just the local slice of a tensor along some dimension.
- `tensor_gather(...)`  
  All-gather partial outputs across tensor-parallel peers.
- `tensor_reduce_grad(...)`  
  All-reduce / average gradients across tensor-parallel peers.

By default these are safe no-ops unless tensor-parallel is actually enabled.

### `Orchestrator` (`orchestrator.py`)

The "brain." Runs on CPU.

Responsibilities:

- Listen on `master_port + {0..7}` for:
  - control connections
  - gradient uploads
  - metric uploads
  - checkpoint uploads
- Maintain a map of registered workers (`gpu_id -> socket, layer range, hostname`)
- Drive training loops

Two training modes are built-in:

1. **Data Parallel Loop**  
   - Broadcast `CONTROL_DATA_PARALLEL_STEP` with input + labels to all workers  
   - Each worker:
     - runs forward
     - computes CE loss
     - backward()
     - uploads gradients via `GRADIENTS_UPLOAD`
   - Orchestrator:
     - waits for all gradients
     - averages them by parameter name
     - applies them to the master model
     - steps the master optimizer
   - Metrics come in via `METRICS_STEP`

2. **Pipeline / Model Parallel Loop**  
   True multi-stage backward:  
   - Phase 1 (`CONTROL_PIPELINE_PHASE1`): everyone runs forward for their shard; final shard caches logits  
   - Phase 2 (`CONTROL_PIPELINE_PHASE2` to last shard only): last shard gets labels, computes CE, backward(), uploads its grads, and sends `BACKWARD_READY` with upstream grad  
   - Backward chain (`CONTROL_PIPELINE_BACKWARD`): orchestrator walks upstream shard by shard, feeding upstream grad so each shard can run backward and upload its own grads  
   - After all shards upload, orchestrator aggregates all grads and optimizer.step() happens once

Also:

- Validation runs on the orchestrator’s master copy (`_run_validation`)
- Checkpoints can be saved with `save_checkpoint(path)`

### `NetworkWorker` (`network_worker.py`)

A runtime wrapper that lives on a GPU box.

It does:

1. Build / slice the model for **this** GPU  
   - constructs the full model using `model_ctor()`  
   - if we're data parallel only: keep full replica  
   - else: extract `[layer_start:layer_end)` into a `ShardModule`

2. Connect to the orchestrator’s control socket  
   - send `CONTROL_HELLO` (gpu_id, layer range, hostname, tp group size)  
   - wait for `CONTROL_ACK`

3. Loop forever:  
   - receive control messages (`CONTROL_DATA_PARALLEL_STEP`, `CONTROL_PIPELINE_PHASE1`, etc.)  
   - run local forward/backward  
   - upload grads (`GRADIENTS_UPLOAD`)  
   - upload metrics (`METRICS_STEP`)  
   - ACK the command so orchestrator knows this shard is in sync

4. On shutdown (`CONTROL_STOP`):  
   - upload checkpoint shard (`CHECKPOINT_SHARD_UPLOAD`)  
   - exit clean

`NetworkWorker` also implements local tensor-parallel calls:

- `tensor_gather(...)`
- `tensor_reduce_grad(...)`

These talk to peers using message types like `TENSOR_FORWARD_GATHER` and `TENSOR_BACKWARD_REDUCE`.

### `protocol.py`

Defines:

- `MessageType` (enum of everything the orchestrator/workers can say to each other)
- `Message` (dataclass carried over the wire)
- `MessageProtocol` (length-prefixed, pickle+zlib framing and helpers)

Important message types:

- Lifecycle / control  
  `CONTROL_HELLO`, `CONTROL_ACK`, `CONTROL_STOP`, `CONTROL_HEARTBEAT`
- Training steps  
  `CONTROL_DATA_PARALLEL_STEP`, `CONTROL_PIPELINE_PHASE1`, `CONTROL_PIPELINE_PHASE2`, `CONTROL_PIPELINE_BACKWARD`
- Metrics & gradients  
  `METRICS_STEP`, `GRADIENTS_UPLOAD`
- Backward coordination  
  `BACKWARD_READY`
- Checkpoint upload  
  `CHECKPOINT_SHARD_UPLOAD`
- Tensor parallel collectives  
  `TENSOR_FORWARD_GATHER`, `TENSOR_BACKWARD_REDUCE`, `TENSOR_SYNC_BARRIER`

`MessageProtocol` also has safe tensor (de)serialization helpers for activation relay.

### `socket_utils.py`

All the low-level socket tuning:

- `optimize_socket_for_network(sock, buffer_size)`  
  - `TCP_NODELAY`  
  - `SO_KEEPALIVE`  
  - enlarged send/recv buffers (16MB on Linux/Win, 8MB on macOS)  
  - blocking mode by default
- helper functions to:
  - connect with retry
  - check port availability
  - pick an unused port
  - send/recv blobs with `[len][payload]` framing (`send_with_size`, `recv_with_size`)

---

## ⚡ Quick Start (Real Commands You Run)

### 0. Make a virtualenv (example)

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
# .\venv\Scripts\activate    # Windows PowerShell
pip install --upgrade pip
pip install --extra-index-url https://test.pypi.org/simple/ guava
```

---

### 1. Launch the orchestrator on the HEAD / CPU box

The orchestrator coordinates training. You run `orchestrator_train.py` like this:

```bash
python orchestrator_train.py     --master-ip 0.0.0.0     --master-port 29500     --gpus 2     --epochs 1     --train-batches 100     --val-batches 20     --val-interval 25     --seq-len 16     --batch-size 2     --vocab-size 100     --d-model 32     --n-layers 2     --n-heads 4
```

What this does:

- Binds the orchestrator to `--master-ip` / `--master-port`
- Expects `--gpus 2` total workers to register
- Builds a tiny demo model (`TinyToyModel`) internally
- Generates random token data
- Waits for workers to connect (they send `CONTROL_HELLO`)
- Starts driving training/validation

You'll see logs like:

```text
GUAVA ORCHESTRATOR START
master_ip    : 0.0.0.0
master_port  : 29500
num_workers  : 2
...
waiting for workers to register...
```

---

### 2. Launch the workers on the GPU machine(s)

Each GPU machine runs `guava_worker.py`.  
You can launch multiple GPUs from ONE process using `--gpu-ids`.

Example: one box with 2 GPUs (GPU 0 and GPU 1) connecting to the orchestrator at `192.168.0.177`:

```bash
python guava_worker.py     --gpu-ids 0,1     --master-ip 192.168.0.177     --master-port 29500     --world-size 2     --batch-size 2     --vocab-size 100     --d-model 32     --n-layers 2     --n-heads 4
```

What this does:

- Spawns one internal worker thread per GPU ID you pass in (`0` and `1` here)
- Builds the SAME tiny demo model (`TinyToyModel`) as the orchestrator
- Connects each worker thread to the orchestrator’s control socket
- Starts responding to `CONTROL_DATA_PARALLEL_STEP`, sending gradients, sending metrics, etc.

When it's happy you'll see prints like:

```text
GUAVA MULTI-GPU WORKER LAUNCHER
local gpu_ids    : [0, 1]
orchestrator_ip  : 192.168.0.177
orchestrator_port: 29500
world_size(total): 2
GPU 0: NVIDIA GeForce RTX ...
GPU 1: NVIDIA GeForce RTX ...
connecting to orchestrator and starting training loop...
```

---

### 3. Watch training

After the orchestrator sees both workers register, it'll drive the training loop:
- send batches (toy random token IDs)
- receive gradients
- aggregate / optimizer.step()
- show step metrics and (optionally) periodic validation

Eventually the orchestrator writes a checkpoint to `./checkpoints`.

---

### 4. Swap in your OWN model later

Right now both `orchestrator_train.py` and `guava_worker.py` define the same `TinyToyModel` inline so they agree on shapes.

Later you can:
- replace that `TinyToyModel` in both scripts with your model class, **OR**
- import the same model class from a shared module in your codebase.

Either way: orchestrator and workers MUST agree on:
- vocab size / embedding dim
- depth / heads / tensor shapes
- loss function assumption

---

## 🧱 Distributed Execution Model

### Control Plane

- Orchestrator opens listening sockets on `master_port + {0..7}`.
- Each GPU worker:
  - connects to `master_port + 0`
  - sends `CONTROL_HELLO`
  - gets `CONTROL_ACK`
  - stays connected there forever for commands

### Data Parallel Step Flow

1. Orchestrator → all workers: `CONTROL_DATA_PARALLEL_STEP`  
   Payload includes `input_ids` and `labels`.

2. Worker:
   - runs forward
   - computes CE loss
   - backward()
   - clips gradients
   - (optional) tensor-parallel gradient reduce
   - uploads gradients to `master_port+2` via `GRADIENTS_UPLOAD`
   - uploads metrics to `master_port+1` via `METRICS_STEP`
   - ACKs the command on the control socket

3. Orchestrator:
   - waits until it has gradients from all workers
   - averages them param-by-param
   - loads averaged grads into the master model
   - optimizer.step()

### Pipeline / Model Parallel Step Flow

Pipeline mode uses 3 control message phases:

**PHASE 1** (`CONTROL_PIPELINE_PHASE1`)  
- Stage0 gets `input_ids` and runs forward on its shard.  
- Each shard runs its slice of layers.  
- Final shard caches logits.

**PHASE 2** (`CONTROL_PIPELINE_PHASE2`)  
- Only last shard gets labels.  
- Last shard:
  - computes CE loss
  - backward()
  - uploads its grads (→ `GRADIENTS_UPLOAD`)
  - replies `BACKWARD_READY` to orchestrator with an `upstream_grad` to feed the previous shard.

**PIPELINE_BACKWARD** (`CONTROL_PIPELINE_BACKWARD`)  
- Orchestrator walks upstream:  
  for shard `N-1`, `N-2`, ... `0`:
  - send them the `upstream_grad`
  - they backward(), upload grads, generate next `upstream_grad`
  - they send `BACKWARD_READY` back

After all shards upload:
- Orchestrator aggregates all grads and steps the optimizer once.

This lets Guava do *true* multi-stage backward across shards.

---

## ⚙ Parallelism Modes

`DistributedConfig` drives which mode(s) you're in:

```python
cfg.data_parallel = True
cfg.model_parallel = False
cfg.pipeline_parallel = False
cfg.tensor_parallel = False
```

- **Pure Data Parallel**  
  - Every GPU gets full model  
  - `layers_per_gpu = [n_layers]*num_workers`  
  - All workers compute grads on the same step, orchestrator averages

- **Pipeline / Model Parallel**  
  - `model_parallel=True` and/or `pipeline_parallel=True`  
  - Layers are split across GPUs according to `cfg.layers_per_gpu`  
  - Orchestrator coordinates forward (PHASE1), loss/backward init (PHASE2), and upstream gradient chaining (PIPELINE_BACKWARD)

- **Tensor Parallel**  
  - `tensor_parallel=True`  
  - `tensor_parallel_size=2` (for example)  
  - Each *layer* is internally sharded across a TP group (see below)  
  - You can run TP alone (each GPU still "owns" full depth but splits heavy matmuls) **or** combine with pipeline/model sharding (hybrid)

`cfg.get_parallelism_strategy()` returns a `ParallelismStrategy` enum summarizing active modes:  
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
    tensor_parallel_size=2,   # used if tensor_parallel=True

    activation_timeout=0.0,   # 0.0 == "wait forever"
    ack_timeout=0.0,          # 0.0 == "wait forever"
    max_resends=0,
    resend_probe_interval=5.0,

    checkpoint_dir="./model/checkpoints",
    save_interval=1000,
)
```

### From Environment

`DistributedConfig.from_env()` will pull:

```bash
export MASTER_IP=192.168.1.50
export MASTER_PORT=29500
export NUM_WORKERS=2

export DATA_PARALLEL=1
export MODEL_PARALLEL=0
export PIPELINE_PARALLEL=0
export TENSOR_PARALLEL=0          # or ENABLE_TENSOR_PARALLEL=1

export MICRO_BATCHES=4
export TENSOR_PARALLEL_SIZE=2

export COMPACT_LOG=1
export LOG_STEP_EVERY=100
export LOG_ACT=0

export ALLOW_ACT_REUSE=0
export ACT_CACHE_STEPS=256

export ACT_TIMEOUT_SEC=0          # 0 => infinite wait
export ACK_TIMEOUT_SEC=0          # 0 => infinite wait
export RESENDS_MAX=0
export RESEND_PROBE_SEC=5
```

After loading env:

```python
cfg = DistributedConfig.from_env()
cfg.layers_per_gpu  # auto-filled mapping per GPU
```

---

## 🌐 Networking / Sockets

All communication is plain TCP with tuned settings from `socket_utils.optimize_socket_for_network()`:

- `TCP_NODELAY` → low latency for step commands / ACKs  
- `SO_KEEPALIVE` → detect dead peers eventually  
- Large send/recv buffers:  
  - Linux/Windows: 16 MB  
  - macOS: 8 MB (macOS clamps socket buffers lower)  
- Blocking sockets by default, no busy loops  
- zlib-compressed pickle messages over a simple:

```text
[4-byte big-endian length][that many bytes of payload]
```

### Port Layout (relative to `master_port`)

- `+0` Control plane  
  Registration (`CONTROL_HELLO`), ACK barriers, BACKWARD_READY messages.
- `+1` Metrics upload  
  Workers connect, send one `METRICS_STEP`, disconnect.
- `+2` Gradient upload  
  Workers connect, send one `GRADIENTS_UPLOAD`, disconnect.
- `+7` Checkpoint upload  
  Workers connect, send `CHECKPOINT_SHARD_UPLOAD`, disconnect.
- `+3,+4,+5,+6` Reserved  
  (Activation relay, heartbeat, tensor-parallel collectives, etc.)

Workers maintain a **long-lived** control socket to `+0`, and use **short-lived** connections for metrics, gradients, and checkpoints.

---

## 🧮 Tensor Parallel

Tensor Parallel (TP) is optional and can be layered on top of data/model/pipeline parallel. It’s inspired by Megatron-LM style intra-layer splitting.

### Goal

Let multiple GPUs collaborate on a *single* wide layer:

- Split big weight matrices across GPUs.
- Each GPU does its slice of the matmul.
- Gather partial outputs to form the full activation.
- During backward, all-reduce gradients so each GPU sees the averaged result.

### How Guava Exposes TP

1. **Config**

```python
cfg.tensor_parallel = True
cfg.tensor_parallel_size = 2   # e.g. groups of 2 GPUs
```

2. **Groups**

```python
cfg.tensor_parallel_groups()
# → [[0,1],[2,3], ...] for tp_size=2
```

3. **On the worker side (`NetworkWorker`)**

- `tensor_split(tensor, dim=-1)`  
  Gives this GPU's slice of a full tensor.
- `tensor_gather(local_tensor, step)`  
  All-gather partial outputs across peers → full tensor on each peer.
- `tensor_reduce_grad(local_grad, step)`  
  All-reduce and average grads across peers.

These helpers wrap socket round-trips with message types:

- `TENSOR_FORWARD_GATHER`
- `TENSOR_BACKWARD_REDUCE`
- `TENSOR_SYNC_BARRIER`

> If `tensor_parallel` is **on** with no model/pipeline sharding:  
> - every GPU still "thinks" it has the full layer stack (`layers_per_gpu = [n_layers]*num_workers`)  
> - but *inside each layer* you're actually splitting the compute.

---

## 🐛 Troubleshooting

> ⚠️ **Note on errors**  
> GPU error codes can get... funky. In practice during training (and sometimes inference), a simple restart of the worker or restarting the training loop will clear a transient CUDA or memory hiccup. You don't always need to deep-dive a mysterious "launch failure" the first time you see it. Hunt the weird ones only when they keep repeating.

### Worker never registers

- Make sure worker can reach `master_ip:master_port+0` over the network.
- Confirm `NUM_WORKERS` and `gpu_id` assignments are correct.
- Check firewall, security group, or local Windows Defender rules.

### Blocked waiting on gradients

- Orchestrator waits until it has a gradient upload from *every expected worker/shard* for that global step.
- If a worker crashes mid-step, orchestrator won't advance.
- Check worker logs for CUDA OOM or connection reset.

### CUDA OOM

Try:

```python
cfg.batch_size = 4
cfg.use_amp = True
cfg.max_grad_norm = 1.0
```

### Connection resets under load

- Use a faster link (10GbE+ if possible).
- Keep orchestrator physically close (same rack / VLAN).
- Verify jumbo frames / MTU settings if you're pushing giant tensors.

---

## 🪪 License

Guava is offered under a **dual license**:

- **Community Edition (Apache 2.0)**  
  You can use, modify, and distribute for personal, research, or non-commercial work.

- **Commercial Edition (Proprietary)**  
  Required if:
  - You integrate Guava into a paid product or service
  - You offer managed training / managed inference built on Guava
  - You sell Guava-powered compute access

For commercial licensing, contact:  
📧 azanipeterking@gmail.com

---

## 📮 Support

- **Issues:** GitHub Issues (open an issue with logs and config details)  
- **Docs / Examples:** more examples and launch scripts coming  
- **Discord / Community:** coming

**Made with ❤️ for the ML community**
