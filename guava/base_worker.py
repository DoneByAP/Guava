"""
Base worker interface for distributed training.
Adds optional Tensor Parallel helpers (no-ops unless a tp_adapter is set).
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
import logging

from .training_components import build_loss_handler, build_optimizer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tensor Parallel adapter + mixin
# -----------------------------------------------------------------------------

class TensorParallelAdapter(Protocol):
    """
    Adapter interface that performs the actual collectives over the network.
    Implementations should provide:
      - gather(local_tensor, step) -> full_tensor
      - reduce(local_grad, step) -> averaged_grad
    In your codebase, NetworkWorker owns the control socket and can pass an
    object implementing this interface into BaseWorker.
    """
    def gather(self, local_tensor: torch.Tensor, step: int) -> torch.Tensor: ...
    def reduce(self, local_grad: torch.Tensor, step: int) -> torch.Tensor: ...


class _NoOpTPAdapter:
    """Default TP adapter that simply no-ops."""
    def gather(self, local_tensor: torch.Tensor, step: int) -> torch.Tensor:
        return local_tensor
    def reduce(self, local_grad: torch.Tensor, step: int) -> torch.Tensor:
        return local_grad


class TensorParallelMixin:
    """
    Shared tensor-parallel utilities for splitting/gathering/reducing tensors.
    These are safe no-ops unless a real tp_adapter is provided AND
    config.enable_tensor_parallel is True.
    """

    def _tp_enabled(self) -> bool:
        return bool(
            getattr(self.config, "enable_tensor_parallel", False)
            and getattr(self.config, "tensor_parallel_size", 1) > 1
        )

    def tensor_split(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Return local shard of a tensor along `dim`. This is a convenience helper
        you can use in custom layers or glue code if you pre-construct full tensors.
        """
        if not self._tp_enabled():
            return tensor
        tp_size = int(getattr(self.config, "tensor_parallel_size", 1) or 1)
        rank = int(getattr(self, "tensor_rank", 0))
        chunks = torch.chunk(tensor, tp_size, dim=dim)
        return chunks[rank].contiguous()

    def tensor_gather(self, local_tensor: torch.Tensor, step: int) -> torch.Tensor:
        """
        All-gather partial outputs across tensor peers to build the full activation.
        Delegates to tp_adapter if TP is enabled, else returns the input.
        """
        if not self._tp_enabled():
            return local_tensor
        return self.tp_adapter.gather(local_tensor, step)

    def tensor_reduce_grad(self, local_grad: Optional[torch.Tensor], step: int) -> Optional[torch.Tensor]:
        """
        All-reduce (average) gradients across tensor peers.
        Delegates to tp_adapter if TP is enabled, else returns the input.
        """
        if local_grad is None or not self._tp_enabled():
            return local_grad
        return self.tp_adapter.reduce(local_grad, step)


# -----------------------------------------------------------------------------
# Base + concrete workers
# -----------------------------------------------------------------------------

class BaseWorker(TensorParallelMixin, ABC):
    """
    Abstract base class for distributed training workers.
    Each worker sits on a specific GPU, holds either a full model (data parallel)
    or a slice of the model (model parallel), and talks to the orchestrator.
    """

    def __init__(self, gpu_id: int, config: Any, tp_adapter: Optional[TensorParallelAdapter] = None):
        """
        Args:
            gpu_id: which local CUDA device this worker should use
            config: DistributedConfig (learning rate, clip norm, etc.)
            tp_adapter: Optional adapter implementing tensor-parallel collectives.
                        If None, TP helpers become safe no-ops.
        """
        self.gpu_id = gpu_id
        self.config = config
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.is_training: bool = False
        self.loss_handler = build_loss_handler(self.config, device=self.device)

        # TP plumbing
        self.tp_adapter: TensorParallelAdapter = tp_adapter or _NoOpTPAdapter()
        tp_size = int(getattr(self.config, "tensor_parallel_size", 1) or 1)
        self.tensor_rank: int = (gpu_id % tp_size) if tp_size > 0 else 0

        # Cached tensors from last forward() so backward() can use them.
        # For model-parallel shard: we cache activations we output.
        # For data-parallel replica: we cache logits we produced.
        self._last_activation: torch.Tensor = None
        self._last_output: torch.Tensor = None

        logger.info(
            f"Worker {gpu_id}: Initialized on {self.device} "
            f"(TP enabled={getattr(self.config, 'enable_tensor_parallel', False)}, "
            f"tp_size={getattr(self.config, 'tensor_parallel_size', 1)}, "
            f"tp_rank={self.tensor_rank})"
        )

    @abstractmethod
    def register_model(self, model: nn.Module) -> None:
        """
        Give this worker its model or model shard, move to device,
        and create the optimizer for JUST those params.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Run forward pass on this worker's model (or shard).
        Must also cache the relevant tensor so backward() can run later.
        """
        ...

    @abstractmethod
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward through this worker's model (or shard).

        For data-parallel workers:
            grad_output is usually dLoss/dLogits and we call .backward() on cached logits.

        For model-shard workers:
            grad_output is the upstream grad from the *next* shard,
            we backprop into our cached activation and return grad_input
            so the previous shard can keep going.
        """
        ...

    def update_weights(self) -> None:
        """One optimizer step + zero_grad()."""
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def get_model_state(self) -> Dict:
        """Return state_dict() for checkpointing / sync."""
        return self.model.state_dict() if self.model is not None else {}

    def load_model_state(self, state_dict: Dict) -> None:
        """Load weights from orchestrator."""
        if self.model is not None:
            self.model.load_state_dict(state_dict)

    def set_training_mode(self, training: bool = True) -> None:
        """
        Flip train/eval mode.
        Orchestrator can call this at epoch boundaries or eval steps.
        """
        self.is_training = training
        if self.model is not None:
            self.model.train(training)

    def cleanup(self) -> None:
        """Free model/optimizer + empty CUDA cache."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Worker {self.gpu_id}: Cleaned up")
        self.loss_handler = None


class ModelShardWorker(BaseWorker):
    """
    Model-parallel worker: holds a slice of the overall network
    (e.g. transformer layers 6..12). It:
    - Receives activations from the previous shard
    - Runs forward on its local layers
    - Sends activations to the next shard
    - Later receives dLoss/dActivation, runs backward on its slice,
      and returns dLoss/dPrevActivation upstream
    """

    def __init__(self, gpu_id: int, config: Any, layer_start: int, layer_end: int, tp_adapter: Optional[TensorParallelAdapter] = None):
        """
        Args:
            gpu_id: GPU index on this machine
            config: DistributedConfig
            layer_start: inclusive global layer index this shard starts at
            layer_end:   exclusive global layer index this shard ends at
            tp_adapter: optional TP adapter for gather/reduce collectives
        """
        super().__init__(gpu_id, config, tp_adapter=tp_adapter)
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.num_layers = layer_end - layer_start

        logger.info(f"Worker {gpu_id}: Handling layers [{layer_start}, {layer_end})")

    def register_model(self, model: nn.Module) -> None:
        """
        `model` is already the sliced nn.Module for JUST this shard.
        We move it to our device and make an optimizer on those params.
        """
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.config, self.model.parameters())

        logger.info(
            f"Worker {self.gpu_id}: Registered model shard with {self.num_layers} layers"
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward through this shard and cache the activation so we
        can later run backward() when orchestrator sends gradients.

        NOTE: If you implement true intra-layer tensor splitting INSIDE layers,
        call self.tensor_split/tensor_gather around those layer ops.
        """
        x = x.to(self.device, non_blocking=True)

        with torch.set_grad_enabled(self.is_training):
            out = self.model(x, *args, **kwargs)

        # Cache activation for pipeline backward.
        self._last_activation = out
        if self.is_training and isinstance(out, torch.Tensor):
            self._last_activation.retain_grad()

        return out

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """
        grad_output: dLoss/dOut from the *next* shard.
        We backprop into our cached activation to get dLoss/dIn.
        Then we return that upstream to the previous shard.
        """
        if self._last_activation is None:
            logger.error("ModelShardWorker.backward() called with no cached activation")
            return None

        grad_output = grad_output.to(self.device, non_blocking=True)

        # Backprop from cached activation
        self._last_activation.backward(grad_output)

        # If doing tensor-parallel inside this shard's layers, you may
        # optionally reduce grads here per-parameter (NetworkWorker currently
        # handles this after loss.backward()):

        # for p in self.model.parameters():
        #     if p.grad is not None:
        #         p.grad = self.tensor_reduce_grad(p.grad, step=<provide step>)

        grad_input = self._last_activation.grad
        return grad_input


class DataParallelWorker(BaseWorker):
    """
    Data-parallel worker: holds the entire model replica.
    - Gets its own batch (inputs, labels)
    - Computes forward, loss, backward
    - Gives gradients to orchestrator for averaging across replicas
    """

    def __init__(self, gpu_id: int, config: Any, tp_adapter: Optional[TensorParallelAdapter] = None):
        super().__init__(gpu_id, config, tp_adapter=tp_adapter)

    def register_model(self, model: nn.Module) -> None:
        """
        model: full model replica.
        """
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.config, self.model.parameters())

        logger.info(f"Worker {self.gpu_id}: Registered full model for data parallelism")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Standard forward.
        We also cache the output logits so:
        - compute_loss_and_backward() can reuse them
        - backward() can apply external gradients if orchestrator sends grad_output

        NOTE: If you add true intra-layer TP here, call tensor_split/tensor_gather
        inside your module implementation. At this level we expose helpers only.
        """
        x = x.to(self.device, non_blocking=True)

        with torch.set_grad_enabled(self.is_training):
            out = self.model(x, *args, **kwargs)

        self._last_output = out
        if self.is_training and isinstance(out, torch.Tensor):
            self._last_output.retain_grad()

        return out

    def backward(self, grad_output: torch.Tensor) -> None:
        """
        If orchestrator is doing the loss somewhere else and just gives us
        dLoss/dLogits, we can still do local backward on cached _last_output.
        """
        if self._last_output is None:
            logger.error("DataParallelWorker.backward() called with no cached output")
            return

        grad_output = grad_output.to(self.device, non_blocking=True)

        # Backprop through cached logits.
        self._last_output.backward(grad_output)

        # Clip after backward, before step.
        if getattr(self.config, "max_grad_norm", 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

    def compute_loss(self, logits: torch.Tensor, batch: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
        """Compute loss for the provided logits using the configured handler."""
        if self.loss_handler is None:
            return None
        batch = batch or {}
        return self.loss_handler.compute(logits, batch, device=self.device)

    def compute_loss_and_backward(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """
        Full local training step:
        - compute configurable loss
        - backward()
        - clip gradients
        Returns scalar loss for logging.

        If you want TP grad-all-reduce per-parameter here, do it in the caller
        (e.g., NetworkWorker) by iterating params and calling tensor_reduce_grad.
        """
        effective_batch: Dict[str, Any] = batch.copy() if batch else {}
        if labels is not None:
            effective_batch.setdefault(getattr(self.loss_handler, "label_key", "labels"), labels)

        loss = self.compute_loss(logits, effective_batch)
        if loss is None:
            return None

        if self.is_training:
            loss.backward()

            if getattr(self.config, "max_grad_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

        return float(loss.detach().item())

    def get_gradients(self) -> List[torch.Tensor]:
        """
        Collect gradients from each parameter so orchestrator
        can average them across all replicas and send the averaged
        version back.
        """
        grads: List[torch.Tensor] = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().clone())
            else:
                grads.append(torch.zeros_like(param, device=self.device))
        return grads

    def average_gradients(self, all_gradients: List[List[torch.Tensor]]) -> None:
        """
        all_gradients: list indexed by worker, each entry is that worker's
        param.grad list. We'll take the mean across workers and copy it into
        OUR .grad so that our optimizer.step() applies the synced update.
        """
        num_workers = len(all_gradients)
        if num_workers == 0:
            logger.warning("average_gradients called with no gradients")
            return

        # We assume param ordering is consistent across workers.
        for i, param in enumerate(self.model.parameters()):
            if param.grad is None:
                continue  # nothing to average into

            stacked = torch.stack(
                [worker_grads[i].to(self.device) for worker_grads in all_gradients],
                dim=0
            )
            avg_grad = stacked.mean(dim=0)
            param.grad.copy_(avg_grad)