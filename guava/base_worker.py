"""
Base worker interface for distributed training.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    """
    Abstract base class for distributed training workers.
    Each worker sits on a specific GPU, holds either a full model (data parallel)
    or a slice of the model (model parallel), and talks to the orchestrator.
    """

    def __init__(self, gpu_id: int, config: Any):
        """
        Args:
            gpu_id: which local CUDA device this worker should use
            config: DistributedConfig (learning rate, clip norm, etc.)
        """
        self.gpu_id = gpu_id
        self.config = config
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        self.model: nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.is_training: bool = False

        # Cached tensors from last forward() so backward() can use them.
        # For model-parallel shard: we cache activations we output.
        # For data-parallel replica: we cache logits we produced.
        self._last_activation: torch.Tensor = None
        self._last_output: torch.Tensor = None

        logger.info(f"Worker {gpu_id}: Initialized on {self.device}")

    @abstractmethod
    def register_model(self, model: nn.Module) -> None:
        """
        Give this worker its model or model shard, move to device,
        and create the optimizer for JUST those params.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Run forward pass on this worker's model (or shard).
        Must also cache the relevant tensor so backward() can run later.
        """
        pass

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
        pass

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

    def __init__(self, gpu_id: int, config: Any, layer_start: int, layer_end: int):
        """
        Args:
            gpu_id: GPU index on this machine
            config: DistributedConfig
            layer_start: inclusive global layer index this shard starts at
            layer_end:   exclusive global layer index this shard ends at
        """
        super().__init__(gpu_id, config)
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

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        logger.info(
            f"Worker {self.gpu_id}: Registered model shard with {self.num_layers} layers"
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward through this shard and cache the activation so we
        can later run backward() when orchestrator sends gradients.
        """
        x = x.to(self.device, non_blocking=True)

        with torch.set_grad_enabled(self.is_training):
            out = self.model(x, *args, **kwargs)

        # Cache activation for pipeline backward.
        # We must keep grad so that when the NEXT shard gives us dLoss/dOut,
        # we can call backward() here and then read grad_input.
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

        # Run backward from cached activation
        self._last_activation.backward(grad_output)

        # Now the grad of the input to this shard lives on self._last_activation.grad
        # but we need the gradient for the *input that originally came into this shard*.
        # After .backward(), PyTorch populated .grad on _last_activation's source.
        grad_input = self._last_activation.grad

        return grad_input


class DataParallelWorker(BaseWorker):
    """
    Data-parallel worker: holds the entire model replica.
    - Gets its own batch (inputs, labels)
    - Computes forward, loss, backward
    - Gives gradients to orchestrator for averaging across replicas
    """

    def __init__(self, gpu_id: int, config: Any):
        super().__init__(gpu_id, config)

    def register_model(self, model: nn.Module) -> None:
        """
        model: full model replica.
        """
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        logger.info(f"Worker {self.gpu_id}: Registered full model for data parallelism")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Standard forward.
        We also cache the output logits so:
        - compute_loss_and_backward() can reuse them
        - backward() can apply external gradients if orchestrator sends grad_output
        """
        x = x.to(self.device, non_blocking=True)

        with torch.set_grad_enabled(self.is_training):
            out = self.model(x, *args, **kwargs)

        # Cache logits/outputs for potential backward() calls
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
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

    def compute_loss_and_backward(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Full local training step:
        - compute CE loss
        - backward()
        - clip gradients
        Returns scalar loss for logging.
        """
        labels = labels.to(self.device, non_blocking=True)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        if self.is_training:
            loss.backward()

            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

        return float(loss.item())

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

            stacked = torch.stack([worker_grads[i].to(self.device) for worker_grads in all_gradients], dim=0)
            avg_grad = stacked.mean(dim=0)
            param.grad.copy_(avg_grad)
