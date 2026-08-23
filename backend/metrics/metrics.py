from abc import ABC, abstractmethod
from typing import Optional
import torch

from backend import config


class Metric(ABC):
    def __init__(self):
        self.result: Optional[torch.Tensor] = None

    @abstractmethod
    def compute(self) -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @staticmethod
    @abstractmethod
    def validate_inputs(inputs: torch.Tensor, targets: torch.Tensor):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}"


def masked_target_scores(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    blurred_inputs: Optional[torch.Tensor],
    targets: torch.Tensor,
    masks: torch.Tensor,
    batch_size: int = config.METRIC_BATCH_SIZE,
) -> torch.Tensor:
    """Target logit for each of the ``(B, S)`` masks, as a ``(B, S)`` tensor.

    Every faithfulness metric walks a sequence of masks over the same image and
    reads one logit per step. The steps do not depend on each other, so the
    masked images go through the model stacked instead of one at a time.
    ``blurred_inputs`` fills what the mask hides; ``None`` blacks it out.
    """
    batch, steps = masks.shape[0], masks.shape[1]
    flat_masks = masks.reshape(batch * steps, 1, *masks.shape[2:])
    # Row i of the stack belongs to image i // steps; indexing per chunk keeps
    # only `batch_size` copies of the image alive instead of B * steps.
    owner = torch.arange(batch * steps, device=inputs.device) // steps

    scores = inputs.new_empty(batch * steps)
    for start in range(0, batch * steps, batch_size):
        stop = start + batch_size
        rows = owner[start:stop]
        chunk_mask = flat_masks[start:stop]
        masked = chunk_mask * inputs[rows]
        if blurred_inputs is not None:
            masked = masked + (1 - chunk_mask) * blurred_inputs[rows]
        with torch.no_grad():
            outputs = model(masked)
        scores[start:stop] = outputs.gather(1, targets[rows].view(-1, 1)).squeeze(1)
    return scores.view(batch, steps)
