import torch
from typing import List, Callable, Optional
from .metrics import Metric, masked_target_scores
from torchvision.transforms import GaussianBlur
import math


class ImportanceScore(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        heatmaps: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.heatmaps = heatmaps.squeeze(1)  # (B, H, W)
        self.targets = targets
        self.output_curves: Optional[torch.Tensor] = None

        self.__dict__.update(kwargs)
        self._validate_inputs()
        self.blurred_inputs: Optional[torch.Tensor] = None
        if self.blur_sigma is not None:
            self._precompute_blurred_inputs()

    @staticmethod
    def validate_inputs(inputs: torch.Tensor, targets: torch.Tensor):
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Batch size mismatch between inputs and targets")

    def _validate_inputs(self):
        self.validate_inputs(self.inputs, self.targets)
        if self.heatmaps.ndim != 3:
            raise ValueError("Heatmaps must be 3D tensor (B, H, W)")
        if self.inputs.shape[0] != self.heatmaps.shape[0]:
            raise ValueError("Batch size mismatch between inputs and heatmaps")
        if self.inputs.device != self.heatmaps.device:
            raise ValueError("Inputs and heatmaps must be on the same device")

    @staticmethod
    def _calculate_kernel_size(sigma: float) -> int:
        return int(2 * torch.ceil(torch.tensor(3 * sigma)).item() + 1)

    def _precompute_blurred_inputs(self):
        kernel_size = self._calculate_kernel_size(self.blur_sigma)
        blurrer = GaussianBlur(kernel_size=kernel_size, sigma=self.blur_sigma)
        self.blurred_inputs = blurrer(self.inputs).to(self.inputs.device)

    def update(
        self,
        mode: str = "lif",
        n_steps: int = 100,
        callbacks: Optional[List[Callable]] = None,
        **kwargs,
    ):
        batch_size, H, W = self.heatmaps.shape
        total_pixels = H * W
        steps = n_steps + 1
        device = self.inputs.device
        heatmaps_flat = self.heatmaps.view(batch_size, -1)  # (B, H*W)

        # Sort by importance values
        if mode == "lif":
            sorted_vals, sorted_indices = torch.sort(heatmaps_flat, dim=1)  # Ascending
        elif mode == "mif":
            sorted_vals, sorted_indices = torch.sort(
                heatmaps_flat, dim=1, descending=True
            )  # Descending
        else:
            raise ValueError("mode must be 'lif' or 'mif'")

        # Compute chunk size: pixels removed per step ~ total/(steps-1)
        num_chunks = max(1, steps - 1)
        chunk_size = math.ceil(total_pixels / num_chunks)

        # Prepare randomized sorted indices, applying tie-breaking only within equal-value blocks
        rand_sorted_indices = sorted_indices.clone()
        for b in range(batch_size):
            vals = sorted_vals[b]
            idxs = sorted_indices[b]

            # Find boundaries of equal-value blocks
            diff = vals[1:] != vals[:-1]
            change_pts = torch.nonzero(diff, as_tuple=False).flatten() + 1
            boundaries = [0] + change_pts.tolist() + [total_pixels]

            # Process each value block in original order. A block only yields more
            # than one subchunk when it is longer than chunk_size, so shorter ones
            # are left as the clone already has them instead of being rebuilt.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end - start <= chunk_size:
                    continue
                block = idxs[start:end]
                # Split block into subchunks of size chunk_size and shuffle their order
                subblocks = [
                    block[i : i + chunk_size]
                    for i in range(0, block.numel(), chunk_size)
                ]
                perm = torch.randperm(len(subblocks), device=device)
                rand_sorted_indices[b, start:end] = torch.cat(
                    [subblocks[i] for i in perm]
                )

        # Prepare fractions progression
        if mode == "lif":
            fractions = torch.linspace(1.0, 0.0, steps, device=device)
        else:
            fractions = torch.linspace(0.0, 1.0, steps, device=device)

        # Every step's mask is a scatter of the top-k ranked pixels, so the whole
        # curve is built up front and scored in batches rather than one forward
        # pass per step.
        masks = self.inputs.new_zeros((batch_size, steps, total_pixels))
        for step, fraction in enumerate(fractions.tolist()):
            k = int(fraction * total_pixels)
            selected = (
                rand_sorted_indices[:, total_pixels - k :]
                if mode == "lif"
                else rand_sorted_indices[:, :k]
            )
            masks[:, step].scatter_(1, selected, 1.0)
        masks = masks.view(batch_size, steps, H, W)

        scores = masked_target_scores(
            self.model,
            self.inputs,
            self.blurred_inputs,
            self.targets,
            masks,
        )

        curves = self.inputs.new_empty((batch_size, steps, 2))
        curves[:, :, 0] = fractions
        curves[:, :, 1] = scores

        if callbacks:
            for step in range(steps):
                for callback in callbacks:
                    callback(masks[:, step])

        self.output_curves = curves

    def compute(self) -> torch.Tensor:
        if self.output_curves is None:
            raise RuntimeError("Must run update() before computing AUC.")

        curves = self.output_curves
        batch_size, steps, _ = curves.shape

        cond = (curves[:, 0, 0] == 1.0).view(-1, 1).expand(-1, steps)

        x = torch.where(
            cond,
            1 - curves[:, :, 0],  # LIF: revealed = removed fraction
            0 - curves[:, :, 0],  # MIF: revealed = preserved fraction
        )
        y = curves[:, :, 1]

        auc = torch.trapz(y, x, dim=1)
        return auc

    def reset(self):
        self.output_curves = None
