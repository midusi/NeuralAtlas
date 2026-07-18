"""Infidelity metric for attribution maps."""

from __future__ import annotations

import torch

from backend.metrics.metrics import Metric


class InfidelityScore(Metric):
    """Mean squared attribution error under Gaussian input perturbations.

    For each sampled perturbation ``delta``, this compares the attribution's
    predicted output change ``sum(delta * attribution)`` with the model's
    observed target-logit change ``f(x) - f(x - delta)``. Lower is better.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        attributions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.inputs = inputs.detach()
        self.attributions = attributions.detach()
        self.targets = targets.detach()
        self._validate_inputs()

    @staticmethod
    def validate_inputs(inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Batch size mismatch between inputs and targets")

    def _validate_inputs(self) -> None:
        self.validate_inputs(self.inputs, self.targets)
        if self.attributions.shape != self.inputs.shape:
            raise ValueError(
                "Attributions must have the same shape as inputs for infidelity"
            )
        if self.attributions.device != self.inputs.device:
            raise ValueError("Inputs and attributions must be on the same device")

    @staticmethod
    def _target_scores(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return outputs.gather(1, targets.view(-1, 1)).squeeze(1)

    def update(
        self,
        n_perturb_samples: int = 25,
        noise_std: float = 0.2,
        max_examples_per_batch: int = 5,
        random_seed: int = 0,
    ) -> None:
        if n_perturb_samples < 1:
            raise ValueError("n_perturb_samples must be positive")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        if max_examples_per_batch < 1:
            raise ValueError("max_examples_per_batch must be positive")

        batch_size = self.inputs.shape[0]
        squared_error_sum = self.inputs.new_zeros(batch_size)
        generator = torch.Generator(device=self.inputs.device).manual_seed(random_seed)

        with torch.no_grad():
            original_scores = self._target_scores(
                self.model(self.inputs), self.targets
            )

            sampled = 0
            while sampled < n_perturb_samples:
                count = min(max_examples_per_batch, n_perturb_samples - sampled)
                perturbations = torch.randn(
                    (count,) + tuple(self.inputs.shape),
                    device=self.inputs.device,
                    dtype=self.inputs.dtype,
                    generator=generator,
                ) * noise_std
                perturbed_inputs = self.inputs.unsqueeze(0) - perturbations

                flat_inputs = perturbed_inputs.flatten(0, 1)
                repeated_targets = self.targets.repeat(count)
                perturbed_scores = self._target_scores(
                    self.model(flat_inputs), repeated_targets
                ).view(count, batch_size)

                predicted_changes = (
                    perturbations * self.attributions.unsqueeze(0)
                ).flatten(2).sum(dim=2)
                observed_changes = original_scores.unsqueeze(0) - perturbed_scores
                squared_error_sum += (
                    predicted_changes - observed_changes
                ).square().sum(dim=0)
                sampled += count

        self.result = squared_error_sum / n_perturb_samples

    def compute(self) -> torch.Tensor:
        if self.result is None:
            raise RuntimeError("Must run update() before computing infidelity")
        return self.result

    def reset(self) -> None:
        self.result = None
