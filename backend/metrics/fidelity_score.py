"""Fidelity metric normalized against a zero-attribution baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from backend.metrics.metrics import Metric


@dataclass(frozen=True, slots=True)
class GaussianNoise:
    """The noisy baseline of Yeh et al. (2019), for local explanations.

    Every pixel gets i.i.d. noise, which probes the sensitivity of the function
    around ``x`` -- what a local explanation reports.
    """

    std: float

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError("std must be positive")

    def sample(
        self,
        inputs: torch.Tensor,
        count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return torch.randn(
            (count,) + tuple(inputs.shape),
            device=inputs.device,
            dtype=inputs.dtype,
            generator=generator,
        ) * self.std


@dataclass(frozen=True, slots=True)
class SquareRemoval:
    """Square removal of Yeh et al. (2019), for global explanations.

    A uniformly placed square is replaced by ``baseline``, so ``delta`` is zero
    outside the patch and ``delta^T a`` only sums the attributions inside it.
    That asks the question a global explanation claims to answer -- how much
    does the logit drop if this region is removed -- and moves the output enough
    to carry signal, unlike small i.i.d. noise.
    """

    size: int
    baseline: float

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be positive")

    def sample(
        self,
        inputs: torch.Tensor,
        count: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        batch, _, height, width = inputs.shape
        if self.size > min(height, width):
            raise ValueError("size must not exceed the smaller input side")

        removed = inputs - torch.as_tensor(
            self.baseline,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        rows = torch.arange(height, device=inputs.device)
        columns = torch.arange(width, device=inputs.device)
        tops = torch.randint(
            height - self.size + 1,
            (count, batch),
            device=inputs.device,
            generator=generator,
        )
        lefts = torch.randint(
            width - self.size + 1,
            (count, batch),
            device=inputs.device,
            generator=generator,
        )
        in_rows = (rows >= tops.unsqueeze(-1)) & (
            rows < (tops + self.size).unsqueeze(-1)
        )
        in_columns = (columns >= lefts.unsqueeze(-1)) & (
            columns < (lefts + self.size).unsqueeze(-1)
        )
        patch = (in_rows.unsqueeze(-1) & in_columns.unsqueeze(-2)).unsqueeze(2)
        return patch.to(inputs.dtype) * removed.unsqueeze(0)


Perturbation = GaussianNoise | SquareRemoval


class FidelityScore(Metric):
    """Relative reduction in infidelity over a zero attribution.

    For each sampled perturbation ``delta``, the attribution predicts an output
    change ``sum(delta * attribution)`` and the model supplies the observed
    target-logit change ``f(x) - f(x - delta)``. The score is

    ``1 - E[(predicted - observed)^2] / E[observed^2]``.

    One is perfect, zero matches the zero-attribution baseline, and negative
    values are worse than that baseline. A zero baseline error makes the score
    undefined and is represented as ``NaN``.

    Yeh et al. (2019) sample ``delta`` differently for each explanation family
    (section 2.5), so the caller passes the perturbation: `GaussianNoise` for
    local explanations, `SquareRemoval` for global ones. Scores from the two
    measure different things and must not be ranked against each other.
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
                "Attributions must have the same shape as inputs for fidelity"
            )
        if self.attributions.device != self.inputs.device:
            raise ValueError("Inputs and attributions must be on the same device")
        if self.inputs.dim() != 4:
            raise ValueError("Fidelity expects BCHW inputs")

    @staticmethod
    def _target_scores(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return outputs.gather(1, targets.view(-1, 1)).squeeze(1)

    def update(
        self,
        perturbation: Perturbation,
        n_perturb_samples: int = 25,
        max_examples_per_batch: int = 5,
        random_seed: int = 0,
    ) -> None:
        if n_perturb_samples < 1:
            raise ValueError("n_perturb_samples must be positive")
        if max_examples_per_batch < 1:
            raise ValueError("max_examples_per_batch must be positive")

        batch_size = self.inputs.shape[0]
        attribution_error_sum = self.inputs.new_zeros(batch_size)
        baseline_error_sum = self.inputs.new_zeros(batch_size)
        generator = torch.Generator(device=self.inputs.device).manual_seed(random_seed)

        with torch.no_grad():
            original_scores = self._target_scores(
                self.model(self.inputs), self.targets
            )

            sampled = 0
            while sampled < n_perturb_samples:
                count = min(max_examples_per_batch, n_perturb_samples - sampled)
                perturbations = perturbation.sample(self.inputs, count, generator)
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
                attribution_error_sum += (
                    predicted_changes - observed_changes
                ).square().sum(dim=0)
                baseline_error_sum += observed_changes.square().sum(dim=0)
                sampled += count

        result = torch.full_like(attribution_error_sum, torch.nan)
        defined = baseline_error_sum > 0
        result[defined] = (
            1 - attribution_error_sum[defined] / baseline_error_sum[defined]
        )
        self.result = result

    def compute(self) -> torch.Tensor:
        if self.result is None:
            raise RuntimeError("Must run update() before computing fidelity")
        return self.result

    def reset(self) -> None:
        self.result = None
