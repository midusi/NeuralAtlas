"""Segment-wise ablation metric and K-means image segmentation.

The model input and segmentation image are deliberately separate. Model inputs
may be normalized, while RGB-to-Lab conversion requires sRGB values in [0, 1].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F

from backend.metrics.metrics import Metric


class SegmentConfig(ABC):
    """Configuration for producing a partition of each input image."""

    @abstractmethod
    def segment(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return disjoint masks shaped ``(B, K, H, W)``."""
        raise NotImplementedError

    @property
    @abstractmethod
    def k(self) -> int:
        """Number of segments produced per image."""
        raise NotImplementedError


class KmeansConfig(SegmentConfig):
    """Segment sRGB images using K-means over color and spatial features.

    Args:
        k: Number of segments per image.
        use_lab: Convert sRGB to CIE Lab before clustering. When false, cluster
            in linear RGB.
        add_xy: Append normalized pixel coordinates to the color features.
        xy_weight: Weight applied to the spatial coordinates.
        n_iters: Maximum number of K-means iterations.
        seed: Random seed used for centroid initialization. ``None`` uses the
            current PyTorch RNG state.
    """

    def __init__(
        self,
        k: int = 25,
        use_lab: bool = True,
        add_xy: bool = True,
        xy_weight: float = 1.0,
        n_iters: int = 10,
        seed: Optional[int] = 0,
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1")
        if n_iters < 1:
            raise ValueError("n_iters must be at least 1")
        self._k = int(k)
        self.use_lab = bool(use_lab)
        self.add_xy = bool(add_xy)
        self.xy_weight = float(xy_weight)
        self.n_iters = int(n_iters)
        self.seed = seed

    @property
    def k(self) -> int:
        return self._k

    @staticmethod
    def _srgb_to_linear(values: torch.Tensor) -> torch.Tensor:
        return torch.where(
            values <= 0.04045,
            values / 12.92,
            ((values + 0.055) / 1.055) ** 2.4,
        )

    @staticmethod
    def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
        """Convert an sRGB batch in ``[0, 1]`` to CIE Lab."""
        batch_size, channels, height, width = rgb.shape
        if channels != 3:
            raise ValueError("use_lab=True requires 3-channel RGB inputs")

        rgb_linear = KmeansConfig._srgb_to_linear(rgb)
        rgb_to_xyz = rgb_linear.new_tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ]
        )
        xyz = rgb_to_xyz @ rgb_linear.reshape(batch_size, 3, -1)

        x = xyz[:, 0] / 0.95047
        y = xyz[:, 1]
        z = xyz[:, 2] / 1.08883
        epsilon = 216 / 24389
        kappa = 24389 / 27

        def lab_transform(values: torch.Tensor) -> torch.Tensor:
            return torch.where(
                values > epsilon,
                values.pow(1 / 3),
                (kappa * values + 16) / 116,
            )

        fx, fy, fz = (lab_transform(channel) for channel in (x, y, z))
        lab = torch.stack(
            [
                (116 * fy - 16).clamp(0, 100),
                500 * (fx - fy),
                200 * (fy - fz),
            ],
            dim=1,
        )
        return lab.reshape(batch_size, 3, height, width)

    @staticmethod
    def _standardize_per_image(features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True, correction=0).clamp_min(1e-6)
        return (features - mean) / std

    def _build_features(self, image: torch.Tensor) -> torch.Tensor:
        channels, height, width = image.shape
        color = (
            self._rgb_to_lab(image.unsqueeze(0)).squeeze(0)
            if self.use_lab
            else self._srgb_to_linear(image)
        )
        features = [color.reshape(channels, -1).transpose(0, 1)]

        if self.add_xy:
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, height, device=image.device, dtype=image.dtype),
                torch.linspace(0, 1, width, device=image.device, dtype=image.dtype),
                indexing="ij",
            )
            xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2) * self.xy_weight
            features.append(xy)

        return self._standardize_per_image(torch.cat(features, dim=1))

    def _kmeans(self, features: torch.Tensor) -> torch.Tensor:
        num_samples = features.shape[0]
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=features.device)
            generator.manual_seed(self.seed)

        indices = torch.randperm(
            num_samples, generator=generator, device=features.device
        )[: self.k]
        centers = features[indices]

        for _ in range(self.n_iters):
            distances = (
                (features * features).sum(dim=1, keepdim=True)
                - 2 * features @ centers.T
                + (centers * centers).sum(dim=1).unsqueeze(0)
            )
            labels = distances.argmin(dim=1)
            new_centers = torch.zeros_like(centers)
            for cluster in range(self.k):
                members = labels == cluster
                if members.any():
                    new_centers[cluster] = features[members].mean(dim=0)
                else:
                    random_index = torch.randint(
                        num_samples,
                        (),
                        generator=generator,
                        device=features.device,
                    )
                    new_centers[cluster] = features[random_index]

            if torch.allclose(new_centers, centers, atol=1e-5, rtol=0):
                centers = new_centers
                break
            centers = new_centers

        return labels

    def segment(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("segmentation inputs must be 4D (B, C, H, W)")
        if not inputs.is_floating_point():
            raise ValueError("segmentation inputs must be floating point")
        if not torch.isfinite(inputs).all():
            raise ValueError("segmentation inputs must contain only finite values")
        if inputs.numel() and (inputs.min() < 0 or inputs.max() > 1):
            raise ValueError("segmentation inputs must be sRGB values in [0, 1]")

        batch_size, channels, height, width = inputs.shape
        if self.use_lab and channels != 3:
            raise ValueError("use_lab=True requires 3-channel RGB inputs")
        if self.k > height * width:
            raise ValueError("k cannot exceed the number of image pixels")

        masks = torch.zeros(
            (batch_size, self.k, height, width),
            device=inputs.device,
            dtype=inputs.dtype,
        )
        for batch_index in range(batch_size):
            labels = self._kmeans(self._build_features(inputs[batch_index]))
            masks[batch_index] = F.one_hot(
                labels, num_classes=self.k
            ).transpose(0, 1).reshape(self.k, height, width).to(inputs.dtype)
        return masks


class SegmentScore(Metric):
    """Area under a segment-wise insertion or deletion curve.

    ``inputs`` are passed to the model and may use any preprocessing expected by
    it. ``segmentation_inputs`` are used only by ``seg_config`` and must be the
    corresponding, spatially aligned sRGB images in ``[0, 1]``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        heatmaps: torch.Tensor,
        targets: torch.Tensor,
        seg_config: SegmentConfig,
        *,
        segmentation_inputs: torch.Tensor,
        mode: str = "deletion",
        blur_sigma: Optional[float] = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.segmentation_inputs = segmentation_inputs
        self.heatmaps = heatmaps.squeeze(1)
        self.targets = targets
        self.seg_config = seg_config
        self.mode = mode
        self.blur_sigma = blur_sigma
        self.__dict__.update(kwargs)

        self._validate_inputs()
        with torch.no_grad():
            self.segments = self.seg_config.segment(self.segmentation_inputs)
        expected_segment_shape = (
            self.inputs.shape[0],
            self.seg_config.k,
            *self.inputs.shape[-2:],
        )
        if self.segments.shape != expected_segment_shape:
            raise ValueError(
                "seg_config returned masks with shape "
                f"{tuple(self.segments.shape)}; expected {expected_segment_shape}"
            )
        if self.segments.device != self.inputs.device:
            raise ValueError("Segmentation masks and model inputs must share a device")

        self.output_curves: Optional[torch.Tensor] = None
        self.blurred_inputs: Optional[torch.Tensor] = None
        if self.blur_sigma is not None:
            self._precompute_blurred_inputs()
        self._compute_importance_ordering()

    @staticmethod
    def validate_inputs(inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Batch size mismatch between inputs and targets")

    def _validate_inputs(self) -> None:
        self.validate_inputs(self.inputs, self.targets)
        if self.inputs.ndim != 4:
            raise ValueError("inputs must be 4D (B, C, H, W)")
        if self.segmentation_inputs.ndim != 4:
            raise ValueError("segmentation_inputs must be 4D (B, C, H, W)")
        if self.heatmaps.ndim != 3:
            raise ValueError("heatmaps must be 3D (B, H, W)")
        if self.inputs.shape[0] != self.heatmaps.shape[0]:
            raise ValueError("Batch size mismatch between inputs and heatmaps")
        if self.segmentation_inputs.shape[0] != self.inputs.shape[0]:
            raise ValueError("Batch size mismatch between model and segmentation inputs")
        if self.segmentation_inputs.shape[-2:] != self.inputs.shape[-2:]:
            raise ValueError("Model and segmentation inputs must be spatially aligned")
        if self.heatmaps.shape[-2:] != self.inputs.shape[-2:]:
            raise ValueError("Heatmaps and inputs must be spatially aligned")
        if not (
            self.inputs.device
            == self.segmentation_inputs.device
            == self.heatmaps.device
            == self.targets.device
        ):
            raise ValueError("All inputs and targets must be on the same device")
        if self.mode not in {"insertion", "deletion"}:
            raise ValueError("mode must be 'insertion' or 'deletion'")
        if self.blur_sigma is not None and self.blur_sigma <= 0:
            raise ValueError("blur_sigma must be positive")

    def _precompute_blurred_inputs(self) -> None:
        sigma = float(self.blur_sigma)
        radius = max(1, int(3 * sigma))
        kernel_size = 2 * radius + 1
        coordinates = (
            torch.arange(kernel_size, device=self.inputs.device, dtype=self.inputs.dtype)
            - radius
        )
        kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
        kernel = (kernel / kernel.sum()).reshape(1, 1, 1, -1)

        channels = self.inputs.shape[1]
        blurred = F.pad(self.inputs, (radius, radius, 0, 0), mode="reflect")
        blurred = F.conv2d(
            blurred, kernel.expand(channels, 1, 1, -1), groups=channels
        )
        vertical_kernel = kernel.transpose(-1, -2)
        blurred = F.pad(blurred, (0, 0, radius, radius), mode="reflect")
        self.blurred_inputs = F.conv2d(
            blurred, vertical_kernel.expand(channels, 1, -1, 1), groups=channels
        )

    def _compute_importance_ordering(self) -> None:
        areas = self.segments.sum(dim=(2, 3)).clamp_min(1)
        sums = (self.segments * self.heatmaps.unsqueeze(1)).sum(dim=(2, 3))
        self.order = torch.argsort(sums / areas, dim=1)

    def _masked_inputs_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        expanded_mask = mask.unsqueeze(1)
        if self.blurred_inputs is not None:
            return expanded_mask * self.inputs + (1 - expanded_mask) * self.blurred_inputs
        return expanded_mask * self.inputs

    def update(
        self,
        callbacks: Optional[list[Callable[[torch.Tensor], object]]] = None,
        **kwargs: object,
    ) -> None:
        batch_size, _, height, width = self.inputs.shape
        num_segments = self.segments.shape[1]
        if self.mode == "insertion":
            current_mask = self.inputs.new_zeros((batch_size, height, width))
        else:
            current_mask = self.inputs.new_ones((batch_size, height, width))

        curves = self.inputs.new_zeros((batch_size, num_segments + 1, 2))
        batch_indices = torch.arange(batch_size, device=self.inputs.device)

        for step in range(num_segments + 1):
            if step:
                segment_indices = self.order[:, step - 1]
                step_mask = self.segments[batch_indices, segment_indices].to(
                    self.inputs.dtype
                )
                if self.mode == "insertion":
                    current_mask = (current_mask + step_mask).clamp(0, 1)
                else:
                    current_mask = (current_mask - step_mask).clamp(0, 1)

            with torch.no_grad():
                outputs = self.model(self._masked_inputs_from_mask(current_mask))
            curves[:, step, 0] = current_mask.mean(dim=(1, 2))
            curves[:, step, 1] = outputs[batch_indices, self.targets]

            for callback in callbacks or []:
                callback(current_mask)

        self.output_curves = curves

    def compute(self) -> torch.Tensor:
        if self.output_curves is None:
            raise RuntimeError("Must run update() before computing AUC")
        sorted_indices = torch.argsort(self.output_curves[:, :, 0], dim=1)
        x = torch.gather(self.output_curves[:, :, 0], 1, sorted_indices)
        y = torch.gather(self.output_curves[:, :, 1], 1, sorted_indices)
        return torch.trapz(y, x, dim=1)

    def reset(self) -> None:
        self.output_curves = None
