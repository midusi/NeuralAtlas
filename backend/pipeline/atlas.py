from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from captum._utils.typing import Module, TensorOrTupleOfTensorsGeneric
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm.auto import tqdm

from attr_config import AttributionConfig
from backend import config
from backend.methods import InsufficientFeaturesError
from backend.records import AttributionFailure, ImageRecord, MetricValue, PredictionRecord


def evaluate_faithfulness(
    model: Module,
    inputs: torch.Tensor,
    attribution: torch.Tensor,
    target: torch.Tensor,
    metrics: set[str],
    segments: torch.Tensor | None = None,
) -> dict[str, MetricValue]:
    """Faithfulness scores for one attribution map, keyed by metric name.

    The heatmap is reduced from the attribution the same way the renderer does
    (channel sum, absolute value) and min-max normalized to [0, 1] so the
    morphology threshold is meaningful across methods with different scales.
    Fidelity instead receives the original, unreduced attribution tensor
    because its magnitude is part of the underlying infidelity calculation.
    """
    from backend.metrics import (
        FidelityScore,
        ImportanceScore,
        KmeansConfig,
        MorphScore,
        SegmentScore,
    )

    heatmap = attribution.detach().sum(dim=1, keepdim=True).abs()  # (B, 1, H, W)
    flat = heatmap.flatten(1)
    lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
    hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
    heatmap = (heatmap - lo) / (hi - lo).clamp_min(1e-8)
    inputs = inputs.detach()

    n_steps = config.FAITHFULNESS_N_STEPS
    blur_sigma = config.FAITHFULNESS_BLUR_SIGMA

    scores: dict[str, MetricValue] = {}
    for mode in ("mif", "lif"):
        if mode in metrics:
            metric = ImportanceScore(model, inputs, heatmap, target, blur_sigma=blur_sigma)
            metric.update(mode=mode, n_steps=n_steps)
            scores[mode] = float(metric.compute()[0].item())
    if "morph" in metrics:
        metric = MorphScore(model, inputs, heatmap, target, blur_sigma=config.MORPH_BLUR_SIGMA)
        metric.update(mode="erode", n_steps=n_steps)
        scores["morph"] = float(metric.compute()[0].item())
    if "segment" in metrics:
        metric = SegmentScore(
            model,
            inputs,
            heatmap,
            target,
            KmeansConfig(),
            segmentation_inputs=inputs,
            segments=segments,
            mode="deletion",
            blur_sigma=blur_sigma,
        )
        metric.update()
        scores["segment"] = float(metric.compute()[0].item())
    if "fidelity" in metrics:
        metric = FidelityScore(model, inputs, attribution, target)
        metric.update(
            n_perturb_samples=config.FIDELITY_N_PERTURB_SAMPLES,
            noise_std=config.FIDELITY_NOISE_STD,
            max_examples_per_batch=config.FIDELITY_MAX_EXAMPLES_PER_BATCH,
            random_seed=config.FIDELITY_RANDOM_SEED,
        )
        fidelity = float(metric.compute()[0].item())
        scores["fidelity"] = fidelity if math.isfinite(fidelity) else None
    return scores


def build_output_filename(
    model_name: str,
    dataset_name: str,
    class_id: str,
    image_id: str,
    method_name: str,
    image_ext: str,
) -> str:
    return f"{model_name}__{dataset_name}__{class_id}__{image_id}__{method_name}.{image_ext}"


@dataclass(slots=True)
class AttributionRenderer:
    out_px: int = 224
    outlier_perc: float = 2.0

    @staticmethod
    def _cumulative_sum_threshold(values: np.ndarray, percentile: float) -> float:
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile for thresholding must be between 0 and 100.")

        flat_values = np.sort(values.reshape(-1))
        if flat_values.size == 0:
            return 0.0

        cum_sums = np.cumsum(flat_values)
        total = float(cum_sums[-1])
        if total <= 0:
            return 0.0

        threshold_idx = int(np.searchsorted(cum_sums, total * 0.01 * percentile))
        threshold_idx = min(threshold_idx, flat_values.size - 1)
        return float(flat_values[threshold_idx])

    def _normalize_absolute_heatmap(self, attr: torch.Tensor, outlier_perc: float) -> np.ndarray:
        attr_np = attr.permute(1, 2, 0).detach().cpu().numpy()
        attr_combined = np.abs(np.sum(attr_np, axis=-1))
        attr_combined -= attr_combined.min()
        threshold = self._cumulative_sum_threshold(
            attr_combined,
            100.0 - outlier_perc,
        )
        if threshold <= 0:
            return np.zeros_like(attr_combined, dtype=np.float32)
        return np.clip(attr_combined / threshold, 0, 1)

    def render(
        self,
        attr: TensorOrTupleOfTensorsGeneric,
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        class_id: str,
        image_id: str,
        method_name: str,
        image_ext: str,
        **kwargs: Any,
    ) -> str:
        if not isinstance(attr, torch.Tensor):
            raise TypeError(
                f"Expected attribution tensor for visualization, got {type(attr)}"
            )
        outlier_perc = float(kwargs.get("outlier_perc", self.outlier_perc))
        attr_np = self._normalize_absolute_heatmap(attr, outlier_perc)
        gray = (attr_np * 255).round().astype(np.uint8)
        img = Image.fromarray(gray, mode='L').resize((self.out_px, self.out_px), Image.LANCZOS)

        filename = build_output_filename(
            model_name=model_name,
            dataset_name=dataset_name,
            class_id=class_id,
            image_id=image_id,
            method_name=method_name,
            image_ext=image_ext,
        )
        save_kwargs: dict[str, Any] = {"format": image_ext.upper()}
        if image_ext == "webp":
            save_kwargs.update({"quality": 85, "method": 6, "lossless": False})
        elif image_ext in {"jpg", "jpeg"}:
            save_kwargs.update({"quality": 85, "optimize": True, "progressive": True})
        elif image_ext == "avif":
            save_kwargs.update({"quality": 75, "speed": 6})
        img.save(output_dir / filename, **save_kwargs)
        return f"{config.OUTPUT_IMAGES_BASE_URL}/{filename}"


class AtlasRunner:
    def __init__(
        self,
        model: Module,
        data: str | datasets.VisionDataset,
        interp_methods: list[AttributionConfig],
        renderer: AttributionRenderer | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        if isinstance(data, str):
            self.data = datasets.ImageFolder(data, transform=kwargs.get("transform", None))
        else:
            self.data = data
        self.interp_methods = interp_methods
        self.renderer = renderer or AttributionRenderer()

    @staticmethod
    def _serialize_prediction(model_output: object) -> PredictionRecord:
        if isinstance(model_output, torch.Tensor):
            logits = model_output
        elif (
            isinstance(model_output, (list, tuple))
            and len(model_output) > 0
            and isinstance(model_output[0], torch.Tensor)
        ):
            logits = model_output[0]
        else:
            raise TypeError(
                "Expected model output tensor (or tuple/list with tensor first item), "
                f"got {type(model_output)}."
            )
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        sample_logits = logits[0].detach().cpu()
        predicted_class_id = int(torch.argmax(sample_logits).item())
        confidence = float(torch.softmax(sample_logits, dim=0)[predicted_class_id].item())
        return PredictionRecord(
            predicted_class_id=str(predicted_class_id),
            confidence=confidence,
        )

    def _resolve_original_url(
        self,
        dataset_name: str,
        class_id: str,
        sample_index: int,
    ) -> str | None:
        sample_path: str | None = None
        if hasattr(self.data, "samples"):
            samples = getattr(self.data, "samples")
            if isinstance(samples, list) and sample_index < len(samples):
                sample = samples[sample_index]
                if isinstance(sample, tuple) and sample:
                    sample_path = str(sample[0])
        if sample_path is None:
            return None
        filename = Path(sample_path).name
        return f"/{dataset_name}/val/{class_id}/{filename}"

    def _replay_image_counters(self, start_index: int) -> defaultdict[str, int]:
        """Per-class image counts for the samples before `start_index`.

        `image_id` is a per-class sequence number, so a window that does not begin at
        0 has to replay the skipped samples' class ids; otherwise the ids restart at
        "0" and collide with the ones an earlier window already wrote.
        """
        counters: defaultdict[str, int] = defaultdict(int)
        if not start_index:
            return counters
        samples = getattr(self.data, "samples", None)
        if not isinstance(samples, list):
            raise TypeError(
                "start_index requires a dataset exposing .samples (e.g. ImageFolder) so "
                "image ids stay aligned with a run that starts at 0."
            )
        for _, skipped_target in islice(samples, start_index):
            counters[self.data.classes[skipped_target]] += 1
        return counters

    def stream(
        self,
        num_samples: int,
        output_dir: Path,
        model_name: str,
        dataset_name: str,
        image_ext: str = "webp",
        metrics: set[str] | None = None,
        start_index: int = 0,
        **kwargs: Any,
    ) -> Iterator[ImageRecord]:

        if start_index < 0:
            raise ValueError(f"start_index must be non-negative, got {start_index}")
        window = range(start_index, min(num_samples, len(self.data)))
        if not window:
            return

        self.model.eval()
        # Subset instead of skipping inside the loop: samples before the window are
        # never decoded, so a late chunk costs the same as an early one.
        dataloader = DataLoader(Subset(self.data, window), batch_size=1, shuffle=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_ext = image_ext.lstrip(".").lower()

        class_image_counters = self._replay_image_counters(start_index)

        with tqdm(total=len(window), desc="Interpreting + Saving") as pbar:
            for offset, (inputs, target) in enumerate(dataloader):
                sample_index = start_index + offset

                # ImageFolder sorts class directories as strings, so the target index
                # only matches the directory name for small datasets ("0".."9"). Read
                # the name back so class ids and attribution targets stay aligned with
                # the real ImageNet class ids.
                class_id = self.data.classes[target.item()]
                attribution_target = torch.tensor(
                    [int(class_id)],
                    device=inputs.device,
                    dtype=target.dtype,
                )
                image_id = str(class_image_counters[class_id])
                original_url = self._resolve_original_url(dataset_name, class_id, sample_index)

                with torch.no_grad():
                    prediction = self._serialize_prediction(self.model(inputs))

                segments = None
                if self.interp_methods and metrics and "segment" in metrics:
                    from backend.metrics import KmeansConfig

                    with torch.no_grad():
                        segments = KmeansConfig().segment(inputs)

                record = ImageRecord(
                    model=model_name,
                    dataset=dataset_name,
                    class_id=class_id,
                    image_id=image_id,
                    original_url=original_url,
                    prediction=prediction,
                    interpretability_metrics={},
                )

                inputs.requires_grad = True
                with tqdm(self.interp_methods, leave=False) as method_pbar:
                    for interp_method in method_pbar:
                        method_name = str(interp_method)
                        method_pbar.set_description(f"Attribution {method_name}")
                        try:
                            attribution = interp_method.attribute(
                                self.model,
                                inputs,
                                attribution_target,
                            )
                        except InsufficientFeaturesError as error:
                            record.attribution_failures[method_name] = AttributionFailure(
                                code="insufficient_features",
                                feature_count=error.feature_count,
                            )
                            method_pbar.write(
                                f"Skipping {method_name} for sample {sample_index}: {error}"
                            )
                            continue
                        if not isinstance(attribution, torch.Tensor):
                            raise TypeError(
                                "Streaming visualization requires tensor attribution output; "
                                f"got {type(attribution)} from {method_name}."
                            )
                        output_url = self.renderer.render(
                            attr=attribution[0],
                            output_dir=output_dir,
                            model_name=model_name,
                            dataset_name=dataset_name,
                            class_id=class_id,
                            image_id=image_id,
                            method_name=method_name,
                            image_ext=image_ext,
                            **kwargs,
                        )
                        record.outputs[method_name] = output_url

                        if metrics:
                            record.interpretability_metrics[method_name] = (
                                evaluate_faithfulness(
                                    self.model,
                                    inputs,
                                    attribution,
                                    attribution_target,
                                    metrics,
                                    segments,
                                )
                            )

                class_image_counters[class_id] += 1
                pbar.update(1)
                yield record
