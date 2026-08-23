#!/usr/bin/env python
"""Deterministic full sweep: ensure the dataset, run the pipeline in growing steps, push
to Hugging Face, then drop the local attribution files.

Built for a remote GPU box with little disk: only one step of attribution images ever
exists locally. Everything is resumable — progress is derived from the persisted run
JSON, so re-running the script picks up exactly where it left off.

Per (model, step) with --chunk N each step covers the half-open window it adds --
[0,N), [N,2N), [2N,3N) ... up to the total -- so no sample is ever computed twice:
  1. `main.py --model M --dataset D --start-index <start> --num-samples <target>`
  2. upload `outputs/images/M__D__*.avif` + the run JSON to the HF attributions repo
  3. delete the just-uploaded images from disk

Configuration comes from `.env` (see `.env.example`):
  HF_TOKEN                 write token for the attributions repo (required)
  HF_ATTRIBUTIONS_REPO     default Matgc04/neuralatlas-attributions

Examples:
  uv run python scripts/run_sweep.py --dry-run
  uv run python scripts/run_sweep.py --chunk 100
  uv run python scripts/run_sweep.py --models resnet101 --total 200
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypeVar

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend import config  # noqa: E402
from backend.ai_dataset.core import load_env  # noqa: E402
from backend.methods import method_catalog  # noqa: E402
from backend.persistence import OutputRepository  # noqa: E402

DEFAULT_MODELS = [
    "resnet101",
    "efficientnet_b4",
    "inception_v3",
    "mobilenet_v2",
    "convnext_tiny",
]
DEFAULT_ATTRIBUTIONS_REPO = "Matgc04/neuralatlas-attributions"
T = TypeVar("T")


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def with_retries(label: str, action: Callable[[], T], attempts: int = 4) -> T:
    """Retry a network action with exponential backoff; a multi-day run will hit blips."""
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = 15 * 2 ** (attempt - 1)
            log(f"warn: {label} failed ({error!r}); retrying in {delay}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


# ------------------------------------------------------------------------- dataset


def dataset_dir(dataset: str) -> Path:
    return REPO_ROOT / config.BASE_PUBLIC_DIR / dataset


def count_dataset_images(dataset: str) -> int:
    val_dir = dataset_dir(dataset) / "val"
    if not val_dir.is_dir():
        return 0
    return sum(1 for path in val_dir.glob("*/*") if path.is_file())


def expected_dataset_images(dataset: str) -> int | None:
    """Image count declared by `<dataset>_structure.json`, if the manifest is present."""
    structure_path = dataset_dir(dataset) / f"{dataset}_structure.json"
    if not structure_path.is_file():
        return None
    structure = json.loads(structure_path.read_text())
    return sum(len(files) for files in structure.values())


def ensure_dataset(dataset: str, dry_run: bool = False) -> int:
    """Return the number of dataset images, building them first if they are missing."""
    present = count_dataset_images(dataset)
    expected = expected_dataset_images(dataset)
    if present > 0 and (expected is None or present == expected):
        log(f"Dataset {dataset} already present: {present} images")
        return present

    if dry_run:
        log(f"Dataset {dataset} incomplete ({present} images, expected {expected}); would download")
        return expected or present

    log(f"Dataset {dataset} incomplete ({present} images, expected {expected}); downloading")
    # Run through `uv run`, not sys.executable: the download script declares its own
    # deps (kagglehub, Pillow) in a PEP-723 header and they are not project deps.
    subprocess.run(
        [
            "uv",
            "run",
            str(REPO_ROOT / "scripts/download_nano_imagenet.py"),
            "--name",
            dataset,
            "--fill-from-train",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    present = count_dataset_images(dataset)
    if present == 0:
        raise SystemExit(f"Dataset {dataset} is still empty after downloading.")
    log(f"Dataset {dataset} ready: {present} images")
    return present


# ------------------------------------------------------------------------ progress


def completed_samples(repository: OutputRepository, model: str, dataset: str, image_ext: str) -> int:
    """How far into the dataset every method has already been persisted.

    Samples are always processed in dataset order, so the per-method output count is a
    prefix length; the slowest method decides how much of the run is truly complete.
    """
    output_counts = repository.method_output_counts(model, dataset, image_ext)
    counts = [output_counts.get(entry.id, 0) for entry in method_catalog()]
    return min(counts) if counts else 0


def sync_run_metadata(api, repo_id: str, model: str, dataset: str) -> int:
    """Restore this worker's run checkpoint from HF without touching global metadata."""
    paths = [
        f"runs/{model}/{dataset}/images.json",
        f"runs/{model}/{dataset}/summary.json",
    ]
    downloaded = 0
    for path in paths:
        exists = with_retries(
            f"check remote {path}",
            lambda path=path: api.file_exists(
                repo_id=repo_id,
                repo_type="dataset",
                filename=path,
            ),
        )
        if not exists:
            continue
        with_retries(
            f"download remote {path}",
            lambda path=path: api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=path,
                local_dir=str(REPO_ROOT / config.OUTPUT_ROOT),
                force_download=True,
            ),
        )
        downloaded += 1
    return downloaded


# --------------------------------------------------------------------------- steps


def run_step(model: str, dataset: str, start: int, target: int, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "main.py",
        "--model", model,
        "--dataset", dataset,
        "--start-index", str(start),
        "--num-samples", str(target),
        "--image-ext", args.image_ext,
        "--export-batch-images", str(args.export_batch_images),
        "--metrics", *args.metrics,
    ]
    log(f"$ {' '.join(command[1:])}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def step_image_files(model: str, dataset: str, image_ext: str) -> list[Path]:
    images_dir = REPO_ROOT / config.OUTPUT_IMAGES_DIR
    if not images_dir.is_dir():
        return []
    prefix = f"{model}__{dataset}__"
    suffix = f".{image_ext}"
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.suffix.lower() == suffix
    )


def upload_step(api, repo_id: str, model: str, label: str, dataset: str, image_ext: str) -> int:
    """Upload this worker's images and run metadata, never shared global metadata."""
    files = step_image_files(model, dataset, image_ext)
    if files:
        with_retries(
            f"upload images for {label}",
            lambda: api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(REPO_ROOT / config.OUTPUT_IMAGES_DIR),
                path_in_repo="images",
                allow_patterns=[f"{model}__{dataset}__*.{image_ext}"],
                commit_message=f"attributions: {label} ({len(files)} files)",
            ),
        )
    with_retries(
        f"upload metadata for {label}",
        lambda: api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(REPO_ROOT / config.OUTPUT_ROOT),
            path_in_repo="",
            allow_patterns=[
                f"runs/{model}/{dataset}/images.json",
                f"runs/{model}/{dataset}/summary.json",
            ],
            commit_message=f"metadata: {label}",
        ),
    )
    return len(files)


def cleanup_step(model: str, dataset: str, image_ext: str) -> int:
    files = step_image_files(model, dataset, image_ext)
    for path in files:
        path.unlink()
    return len(files)


# ---------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"Torchvision models to sweep (default: {' '.join(DEFAULT_MODELS)})")
    parser.add_argument("--dataset", default="imagenet-pico",
                        help="Dataset directory under interpretability-viewer/public (default: imagenet-pico)")
    parser.add_argument("--total", type=int,
                        help="Upper bound on samples per model (default: the whole dataset). "
                             "Samples run in class order, so a partial total only covers the "
                             "lowest class ids.")
    parser.add_argument("--chunk", type=int, default=100,
                        help="How much the --num-samples target grows per run/upload/cleanup "
                             "cycle (default: 100)")
    parser.add_argument("--image-ext", default=config.DEFAULT_IMAGE_EXT,
                        help=f"Attribution image format (default: {config.DEFAULT_IMAGE_EXT})")
    parser.add_argument("--metrics", nargs="*", default=list(config.FAITHFULNESS_METRICS),
                        choices=list(config.FAITHFULNESS_METRICS),
                        metavar="{" + ",".join(config.FAITHFULNESS_METRICS) + "}",
                        help="Faithfulness metrics to compute (default: all)")
    parser.add_argument("--export-batch-images", type=int, default=10,
                        help="Images buffered before the run JSON is rewritten (default: 10)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip the Hugging Face upload (implies --keep-local)")
    parser.add_argument("--keep-local", action="store_true",
                        help="Do not delete the attribution images after uploading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without running anything")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk <= 0 or (args.total is not None and args.total <= 0):
        raise SystemExit("--chunk and --total must be positive.")
    args.image_ext = args.image_ext.lstrip(".").lower()

    load_env(REPO_ROOT / ".env")
    token = os.getenv("HF_TOKEN")
    attributions_repo = os.getenv("HF_ATTRIBUTIONS_REPO", DEFAULT_ATTRIBUTIONS_REPO)
    upload = not args.no_upload
    cleanup = upload and not args.keep_local

    api = None
    if upload:
        from huggingface_hub import HfApi

        if not token:
            raise SystemExit("HF_TOKEN is not set (put it in .env), or pass --no-upload.")
        api = HfApi(token=token)
        # Fail fast on a bad token or a missing repo rather than after hours of compute.
        api.repo_info(repo_id=attributions_repo, repo_type="dataset")
        log(f"Uploading to hf.co/datasets/{attributions_repo} as {api.whoami().get('name', '?')}")

    available = ensure_dataset(args.dataset, dry_run=args.dry_run)
    total = min(args.total, available) if args.total else available
    targets = [min(target, total) for target in range(args.chunk, total + args.chunk, args.chunk)]
    repository = OutputRepository(REPO_ROOT / config.OUTPUT_ROOT)

    log(
        f"Plan: {len(args.models)} models x {total} samples in {len(targets)} steps of "
        f"{args.chunk} | metrics={args.metrics or ['none']} | upload={upload} cleanup={cleanup}"
    )
    for model in args.models:
        done = completed_samples(repository, model, args.dataset, args.image_ext)
        log(f"  {model}: {done}/{total} samples already complete")
    if args.dry_run:
        return

    failures: list[str] = []
    for model in args.models:
        # An interrupted or upload-failed run leaves images on disk that the run JSON
        # already counts as done. That local checkpoint is newer than HF, so flush it
        # before considering a remote restore. With no pending files, HF is the source
        # of truth and lets a fresh GPU box resume an existing run.
        pending = step_image_files(model, args.dataset, args.image_ext)
        if pending and upload:
            log(f"Reconciling {len(pending)} leftover files for {model}")
            upload_step(api, attributions_repo, model, f"{model} resume", args.dataset, args.image_ext)
            if cleanup:
                cleanup_step(model, args.dataset, args.image_ext)
        elif upload:
            downloaded = sync_run_metadata(api, attributions_repo, model, args.dataset)
            if downloaded:
                log(f"Restored {downloaded} checkpoint files from HF for {model}")

        done = completed_samples(repository, model, args.dataset, args.image_ext)
        log(f"Starting {model} from remote-confirmed checkpoint {done}/{total}")

        for target in targets:
            if done >= target:
                log(f"Skipping {model} @{target}: already complete")
                continue
            # Each step covers only the samples it adds. Passing a plain prefix instead
            # would recompute every earlier sample, making the sweep quadratic.
            start = done
            label = f"{model} @{start}-{target}"
            try:
                with_retries(
                    f"step {label}",
                    lambda: run_step(model, args.dataset, start, target, args),
                    attempts=2,
                )
                if upload:
                    uploaded = upload_step(
                        api, attributions_repo, model, label, args.dataset, args.image_ext
                    )
                    log(f"Uploaded {uploaded} attribution files for {label}")
            except Exception as error:
                # Never clean up after a failed upload — the only copy is still local.
                log(f"error: {label} failed ({error!r}); abandoning {model}")
                failures.append(label)
                break

            done = target
            if cleanup:
                log(f"Freed {cleanup_step(model, args.dataset, args.image_ext)} local files")
            free_gb = shutil.disk_usage(REPO_ROOT).free / 1e9
            log(f"Done {label} | {free_gb:.1f} GB free")

    if failures:
        raise SystemExit(f"Sweep finished with failures: {', '.join(failures)}")
    log("Sweep complete.")


if __name__ == "__main__":
    main()
