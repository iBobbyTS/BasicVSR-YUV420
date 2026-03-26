from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List


HF_BASE_URL = "https://huggingface.co/datasets/snah/REDS/resolve/main"

STANDARD_SUBSETS = [
    "train_sharp",
    "train_blur",
    "train_blur_comp",
    "train_sharp_bicubic",
    "train_blur_bicubic",
    "train_blur_jpeg",
    "val_sharp",
    "val_blur",
    "val_blur_comp",
    "val_sharp_bicubic",
    "val_blur_bicubic",
    "val_blur_jpeg",
    "test_blur",
    "test_blur_comp",
    "test_sharp_bicubic",
    "test_blur_bicubic",
]

PRESETS = {
    "basicvsr_train": ["train_sharp", "train_sharp_bicubic"],
    "basicvsr_train_val": ["train_sharp", "train_sharp_bicubic", "val_sharp", "val_sharp_bicubic"],
    "all_public_standard": STANDARD_SUBSETS,
}


def build_subset_url(subset: str) -> str:
    return "{}/{}.zip".format(HF_BASE_URL, subset)


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return "{:.2f} {}".format(size, unit)
        size /= 1024.0
    return "{} B".format(num_bytes)


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "basicvsr-yuv420-reds-downloader"})

    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        total_size = response.headers.get("Content-Length")
        total_bytes = int(total_size) if total_size is not None else None
        downloaded = 0

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)

            if total_bytes:
                percent = downloaded * 100.0 / total_bytes
                message = "\rDownloading {} / {} ({:.2f}%)".format(
                    format_size(downloaded),
                    format_size(total_bytes),
                    percent,
                )
            else:
                message = "\rDownloading {}".format(format_size(downloaded))
            sys.stdout.write(message)
            sys.stdout.flush()

    sys.stdout.write("\n")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)


def resolve_requested_subsets(args: argparse.Namespace) -> List[str]:
    subsets = []  # type: List[str]

    for preset in args.presets:
        for subset in PRESETS[preset]:
            if subset not in subsets:
                subsets.append(subset)

    for subset in args.subsets:
        if subset not in STANDARD_SUBSETS:
            raise ValueError("Unknown REDS subset: {}".format(subset))
        if subset not in subsets:
            subsets.append(subset)

    if not subsets:
        raise ValueError("No REDS subsets selected. Use --preset or --subsets.")

    return subsets


def print_available_options() -> None:
    print("Available presets:")
    for name, subsets in PRESETS.items():
        print("  {}: {}".format(name, ", ".join(subsets)))
    print("")
    print("Available standard subsets:")
    for subset in STANDARD_SUBSETS:
        print("  {}".format(subset))


def iterate_downloads(subsets: Iterable[str], output_dir: Path, extract: bool, keep_zip: bool, skip_existing: bool) -> None:
    for subset in subsets:
        url = build_subset_url(subset)
        zip_path = output_dir / "{}.zip".format(subset)
        extract_dir = output_dir / subset

        print("Subset: {}".format(subset))
        print("URL: {}".format(url))
        print("Archive: {}".format(zip_path))

        if skip_existing and zip_path.exists():
            print("Skipping existing archive.")
        else:
            download_file(url, zip_path)

        if extract:
            if skip_existing and extract_dir.exists() and any(extract_dir.iterdir()):
                print("Skipping extraction because target directory already exists and is not empty.")
            else:
                print("Extracting to {}".format(extract_dir))
                extract_zip(zip_path, extract_dir)

            if not keep_zip and zip_path.exists():
                zip_path.unlink()
                print("Removed archive {}".format(zip_path))

        print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the official REDS dataset from the official Hugging Face mirrors."
    )
    parser.add_argument(
        "--preset",
        dest="presets",
        nargs="*",
        default=[],
        choices=sorted(PRESETS.keys()),
        help="Named subset groups to download.",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=[],
        help="Explicit REDS subsets to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/REDS_downloads",
        help="Directory used for downloaded archives and extracted folders.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract each downloaded zip archive.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the zip archive after extraction.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip archives or extracted folders that already exist.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available presets and subsets, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print_available_options()
        return

    subsets = resolve_requested_subsets(args)
    output_dir = Path(args.output_dir)
    iterate_downloads(
        subsets=subsets,
        output_dir=output_dir,
        extract=args.extract,
        keep_zip=args.keep_zip,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
