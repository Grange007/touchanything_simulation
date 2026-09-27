#!/usr/bin/env python3
"""Export color/normal/depth/mask images for frame indices 0..n."""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FrameEntry:
    idx: int
    base: Path
    color_path: Path


class ImageWriter:
    """Write image arrays with any available backend."""

    def __init__(self) -> None:
        self.backend = None
        self.cv2 = None
        self.pil_image = None
        self.iio = None

        try:
            import cv2  # type: ignore

            self.backend = "cv2"
            self.cv2 = cv2
            return
        except Exception:
            pass

        try:
            from PIL import Image  # type: ignore

            self.backend = "pil"
            self.pil_image = Image
            return
        except Exception:
            pass

        try:
            import imageio.v3 as iio  # type: ignore

            self.backend = "imageio"
            self.iio = iio
            return
        except Exception:
            pass

        self.backend = "builtin_png"

    def save_png(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.backend == "cv2":
            img = array
            if array.ndim == 3 and array.shape[2] == 3:
                img = self.cv2.cvtColor(array, self.cv2.COLOR_RGB2BGR)
            ok = self.cv2.imwrite(str(path), img)
            if not ok:
                raise RuntimeError(f"Failed to write image: {path}")
            return

        if self.backend == "pil":
            self.pil_image.fromarray(array).save(str(path))
            return

        if self.backend == "imageio":
            self.iio.imwrite(str(path), array)
            return

        save_png_builtin(path, array)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = struct.pack("!I", len(data))
    crc = struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return length + chunk_type + data + crc


def save_png_builtin(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        raise ValueError(f"builtin PNG writer expects uint8, got {arr.dtype}")

    if arr.ndim == 2:
        h, w = arr.shape
        color_type = 0  # grayscale
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        color_type = 2  # RGB
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    else:
        raise ValueError(f"Unsupported array shape for PNG: {arr.shape}")

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    idat = zlib.compress(raw, level=9)

    with open(path, "wb") as f:
        f.write(signature)
        f.write(_png_chunk(b"IHDR", ihdr))
        f.write(_png_chunk(b"IDAT", idat))
        f.write(_png_chunk(b"IEND", b""))


FRAME_COLOR_RE = re.compile(
    r"^(?P<base>.+_(?P<idx>\d{4}))_color\.(png|jpg|jpeg)$", re.IGNORECASE
)


def resolve_data_dir(input_dir: Path) -> Path:
    direct_has_color = any(input_dir.glob("*_color.*"))
    if direct_has_color:
        return input_dir

    raw_dir = input_dir / "raw"
    raw_has_color = raw_dir.is_dir() and any(raw_dir.glob("*_color.*"))
    if raw_has_color:
        return raw_dir

    raise FileNotFoundError(
        f"No '*_color.*' files found in '{input_dir}' or '{input_dir / 'raw'}'"
    )


def discover_entries(data_dir: Path) -> dict[int, FrameEntry]:
    entries: dict[int, FrameEntry] = {}
    for color_file in sorted(data_dir.glob("*_color.*")):
        m = FRAME_COLOR_RE.match(color_file.name)
        if not m:
            continue

        idx = int(m.group("idx"))
        base = data_dir / m.group("base")
        entries[idx] = FrameEntry(idx=idx, base=base, color_path=color_file)
    return entries


def normal_to_uint8(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=np.float32)
    if n.ndim != 3:
        raise ValueError(f"normal must be 3D, got shape {n.shape}")

    if n.shape[0] == 3 and n.shape[2] != 3:
        n = np.transpose(n, (1, 2, 0))
    if n.shape[2] != 3:
        raise ValueError(f"normal last dim must be 3, got shape {n.shape}")

    n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)

    # Most datasets store normal in [-1, 1]. If already [0, 1], keep as-is.
    if float(n.min()) < -0.05 or float(n.max()) > 1.05:
        n = (n + 1.0) / 2.0

    n = np.clip(n, 0.0, 1.0)
    return (n * 255.0).round().astype(np.uint8)


def _depth_normalize_01(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth must be 2D, got shape {d.shape}")

    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    valid = d > 0
    if not np.any(valid):
        return np.zeros_like(d, dtype=np.float32), valid

    vals = d[valid]
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if hi <= lo:
        lo = float(vals.min())
        hi = float(vals.max())
    if hi <= lo:
        out = np.zeros_like(d, dtype=np.float32)
        out[valid] = 1.0
        return out, valid

    scaled = (d - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[~valid] = 0.0
    return scaled.astype(np.float32), valid


def depth_to_colormap_uint8(depth: np.ndarray) -> np.ndarray:
    """Convert depth to RGB pseudo-color image (jet-like colormap)."""
    t, valid = _depth_normalize_01(depth)

    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    rgb[~valid] = 0.0
    return (rgb * 255.0).round().astype(np.uint8)


def find_existing_file(base: Path, suffixes: list[str]) -> Path | None:
    for suffix in suffixes:
        p = Path(str(base) + suffix)
        if p.exists():
            return p
    return None


def export_frame(
    entry: FrameEntry,
    out_dir: Path,
    writer: ImageWriter,
) -> dict[str, bool]:
    stem = entry.base.name
    status = {
        "color": False,
        "mask": False,
        "normal": False,
        "depth": False,
    }

    out_color = out_dir / f"{stem}_color{entry.color_path.suffix.lower()}"
    shutil.copy2(entry.color_path, out_color)
    status["color"] = True

    mask_path = find_existing_file(entry.base, ["_mask.png", "_mask.jpg", "_mask.jpeg"])
    if mask_path is not None:
        out_mask = out_dir / f"{stem}_mask{mask_path.suffix.lower()}"
        shutil.copy2(mask_path, out_mask)
        status["mask"] = True
    else:
        print(f"[WARN] Missing mask for index {entry.idx:04d}")

    normal_path = Path(str(entry.base) + "_normal.npy")
    if normal_path.exists():
        normal = np.load(normal_path)
        normal_img = normal_to_uint8(normal)
        writer.save_png(out_dir / f"{stem}_normal.png", normal_img)
        status["normal"] = True
    else:
        print(f"[WARN] Missing normal for index {entry.idx:04d}")

    depth_path = Path(str(entry.base) + "_depth.npy")
    if depth_path.exists():
        depth = np.load(depth_path)
        depth_img = depth_to_colormap_uint8(depth)
        writer.save_png(out_dir / f"{stem}_depth.png", depth_img)
        status["depth"] = True
    else:
        print(f"[WARN] Missing depth for index {entry.idx:04d}")

    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export frame 0..n from a tactile dataset folder. "
            "Copy color/mask and convert normal/depth .npy to PNG "
            "(depth is exported as pseudo-color RGB)."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Input object folder, e.g. dataset/training_dataset/006_mustard_bottle_google_16k",
    )
    parser.add_argument(
        "--n",
        default=10,
        type=int,
        help="Last frame index to export (inclusive). Default: 10",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Output folder. Default: <input_dir>/Output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 0:
        raise ValueError("--n must be >= 0")

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dir does not exist: {input_dir}")

    data_dir = resolve_data_dir(input_dir)
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (input_dir / "Output")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = discover_entries(data_dir)
    if not entries:
        raise RuntimeError(f"No valid '*_xxxx_color.*' files found in: {data_dir}")

    writer = ImageWriter()
    print(f"[INFO] Input data dir: {data_dir}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Image backend: {writer.backend}")
    print(f"[INFO] Exporting indices: 0000..{args.n:04d}")

    exported = {"color": 0, "mask": 0, "normal": 0, "depth": 0}
    skipped_indices = []

    for idx in range(args.n + 1):
        if idx not in entries:
            skipped_indices.append(idx)
            print(f"[WARN] Missing color frame for index {idx:04d}, skipped")
            continue

        status = export_frame(entries[idx], out_dir, writer)
        for k in exported:
            exported[k] += int(status[k])

    print(
        "[DONE] Export summary: "
        f"color={exported['color']}, mask={exported['mask']}, "
        f"normal={exported['normal']}, depth={exported['depth']}"
    )
    if skipped_indices:
        skipped = ", ".join(f"{i:04d}" for i in skipped_indices)
        print(f"[DONE] Missing indices: {skipped}")


if __name__ == "__main__":
    main()
