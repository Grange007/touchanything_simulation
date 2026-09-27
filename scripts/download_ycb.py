"""Download YCB Google mesh archives and retain only nontextured PLY meshes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

BASE_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"


def download_mesh(task):
    object_name, resolution, output_root, base_url = task
    destination = output_root / resolution / f"{object_name}.ply"
    if destination.is_file():
        print(f"Already exists: {destination}")
        return True
    url = f"{base_url}google/{object_name}_{resolution}.tgz"
    try:
        # Each task owns its archive; no shared extracted object directories.
        with tempfile.TemporaryDirectory(prefix="ycb-") as temporary:
            archive = Path(temporary) / "mesh.tgz"
            with urllib.request.urlopen(url, timeout=120) as response:
                with archive.open("wb") as target:
                    shutil.copyfileobj(response, target)
            with tarfile.open(archive, "r:gz") as bundle:
                candidates = [
                    member for member in bundle.getmembers()
                    if member.isfile()
                    and Path(member.name).name == "nontextured.ply"
                    and resolution in Path(member.name).parts
                ]
                if len(candidates) != 1:
                    raise ValueError(f"Expected one {resolution}/nontextured.ply")
                destination.parent.mkdir(parents=True, exist_ok=True)
                staged = Path(temporary) / "mesh.ply"
                with bundle.extractfile(candidates[0]) as source:
                    with staged.open("wb") as target:
                        shutil.copyfileobj(source, target)
                shutil.move(str(staged), str(destination))
        print(f"Downloaded: {destination}")
        return True
    except Exception as error:
        print(f"Failed: {object_name} ({resolution}): {error}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("../data/ycb"))
    parser.add_argument("--objects", nargs="+", help="YCB object names; omitted means all objects")
    parser.add_argument("--resolutions", nargs="+", choices=["google_16k", "google_64k"],
                        default=["google_16k", "google_64k"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base_url", default=BASE_URL, help="YCB data endpoint or compatible mirror")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    base_url = args.base_url.rstrip("/") + "/"
    objects = args.objects
    if objects is None:
        with urllib.request.urlopen(base_url + "objects.json", timeout=120) as response:
            objects = json.load(response)["objects"]
    for name in objects:
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            parser.error(f"Invalid object name: {name!r}")
    tasks = [(name, resolution, args.output_dir, base_url)
             for name in dict.fromkeys(objects) for resolution in dict.fromkeys(args.resolutions)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(download_mesh, tasks))
    print(f"Completed: {sum(results)}/{len(results)} meshes")
    if not tasks or not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
