# TouchAnything Simulation Data Pipeline

Simulation data preparation for [TouchAnything: Diffusion-Guided 3D
Reconstruction from Sparse Robot Touches](https://grange007.github.io/touchanything/).

This repository generates GelSight-style tactile observations from object meshes,
trains or runs a multi-task U-Net for local geometry prediction, and exports
per-object data records for the
[TouchAnything reconstruction repository](https://github.com/Grange007/touchanything).
The diffusion-guided SDF and DMTet reconstruction stages run in that separate
repository.

Generated simulation datasets and YCB meshes are **not distributed here**.
Download YCB meshes and generate the training observations locally using the
steps below. The sensor calibration resources needed by the simulator are
included; they are distinct from the generated training dataset.

## Pipeline

```text
YCB download / user-provided PLY mesh
  -> center and scale mesh, remesh, sample contact poses
  -> Taxim: tactile RGB + depth/normal/mask ground truth
  -> filter contacts and order them using geodesic farthest-point sampling
  -> train a local geometry U-Net OR load released weights
  -> predict depth/normal/mask from simulated RGB observations
  -> reproject geometry and export images, arrays, and camera metadata
  -> TouchAnything: diffusion-guided object reconstruction
```

Training data and reconstruction data are different outputs: training uses
simulated RGB paired with geometry ground truth, while reconstruction uses the
exported per-object JSON and geometry arrays.

## Repository Layout

```text
scripts/
  download_ycb.py          # Download Google YCB meshes; retain PLY files only
  generate_ycb_dataset.py  # Generate a training dataset from downloaded meshes
  generate_object.py       # Simulate observations for one mesh
  sample_contacts.py       # Mesh preparation and contact-pose sampling
  simulate_tactile.py      # Taxim simulation, filtering, and contact ordering
  train_geometry_model.py  # Multi-task U-Net training
  predict_geometry.py      # RGB-to-local-geometry inference
  export_touchanything.py  # Export reconstruction input records
  export_modal_images.py   # Optional depth/normal visualization export
Taxim/                    # Required simulator subset and calibration
run_pipeline.sh           # One-mesh simulation, inference, and export
requirements.txt
environment.yml           # Conda base environment (install pip dependencies next)
NOTICE.md
```

`data/`, `outputs/`, and `checkpoints/` are created locally and ignored by Git.

## Installation

Use the `touchanything-sim` Conda environment with Python 3.9.25 and
PyTorch 2.4.1. Simulation and geometry export use CPU processing; model
training and inference can use CUDA.

### Create a Conda Environment

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate touchanything-sim
python -m pip install --upgrade pip
```

### Install PyTorch and Python Dependencies

With `touchanything-sim` activated, install the CUDA 12.4 build for NVIDIA GPUs:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

The GPU driver must support the chosen PyTorch CUDA runtime. A standalone CUDA
toolkit or CUDA extension compilation is not required by this simulation
repository. Check [PyTorch's version installation commands](https://pytorch.org/get-started/previous-versions/)
when selecting another CUDA build.

For CPU-only use, replace the first command with:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

`requirements.txt` pins the direct Python dependencies used by this pipeline.

| Dependency | Purpose |
| --- | --- |
| NumPy, SciPy | Geometry arrays, interpolation, filters, mesh-graph distances |
| Open3D, PyMeshLab | Mesh processing, contact sampling, raycasting, reprojection |
| OpenCV, Matplotlib | Image processing and visualization |
| PyTorch, torchvision | Geometry U-Net training, inference, and preprocessing |
| tqdm, W&B | Training progress and experiment logging |

YCB downloading uses only the Python standard library. ROS, PyBullet, and the
full `pybullet-object-models` package are not required.

### Linux System Libraries

Open3D, OpenCV, and PyMeshLab may need system libraries. On Ubuntu/Debian, if
imports report missing OpenGL or GLib libraries, install:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libegl1 libgomp1
```

### Check the Environment

```bash
python -c "import numpy, scipy, matplotlib, cv2, open3d, pymeshlab, torch, torchvision, tqdm, wandb; print('Dependencies OK'); print('CUDA:', torch.cuda.is_available())"
```

Run individual Python scripts from **`scripts/`**. Their relative defaults use
the sibling `Taxim/`, `data/`, `outputs/`, and `checkpoints/` directories.
Run `run_pipeline.sh` from the repository root; it resolves the mesh and weight
paths before changing directories.

## Pretrained Geometry Model

The geometry model predicts local depth, normals, and contact masks from tactile
RGB. It is separate from the diffusion model used by TouchAnything.

**Pretrained weights: [best_model.pth](https://huggingface.co/Grange007/touchanything-tactile-geometry-model/resolve/main/best_model.pth)**

Download the weights from the
[Hugging Face model repository](https://huggingface.co/Grange007/touchanything-tactile-geometry-model)
by running the following commands from this repository's root:

```bash
mkdir -p checkpoints
curl -fL --retry 3 \
  'https://huggingface.co/Grange007/touchanything-tactile-geometry-model/resolve/main/best_model.pth' \
  -o checkpoints/best_model.pth
```

The expected checkpoint is a PyTorch `state_dict` compatible with the bundled
`MultiTaskUNet`, approximately 126 MB. Downloading weights lets you skip training.

## Generate Training Data From YCB

### 1. Download Meshes

The downloader uses the [YCB dataset](https://www.ycbbenchmarks.com/) Google mesh
archives. Each archive is processed in a separate temporary directory, and only
`nontextured.ply` is retained. Images, textures, and archives are discarded.

From the repository root:

```bash
cd scripts
python download_ycb.py --output_dir ../data/ycb
```

By default this downloads all listed objects at both `google_16k` and
`google_64k` resolutions. Start with a smaller selection if preferred:

```bash
python download_ycb.py \
  --objects 002_master_chef_can 003_cracker_box \
  --resolutions google_16k --workers 2
```

Output layout:

```text
data/ycb/
  google_16k/002_master_chef_can.ply
  google_16k/003_cracker_box.ply
  google_64k/...
```

Existing meshes are skipped. Failed or unavailable archives are reported and
produce a nonzero exit status; rerunning retries missing meshes. The downloader
accepts `--base_url` for a compatible mirror if the YCB endpoint changes.

### 2. Generate Simulated Observations

Still in `scripts/`:

```bash
python generate_ycb_dataset.py \
  --ycb_root ../data/ycb \
  --folders google_16k google_64k \
  --output_temp ../outputs/ycb_intermediate \
  --dataset_output ../data/training_dataset \
  --points 200 --size 0.2
```

If only one resolution was downloaded, pass `--folders google_16k`.
Object names include their mesh resolution to avoid collisions.

```text
data/training_dataset/<object>_<resolution>/
  alignment/contact_frames.npy
  raw/
    <object>_<resolution>_0000_color.png
    <object>_<resolution>_0000_depth.npy
    <object>_<resolution>_0000_normal.npy
    <object>_<resolution>_0000_mask.png
    ...
```

`--points` controls attempted contact samples. Invalid contacts are removed;
the current simulator retains up to 300 valid contacts in geodesic FPS order.
`--size` is a multiplicative mesh scale after centering, not a target bounding-box
dimension. The default is 0.2; adjust it to your input mesh units and intended
object size.

Simulation uses the bundled 320 x 240 GelSight Mini calibration, a pixel spacing
of 0.0634 mm, and a default pressing depth of 3 mm. Intermediate meshes and poses
are written to `outputs/ycb_intermediate/`.

The calibration is located in `Taxim/calibs/gelsight_mini/` and contains
`sensorParams.py`, `dataPack.npz`, `polycalib.npz`, `shadowTable.npz`, and
`gelmap.npy`.

### 3. Train the Geometry Model (Optional)

```bash
WANDB_MODE=offline python train_geometry_model.py \
  --data_root ../data/training_dataset \
  --checkpoint_dir ../checkpoints \
  --epochs 50 --batch_size 32 --lr 0.001
```

The training script recursively pairs `_color.png` with `_depth.npy`,
`_normal.npy`, and `_mask.png`, then makes a seeded 90/10 frame-level
training/validation split. This is not an object-disjoint evaluation split.
It optimizes depth L1, normal cosine, and mask binary cross-entropy losses.
Reduce the batch size if GPU memory is insufficient.

The best checkpoint is saved to `checkpoints/best_model.pth`; periodic epoch
checkpoints are also written. `WANDB_MODE=offline` avoids requiring a W&B login.
Remove that setting and configure W&B if online experiment logging is desired.

## Export One Object for TouchAnything

From the repository root, with downloaded or locally trained weights:

```bash
bash run_pipeline.sh data/ycb/google_16k/002_master_chef_can.ply
```

An alternative checkpoint can be supplied as the second argument. The wrapper
attempts 350 contacts, uses mesh scale 0.2, and stops on command failures or
missing expected outputs. Use a new output directory or remove an object's old
generated output before rerunning with changed parameters to avoid stale frames.

Intermediate observations and predictions:

```text
outputs/objects/<object>/
  normalized_mesh/
  alignment/contact_frames.npy
  raw/                            # Simulated RGB and ground truth
  reconstruction/                 # U-Net predictions
```

Final reconstruction input:

```text
outputs/touchanything/<object>/
  meta_data.json
  <object>_10.json
  <object>_20.json
  <object>_40.json
  <object>_100.json
  <object>_300.json
  000000_rgb.png
  000000_depth.npy
  000000_normal.npy
  000000_foreground_mask.png
  ...
```

Subset JSON files are created only when enough frames are available. Each
references the first N FPS-ordered contacts; the full `meta_data.json` references
all exported frames. Paths in JSON are relative to the object directory.
The metadata includes the `OPENCV` camera model, 4 x 4 intrinsics, 4 x 4
camera-to-world transforms, depth/normal/mask paths, and scene bounds.

The exporter uses black RGB placeholder images: tactile RGB is used for geometry
prediction, but is not an object appearance image. The exported normals are
recomputed from reprojected depth. The exporter's `--size` defaults to 9 and is
a uniform scale factor applied to geometry and camera translations; it is
different from the mesh-preparation scale. Exported depths are in scaled scene
units. Keep depth and camera translation scales consistent.

For independent stages or custom settings, run these commands from `scripts/`:

```bash
python generate_object.py --mesh_path ../data/ycb/google_16k/002_master_chef_can.ply \
  --output_root ../outputs/objects --points 350 --size 0.2
python predict_geometry.py --data_root ../outputs/objects \
  --mesh_name 002_master_chef_can --model_path ../checkpoints/best_model.pth
python export_touchanything.py --data_root ../outputs/objects \
  --mesh_name 002_master_chef_can --output_dir ../outputs/touchanything --size 9
```

Add `--save_pcd` to export debug point clouds. These PLY files are not required
by the reconstruction pipeline. Run any script with `--help` for its parameters.

## Run TouchAnything Reconstruction

Install the reconstruction repository separately, then run from its root:

```bash
bash scripts/reconstruct_object.sh \
  --data-root /absolute/path/to/touchanything-simulation/outputs/touchanything/002_master_chef_can \
  --json 002_master_chef_can_20.json \
  --prompt "a can"
```

Use `meta_data.json` for all contacts or another available subset JSON. See
TouchAnything's [data format documentation](https://github.com/Grange007/touchanything/blob/main/docs/data_format.md)
and reconstruction configuration for the required fields and coordinate conventions.

## Attribution

Taxim source and calibration dependencies are bundled with its MIT license;
see `Taxim/LICENSE` and `NOTICE.md`. Cite Taxim when using its simulator:
[Taxim: An Example-based Simulation Model for GelSight Tactile Sensors](https://arxiv.org/abs/2109.04027).
Follow the YCB dataset's attribution and redistribution terms for downloaded meshes.

Generated datasets, pretrained weights, downloaded meshes, and execution outputs
are excluded from version control. Pretrained weights are available through the
Hugging Face download link above.

## Acknowledgments

We thank the authors and contributors of
[gs_sdk](https://github.com/joehjhuang/gs_sdk) and
[Taxim](https://github.com/CMURoboTouch/Taxim) for their open-source contributions
to tactile sensing and simulation.
