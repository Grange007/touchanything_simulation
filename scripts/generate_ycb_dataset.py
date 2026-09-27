import os
import glob
import argparse
from sample_contacts import process_mesh_step0
from simulate_tactile import process_mesh_step1

def main():
    parser = argparse.ArgumentParser(description="Batch process YCB meshes for tactile dataset generation")
    parser.add_argument("--ycb_root", type=str, default="../data/ycb", help="Path to downloaded YCB data")
    parser.add_argument("--output_temp", type=str, default="../outputs/ycb_intermediate", help="Path for intermediate files (normalized mesh, frames)")
    parser.add_argument("--dataset_output", type=str, default="../data/training_dataset", help="Path for final training data")
    parser.add_argument("--folders", nargs="+", default=["google_16k", "google_64k"], help="Subfolders to scan")
    parser.add_argument("--taxim_calib", type=str, default="../Taxim/calibs/gelsight_mini")
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--size", type=float, default=0.2)
    
    args = parser.parse_args()

    # 1. 扫描所有 ply 文件
    mesh_files = []
    for folder in args.folders:
        search_path = os.path.join(args.ycb_root, folder, "*.ply")
        files = glob.glob(search_path)
        print(f"Found {len(files)} meshes in {folder}")
        mesh_files.extend(files)

    print(f"Total meshes to process: {len(mesh_files)}")
    if not mesh_files:
        raise FileNotFoundError(f"No PLY meshes found under {args.ycb_root}")
    
    # 2. 遍历处理
    for idx, mesh_path in enumerate(mesh_files):
        # mesh_path 示例: ./ycb/google_16k/002_master_chef_can.ply
        mesh_filename = os.path.basename(mesh_path) # 002_master_chef_can.ply
        mesh_name = os.path.splitext(mesh_filename)[0] # 002_master_chef_can
        
        # 为了区分不同分辨率的同名物体，可以在名字加后缀
        folder_name = os.path.basename(os.path.dirname(mesh_path)) # google_16k
        unique_name = f"{mesh_name}_{folder_name}"
        
        print(f"\n[{idx+1}/{len(mesh_files)}] Processing: {unique_name}...")
        
        # --- Step 0: Sampling ---
        # 采样点数可以根据物体大小调整，这里设为 200 (比原来的 100 多一点以获取更多数据)
        success = process_mesh_step0(
            mesh_name=unique_name,
            mesh_input_path=mesh_path,
            output_root=args.output_temp,
            points=args.points,
            size=args.size
        )
        
        if not success:
            print("Step 0 failed. Skipping.")
            continue
            
        # --- Step 1: Simulation ---
        process_mesh_step1(
            mesh_name=unique_name,
            data_root=args.output_temp,
            dataset_output_dir=args.dataset_output,
            taxim_calib_dir=args.taxim_calib,
            shadow=False # 根据需要开启阴影
        )

    print("\nAll processing complete!")
    print(f"Dataset saved to: {os.path.abspath(args.dataset_output)}")

if __name__ == "__main__":
    main()
