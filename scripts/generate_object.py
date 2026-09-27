import os
import argparse
import sys

# 假设 sample_contacts 和 simulate_tactile 在同一目录下
# 或者你可以根据实际情况调整 import 路径
try:
    from sample_contacts import process_mesh_step0
    from simulate_tactile import process_mesh_step1
except ImportError:
    print("Error: Could not import pipeline steps. Please ensure 'sample_contacts.py' and 'simulate_tactile.py' are in the same directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate tactile data for a SINGLE mesh object.")
    
    # 必需参数
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the input .ply mesh file.")
    
    # 可选参数
    parser.add_argument("--output_root", type=str, default="./single_object_data", help="Root directory to save output data.")
    parser.add_argument("--mesh_name_override", type=str, default=None, help="Override the mesh name (default: extracted from filename).")
    parser.add_argument("--taxim_calib", type=str, default="../Taxim/calibs/gelsight_mini", help="Path to Taxim calibration folder.")
    parser.add_argument("--gelpad_path", type=str, default="../Taxim/calibs/gelsight_mini/gelmap.npy", help="Path to gelmap.npy.")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of point clouds during processing.")
    
    # 采样与仿真参数
    parser.add_argument("--points", type=int, default=100, help="Number of contact points to sample.")
    parser.add_argument("--size", type=float, default=0.2, help="Normalization size for the mesh.")
    parser.add_argument("--press_depth", type=float, default=3.0, help="Press depth in mm.")
    parser.add_argument("--shadow", action="store_true", help="Enable shadow in Taxim simulation (slower but more realistic).")

    args = parser.parse_args()

    # 1. 路径解析与检查
    if not os.path.exists(args.mesh_path):
        print(f"Error: Mesh file not found at {args.mesh_path}")
        sys.exit(1)

    mesh_filename = os.path.basename(args.mesh_path) # e.g., obj.ply
    mesh_name = os.path.splitext(mesh_filename)[0]   # e.g., obj
    
    # 如果提供了 mesh_name_override，使用它
    if args.mesh_name_override:
        mesh_name = args.mesh_name_override
    
    print(f"=== Processing Single Object: {mesh_name} ===")
    print(f"Input: {args.mesh_path}")
    print(f"Output: {args.output_root}")

    # 2. 运行 Step 0: 归一化 + 采样接触点 + 生成位姿
    print("\n--- Running Step 0: Sampling & Alignment ---")
    success = process_mesh_step0(
        mesh_name=mesh_name,
        mesh_input_path=args.mesh_path,
        output_root=args.output_root,
        points=args.points,
        visualize=args.visualize,
        size=args.size
    )

    if not success:
        print("Error: Step 0 failed.")
        sys.exit(1)

    # 3. 运行 Step 1: Taxim 仿真 + 生成数据
    print("\n--- Running Step 1: Taxim Simulation ---")
    # Step 1 会读取 Step 0 在 output_root 生成的 normalized_mesh 和 alignment
    process_mesh_step1(
        mesh_name=mesh_name,
        data_root=args.output_root,      # Step 0 的输出目录也是 Step 1 的输入目录
        dataset_output_dir=args.output_root, # 最终数据也存这 (会自动创建子文件夹)
        taxim_calib_dir=args.taxim_calib,
        gelpad_path=args.gelpad_path,
        press_depth=args.press_depth,
        shadow=args.shadow,
        save_accumulated_ply=True
    )

    print(f"\n=== Done! Data generated in {os.path.join(args.output_root, mesh_name)} ===")

if __name__ == "__main__":
    main()