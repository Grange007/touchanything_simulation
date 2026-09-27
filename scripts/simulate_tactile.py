import os
import numpy as np
import cv2
import open3d as o3d
import sys
import argparse
import copy
import scipy.sparse
from scipy.sparse.csgraph import dijkstra

# --- 配置 Taxim 路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
taxim_path = os.path.join(current_dir, "../Taxim") 
if taxim_path not in sys.path:
    sys.path.append(taxim_path)

try:
    from OpticalSimulation.tactile_simulator import simulator
except ImportError:
    sys.path.append("../Taxim") 
    try:
        from OpticalSimulation.tactile_simulator import simulator
    except:
        print("Warning: Could not import Taxim simulator.")

# --- 核心算法：基于 Mesh 测地线的 FPS (保持不变) ---

def build_mesh_graph(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    diff = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    weights = np.linalg.norm(diff, axis=1)
    n_vertices = len(vertices)
    adj = scipy.sparse.coo_matrix(
        (weights, (edges[:, 0], edges[:, 1])),
        shape=(n_vertices, n_vertices)
    )
    adj = adj + adj.T
    return adj

def geodesic_farthest_point_sample(mesh, points, npoint):
    N = points.shape[0]
    if N < npoint:
        print(f"Warning: Not enough points for FPS ({N} < {npoint}), returning all.")
        return np.arange(N)

    # 1. Snap
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    vertex_indices = []
    for pt in points:
        _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
        vertex_indices.append(idx[0])
    vertex_indices = np.array(vertex_indices)
    
    # 2. Dijkstra
    graph = build_mesh_graph(mesh)
    print("   Calculating geodesic distances (Dijkstra)...")
    full_dist_matrix = dijkstra(graph, directed=False, indices=vertex_indices)
    geodesic_dist = full_dist_matrix[:, vertex_indices]
    
    if np.isinf(geodesic_dist).any():
        max_valid = np.nanmax(geodesic_dist[np.isfinite(geodesic_dist)])
        geodesic_dist[np.isinf(geodesic_dist)] = max_valid * 100.0

    # 3. FPS
    centroids = np.zeros((npoint,), dtype=int)
    min_distances = np.ones((N,)) * 1e20 
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        dist_to_new_centroid = geodesic_dist[farthest, :]
        mask = dist_to_new_centroid < min_distances
        min_distances[mask] = dist_to_new_centroid[mask]
        farthest = np.argmax(min_distances)
        
    return centroids

def create_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def process_mesh_step1(mesh_name, data_root, dataset_output_dir, 
                        taxim_calib_dir="../Taxim/calibs/gelsight_mini",
                        gelpad_path="../Taxim/calibs/gelsight_mini/gelmap.npy",
                        press_depth=3.0, shadow=False, 
                        save_accumulated_ply=True,
                        visualize=False): 
    """
    修改版: 
    1. 仿真所有点 -> 过滤
    2. Geodesic FPS 排序
    3. 按 FPS 顺序保存文件 (0000对应FPS第1个点)
    """
    
    # 路径准备
    norm_mesh_path = os.path.join(data_root, mesh_name, "normalized_mesh", f"{mesh_name}_normalized.ply")
    frames_path = os.path.join(data_root, mesh_name,"alignment", "contact_frames.npy")
    
    if not os.path.exists(norm_mesh_path) or not os.path.exists(frames_path):
        raise FileNotFoundError(f"[{mesh_name}] Missing normalized mesh or contact frames")

    # 输出目录
    raw_out_dir = os.path.join(dataset_output_dir, mesh_name, "raw")
    refined_out_dir = os.path.join(dataset_output_dir, mesh_name, "refined_poses")
    align_out_dir = os.path.join(dataset_output_dir, mesh_name, "alignment") # 用于保存最终排序后的 frames
    
    os.makedirs(raw_out_dir, exist_ok=True)
    os.makedirs(refined_out_dir, exist_ok=True)
    os.makedirs(align_out_dir, exist_ok=True)

    # Load Mesh & Scene
    print(f"[{mesh_name}] Loading mesh and frames...")
    mesh = o3d.io.read_triangle_mesh(norm_mesh_path)
    # o3d.io.write_triangle_mesh(os.path.join(refined_out_dir, f"{mesh_name}_normalized.ply"), mesh) 
    
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
        
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    contact_frames = np.load(frames_path, allow_pickle=True)
    
    # Init Simulator
    try:
        taxim_sim = simulator(taxim_calib_dir, data_root)
    except Exception as e:
        raise RuntimeError("Could not initialize Taxim") from e
    
    # Pre-compute Geometry
    imgh, imgw = 240, 320
    ppmm = 0.0634
    x = (np.arange(imgw) - imgw / 2 + 0.5) * ppmm / 1000.0
    y = (np.arange(imgh) - imgh / 2 + 0.5) * ppmm / 1000.0
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros_like(xv)
    origins_sensor_template = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
    directions_sensor_template = np.tile(np.array([[0, 0, 1]]), (imgh * imgw, 1))

    # Filtering Thresholds
    MIN_CONTACT_AREA_RATIO = 0.1
    MIN_AVG_GRADIENT = 0.05
    MIN_GRAD_STD = 0.02
    
    accumulated_points_list = []
    
    # 【改动 1】创建一个 Buffer 列表，用来暂存通过 Filter 的所有数据
    # 不再直接写入文件，而是存入内存，等待 FPS 排序
    valid_data_buffer = [] 
    
    print(f"[{mesh_name}] Simulating {len(contact_frames)} frames...")
    
    # --- SIMULATION LOOP ---
    for i, contact_frame in enumerate(contact_frames):
        # 1. Raycasting
        world_T_ref = np.eye(4, dtype=np.float32)
        world_T_ref[:3, :3] = contact_frame[:3, :3] 
        world_T_ref[:3, 3] = contact_frame[:3, 3]
        
        R_ref = world_T_ref[:3, :3]
        t_ref = world_T_ref[:3, 3]
        
        origins_world = (R_ref @ origins_sensor_template.T).T + t_ref
        directions_world = (R_ref @ directions_sensor_template.T).T
        
        rays = np.concatenate([origins_world, directions_world], axis=-1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = ans["t_hit"].numpy()
        hit = np.isfinite(t_hit)
        
        if np.sum(hit) < 10: continue

        points_world_hit = origins_world[hit] + t_hit[hit][:, None] * directions_world[hit]
        points_sensor_z = (R_ref.T @ (points_world_hit - t_ref).T).T[:, 2]
        
        true_H_raw = np.zeros(imgh * imgw, dtype=np.float32)
        true_H_raw[hit] = points_sensor_z
        true_H_raw = true_H_raw.reshape((imgh, imgw)) * 1000 / ppmm 

        # 2. Taxim Simulation
        try:
            height_taxim_raw, gel_map, contact_mask_raw = taxim_sim.generateHeightMap_knownHmap(
                gelpad_model_path=gelpad_path, 
                pressing_height_mm=press_depth, 
                heightMap_input=true_H_raw, 
                dx=0, dy=0
            )
            true_H, contact_mask, contact_height = taxim_sim.deformApprox(
                pressing_height_mm=press_depth,
                height_map=height_taxim_raw,
                gel_map=gel_map,
                contact_mask=contact_mask_raw
            )
        except Exception as e:
            continue
        
        # 3. Filters
        contact_ratio = np.sum(contact_mask) / (imgh * imgw)
        if contact_ratio < MIN_CONTACT_AREA_RATIO: continue

        true_N = taxim_sim.generate_normals_direct(true_H)
        
        if contact_ratio > 0.8:
            nz = true_N[:, :, 2:]
            nz[np.abs(nz) < 1e-6] = 1e-6
            gxy = -true_N[:, :, :2] / nz
            gx, gy = gxy[:, :, 0], gxy[:, :, 1]
            mask_bool = contact_mask.astype(bool)
            valid_gx, valid_gy = gx[mask_bool], gy[mask_bool]
            
            if valid_gx.size > 0:
                grad_mag = np.sqrt(valid_gx**2 + valid_gy**2)
                avg_grad = np.mean(grad_mag)
                grad_std = np.sqrt(np.std(valid_gx)**2 + np.std(valid_gy)**2)
                if avg_grad < MIN_AVG_GRADIENT or grad_std < MIN_GRAD_STD:
                    continue

        # 4. Generate Image (but don't save yet)
        sim_img, shadow_sim_img = taxim_sim.simulating(true_H, contact_mask, contact_height, shadow=shadow)
        tactile_img = shadow_sim_img if shadow else sim_img
        
        # 存入 Buffer
        valid_data_buffer.append({
            "image": tactile_img,
            "depth": true_H,
            "normal": true_N,
            "mask": contact_mask,
            "frame": contact_frame,
            "original_index": i
        })

        # 累积点云逻辑保持不变 (用于可视化 Ground Truth)
        if save_accumulated_ply:
            mask_flat = contact_mask.flatten() > 0
            valid_idx = np.logical_and(hit, mask_flat)
            if np.sum(valid_idx) > 0:
                p_origins = origins_world[valid_idx]
                p_dirs = directions_world[valid_idx]
                p_t = t_hit[valid_idx]
                p_contact_world = p_origins + p_t[:, None] * p_dirs
                n_points = p_contact_world.shape[0]
                if n_points > 500:
                    choice_indices = np.random.choice(n_points, size=int(n_points * 0.2), replace=False)
                    p_contact_world = p_contact_world[choice_indices]
                accumulated_points_list.append(p_contact_world)

    # === Filtering Done ===
    num_valid = len(valid_data_buffer)
    print(f"[{mesh_name}] Filtering Done. {num_valid} valid frames collected.")

    if num_valid == 0:
        raise RuntimeError(f"[{mesh_name}] No valid tactile contacts generated")

    # --- Post-processing: Geodesic FPS Resampling ---
    # 提取所有有效帧的位置用于 FPS
    valid_frames_arr = np.array([d["frame"] for d in valid_data_buffer])
    valid_positions = valid_frames_arr[:, :3, 3]

    print(f"[{mesh_name}] Performing Geodesic FPS on {num_valid} valid frames...")
    
    # 1. 先对【所有】有效点进行 FPS 排序，确保采样的全局均匀性
    fps_sort_idx = geodesic_farthest_point_sample(mesh, valid_positions, num_valid)
    
    # 2. 确定最大保存数量 (max of subsets)
    subsets = [10, 20, 40, 100, 300]
    max_save_count = max(subsets)

    # 3. 截断索引列表：只保留前 max_save_count 个
    # 注意：如果 num_valid < 100，切片操作是安全的，会保留所有点
    if num_valid > max_save_count:
        print(f"[{mesh_name}] Truncating output: Saving only top {max_save_count} frames (from {num_valid} valid).")
        fps_sort_idx = fps_sort_idx[:max_save_count]
    else:
        print(f"[{mesh_name}] Valid count ({num_valid}) < Target ({max_save_count}). Saving all available.")

    # --- Saving Loop ---
    sorted_frames_final = []
    
    print(f"[{mesh_name}] Saving sorted data to disk...")
    # 这里循环次数已经被限制为最多 100 次
    for rank, idx in enumerate(fps_sort_idx):
        data = valid_data_buffer[idx]
        
        base_name = f"{mesh_name}_{rank:04d}" # 0000, 0001 ... 0099
        
        # Save Image
        cv2.imwrite(os.path.join(raw_out_dir, f"{base_name}_color.png"), data["image"])
        
        # Save Depth
        np.save(os.path.join(raw_out_dir, f"{base_name}_depth.npy"), data["depth"])
        
        # Save Normal
        np.save(os.path.join(raw_out_dir, f"{base_name}_normal.npy"), data["normal"])
        
        # Save Mask
        mask_to_save = data["mask"]
        mask_save = (mask_to_save * 255).astype(np.uint8) if np.max(mask_to_save) <= 1 else mask_to_save.astype(np.uint8)
        cv2.imwrite(os.path.join(raw_out_dir, f"{base_name}_mask.png"), mask_save)
        
        sorted_frames_final.append(data["frame"])

    sorted_frames_final = np.array(sorted_frames_final)

    # --- 保存对齐用的 Frames ---
    # 这个 npy 文件现在也只包含最多 100 帧，与 raw/ 文件夹下的图片一一对应
    np.save(os.path.join(align_out_dir, "contact_frames.npy"), sorted_frames_final)
    print(f"[{mesh_name}] Top {len(sorted_frames_final)} sorted frames saved to alignment/contact_frames.npy")

    # --- 保存分层结果 (NPY + PLY) ---
    # valid_poses_all.npy 现在也只存前 100 个
    # np.save(os.path.join(refined_out_dir, "valid_poses_all.npy"), sorted_frames_final)
    
    # pcd_all = create_point_cloud(sorted_frames_final[:, :3, 3])
    # pcd_all.paint_uniform_color([0, 0, 1])
    # o3d.io.write_point_cloud(os.path.join(refined_out_dir, "valid_subset_all.ply"), pcd_all)

    # 子集保存逻辑保持不变，因为 subsets 都在 max_save_count 范围内
    for k in subsets:
        if k > len(sorted_frames_final):
            k_eff = len(sorted_frames_final)
        else:
            k_eff = k
            
        subset_frames = sorted_frames_final[:k_eff]
        
        # NPY
        # np.save(os.path.join(refined_out_dir, f"valid_poses_{k:03d}.npy"), subset_frames)
        
        # PLY
        if visualize:
            subset_positions = subset_frames[:, :3, 3]
            pcd_subset = create_point_cloud(subset_positions)
            pcd_subset.paint_uniform_color([1, 0, 0])
            o3d.io.write_point_cloud(os.path.join(refined_out_dir, f"valid_subset_{k_eff:03d}.ply"), pcd_subset)
        
        
            # --- 保存累积点云 ---
            if save_accumulated_ply and len(accumulated_points_list) > 0:
                all_points = np.concatenate(accumulated_points_list, axis=0)
                pcd = create_point_cloud(all_points)
                pcd = pcd.voxel_down_sample(voxel_size=0.001)
                pcd.paint_uniform_color([0.7, 0.7, 0.7]) 
                
                ply_save_path = os.path.join(raw_out_dir, f"{mesh_name}_accumulated_gt.ply")
                o3d.io.write_point_cloud(ply_save_path, pcd)
                print(f"[{mesh_name}] Accumulation visualization saved.")

    print(f"[{mesh_name}] Step 1 Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-r", "--root", type=str, required=True, help="Data root")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    process_mesh_step1(args.name, args.root, args.output)
