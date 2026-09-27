import os
import numpy as np
import open3d as o3d
import argparse
import sys
import copy
import pymeshlab

def clean_mesh_simple(mesh):
    print(f"原始点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
    
    # 1. 去除重复的三角形（完全一样的顶点索引）
    mesh.remove_duplicated_triangles()
    
    # 2. 去除重复的顶点（坐标完全一样的点）
    mesh.remove_duplicated_vertices()
    
    # 3. 去除退化三角形（面积为0的）
    mesh.remove_degenerate_triangles()
    
    print(f"清洗后点数: {len(mesh.vertices)}, 面数: {len(mesh.triangles)}")
    return mesh

def remesh_using_pymeshlab(input_path, output_path, target_percent=1.0):
    """
    使用 PyMeshLab 进行各向同性重网格化 (Isotropic Remeshing)
    target_percent: 目标边长相对于包围盒对角线的百分比 (1.0 约等于 1%)
    """
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_path)
        
        print(f"[Remesh] Running Isotropic Remeshing (Target Len: {target_percent}%)...")
        # 核心算法：重新布线，使其均匀
        ms.meshing_isotropic_explicit_remeshing(
            iterations=3, 
            targetlen=pymeshlab.PercentageValue(target_percent)
        )
        
        ms.save_current_mesh(output_path)
        return True
    except Exception as e:
        print(f"[Remesh] Error: {e}")
        return False

def process_mesh_step0(mesh_name, mesh_input_path, output_root, points=200, size=0.2, visualize=False):
    """
    Step 0: 归一化 Mesh -> [新增: 重新布线] -> 采样接触点
    """
    
    # 路径设置
    norm_mesh_dir = os.path.join(output_root, mesh_name, "normalized_mesh")
    align_dir = os.path.join(output_root, mesh_name, "alignment")
    os.makedirs(norm_mesh_dir, exist_ok=True)
    os.makedirs(align_dir, exist_ok=True)

    # Load Mesh
    if not os.path.exists(mesh_input_path):
        print(f"Error: Mesh not found at {mesh_input_path}")
        return False

    mesh = o3d.io.read_triangle_mesh(mesh_input_path)
    mesh = clean_mesh_simple(mesh)

    # Normalize Mesh
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    mesh.translate(-center)
    mesh.scale(size, center=np.array([0, 0, 0]))

    # Save Normalized Mesh (Temporary or Final)
    # 我们先保存这个归一化的版本，给 PyMeshLab 读取
    norm_mesh_path = os.path.join(norm_mesh_dir, f"{mesh_name}_normalized.ply")
    o3d.io.write_triangle_mesh(norm_mesh_path, mesh)

    # # ==========================================
    # # [新增] 核心修改：在此处插入 Remeshing
    # # ==========================================
    remeshed_path = os.path.join(norm_mesh_dir, f"{mesh_name}_remeshed.ply")
    
    # 执行重网格化
    # target_percent=1.0 表示边长约为物体尺寸的 1%，这对于后续采样非常完美
    success = remesh_using_pymeshlab(norm_mesh_path, remeshed_path, target_percent=1.0)
    
    if success:
        print(f"[{mesh_name}] Reloading remeshed model...")
        # 重新读取优化后的 Mesh 覆盖变量
        mesh = o3d.io.read_triangle_mesh(remeshed_path)
    else:
        print(f"[{mesh_name}] Remeshing failed, using original normalized mesh.")

    # # ==========================================
    # # 后续逻辑保持不变 (但现在 mesh 是拓扑完美的了)
    # # ==========================================

    # --- 法线与 KD-Tree 准备 ---
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    mesh.compute_vertex_normals() 

    tri_normals = np.asarray(mesh.triangle_normals)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    tri_centers = vertices[triangles].mean(axis=1)
    
    pcd_centers = o3d.geometry.PointCloud()
    pcd_centers.points = o3d.utility.Vector3dVector(tri_centers)
    kdtree = o3d.geometry.KDTreeFlann(pcd_centers)
    
    # --- Sample Points ---
    print(f"[{mesh_name}] Sampling {points} points (Poisson Disk)...")
    
    # 因为现在 Mesh 质量很高，Poisson Disk Sampling 会工作得非常完美
    # init_factor 可以稍微降低一点提高速度，或者保持为 5-10
    pcd = mesh.sample_points_poisson_disk(number_of_points=points, init_factor=5)
    
    uniform_points = np.asarray(pcd.points)
    uniform_normals = np.asarray(pcd.normals) 
    
    # 法线兜底逻辑
    if len(uniform_points) > 0 and (len(uniform_normals) == 0):
        print(f"[{mesh_name}] Warning: Recomputing sampled normals...")
        recomputed_normals = []
        for pt in uniform_points:
            _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
            recomputed_normals.append(tri_normals[idx[0]])
        uniform_normals = np.array(recomputed_normals)

    # 补点逻辑 (现在应该很少会触发这个了，除非 points 数量极大)
    if len(uniform_points) < points:
        print(f"[{mesh_name}] Poisson disk missed {points - len(uniform_points)} points, filling uniformly.")
        remaining = points - len(uniform_points)
        pcd_extra = mesh.sample_points_uniformly(number_of_points=remaining)
        extra_points = np.asarray(pcd_extra.points)
        extra_normals = np.asarray(pcd_extra.normals)
        if len(extra_points) > 0:
            if len(extra_normals) == 0:
                recomputed_extra = []
                for pt in extra_points:
                    _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
                    recomputed_extra.append(tri_normals[idx[0]])
                extra_normals = np.array(recomputed_extra)
            uniform_points = np.vstack([uniform_points, extra_points])
            uniform_normals = np.vstack([uniform_normals, extra_normals])

    if visualize:
        o3d.io.write_point_cloud(os.path.join(align_dir, "sampled_points.ply"), pcd)
    print(f"[{mesh_name}] Sampled {len(uniform_points)} points.")

    # Setup Raycasting
    scene = o3d.t.geometry.RaycastingScene()
    # 注意：这里使用的是新的 remeshed mesh
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    # Pre-compute sensor geometry
    imgh, imgw = 240, 320
    ppmm = 0.0634
    x = (np.arange(imgw) - imgw / 2 + 0.5) * ppmm / 1000.0
    y = (np.arange(imgh) - imgh / 2 + 0.5) * ppmm / 1000.0
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros_like(xv)
    origins_sensor_template = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)

    # 可视化用的降采样模板
    vis_step = 10 
    x_vis = x[::vis_step]
    y_vis = y[::vis_step]
    xv_vis, yv_vis = np.meshgrid(x_vis, y_vis)
    zv_vis = np.zeros_like(xv_vis)
    sensor_template_vis = np.stack([xv_vis, yv_vis, zv_vis], axis=-1).reshape(-1, 3)

    def build_rotation_matrix(z_axis):
        z_axis = z_axis / np.linalg.norm(z_axis)
        if np.abs(z_axis[2]) < 0.99:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        return np.column_stack([x_axis, y_axis, z_axis])

    # --- 核心循环 ---
    frames = []
    
    vis_data_stage1 = [] 
    vis_hits_stage1 = [] 
    vis_data_final  = [] 
    vis_sensor_planes = [] 
    contact_points_all = []
    probe_distance = 0.05 

    print(f"uniform_points shape: {uniform_points.shape}, uniform_normals shape: {uniform_normals.shape}")
    for i in range(len(uniform_points)):
        pt = uniform_points[i]
        if i >= len(uniform_normals): break
        nm = uniform_normals[i]
        
        nm_norm = np.linalg.norm(nm)
        if nm_norm < 1e-6: continue
        nm = nm / nm_norm

        # === Stage 1: Probe ===
        sensor_pos_probe = pt + nm * probe_distance
        look_at_dir_probe = -nm 
        
        R_probe = build_rotation_matrix(look_at_dir_probe)
        t_probe = sensor_pos_probe
        
        origins_world = (R_probe @ origins_sensor_template.T).T + t_probe
        directions_world = np.tile(look_at_dir_probe, (len(origins_world), 1)).astype(np.float32)
        
        rays = np.concatenate([origins_world, directions_world], axis=-1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = ans["t_hit"].numpy()
        
        valid_mask = np.isfinite(t_hit)
        if not np.any(valid_mask): continue

        min_dist = np.min(t_hit[valid_mask])
        
        hit_idx = np.argmin(np.abs(t_hit - min_dist))
        real_contact_on_mesh = origins_world[hit_idx] + directions_world[hit_idx] * min_dist

        # === 构造最终 Frame ===
        R_final = R_probe 
        final_sensor_pos = t_probe + look_at_dir_probe * min_dist
        
        world_T_final = np.eye(4, dtype=np.float32)
        world_T_final[:3, :3] = R_final
        world_T_final[:3, 3] = final_sensor_pos
        
        contact_points_all.append(real_contact_on_mesh)
        frames.append(world_T_final)

        current_sensor_vis_points = (R_final @ sensor_template_vis.T).T + final_sensor_pos
        vis_sensor_planes.append(current_sensor_vis_points)

        vis_data_stage1.append((sensor_pos_probe, look_at_dir_probe))
        vis_hits_stage1.append(real_contact_on_mesh)
        vis_data_final.append((final_sensor_pos, look_at_dir_probe))

    # Save Results
    np.save(os.path.join(align_dir, "contact_frames.npy"), frames)
    print(f"[{mesh_name}] Saved {len(frames)} frames.")
    
    if visualize:
        pcd_contact = o3d.geometry.PointCloud()
        if len(contact_points_all) > 0:
            pcd_contact.points = o3d.utility.Vector3dVector(np.array(contact_points_all))
            pcd_contact.paint_uniform_color([1, 0, 0]) 
            o3d.io.write_point_cloud(os.path.join(align_dir, f"{mesh_name}_contact_step0.ply"), pcd_contact)

        if len(vis_sensor_planes) > 0:
            all_sensor_points = np.vstack(vis_sensor_planes)
            pcd_sensors = o3d.geometry.PointCloud()
            pcd_sensors.points = o3d.utility.Vector3dVector(all_sensor_points)
            pcd_sensors.paint_uniform_color([0, 1, 0]) 
            o3d.io.write_point_cloud(os.path.join(align_dir, f"{mesh_name}_sensor_planes_vis.ply"), pcd_sensors)

    # Debug func
    def save_debug_all(vis_data, hit_points, filename_base, color_origin, color_hit):
        if len(vis_data) == 0: return
        points_origin = np.array([d[0] for d in vis_data])
        normals_origin = np.array([d[1] for d in vis_data])
        print(f"points_origin shape: {points_origin.shape}, normals_origin shape: {normals_origin.shape}")
        if hit_points is not None and len(hit_points) > 0:
            pcd_hit = o3d.geometry.PointCloud()
            pcd_hit.points = o3d.utility.Vector3dVector(np.array(hit_points))
            if color_hit: pcd_hit.paint_uniform_color(color_hit)
            o3d.io.write_point_cloud(filename_base + "_hits.ply", pcd_hit)
        scale = 0.02 
        line_points, lines, colors = [], [], []
        for idx, (p, n) in enumerate(zip(points_origin, normals_origin)):
            end_p = p + n * scale
            line_points.extend([p, end_p]) 
            lines.append([2*idx, 2*idx+1])
            colors.append(color_origin)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_line_set(filename_base + "_vectors.ply", line_set)
        pcd_origin = o3d.geometry.PointCloud()
        pcd_origin.points = o3d.utility.Vector3dVector(points_origin)
        pcd_origin.paint_uniform_color(color_origin)
        o3d.io.write_point_cloud(filename_base + "_origins.ply", pcd_origin)

    if visualize:
        save_debug_all(vis_data_stage1, vis_hits_stage1, os.path.join(align_dir, f"{mesh_name}_debug_probe"), [0, 0, 1], [0, 1, 1])
        save_debug_all(vis_data_final, None, os.path.join(align_dir, f"{mesh_name}_debug_final"), [1, 0, 0], None)
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default=".")
    parser.add_argument("-v", "--visualize", action="store_true")
    args = parser.parse_args()
    process_mesh_step0(mesh_name=args.name, mesh_input_path=args.input, output_root=args.output, visualize=args.visualize)