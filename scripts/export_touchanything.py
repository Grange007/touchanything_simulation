import os
import glob
import argparse
import json
import numpy as np
import cv2
import open3d as o3d
from scipy.interpolate import griddata
import time

import open3d as o3d

# ==========================================
# 辅助类：网格渲染器
# ==========================================
class MeshRenderer:
    def __init__(self, H, W, ppmm):
        self.H = H
        self.W = W
        self.ppmm = ppmm
        
        # 1. 预计算网格坐标 (用于生成顶点)
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        self.grid_x = xx.flatten() # (N,)
        self.grid_y = yy.flatten() # (N,)
        
        # 2. 预计算基础三角形索引 (Full Grid)
        # 这是一个标准的 Grid Triangulation
        idx_map = np.arange(H * W).reshape(H, W)
        
        # 定义每个像素格子的两个三角形
        # T1: (r, c), (r, c+1), (r+1, c)
        v1 = idx_map[:-1, :-1].flatten()
        v2 = idx_map[:-1, 1:].flatten()
        v3 = idx_map[1:, :-1].flatten()
        self.t1_base = np.stack([v1, v2, v3], axis=1) # (M, 3)
        
        # T2: (r+1, c), (r, c+1), (r+1, c+1)
        v4 = idx_map[1:, 1:].flatten()
        self.t2_base = np.stack([v3, v2, v4], axis=1) # (M, 3)
        
        # 拼合所有可能的三角形索引 (作为模板)
        self.all_triangles = np.concatenate([self.t1_base, self.t2_base], axis=0).astype(np.int32)
        
        # 初始化 Open3D 设备 (优先尝试 CUDA)
        # try:
        # self.device = o3d.core.Device("CUDA:0")
        # # 测试一下能否分配内存，如果报错则回退
        # o3d.core.Tensor([1.0], device=self.device)
        # print(">>> Using CUDA for Raycasting (Fast Mode)")
        # except:
        self.device = o3d.core.Device("CPU:0")
        print(">>> Using CPU for Raycasting")

    def render(self, height_map, mask, world_T_frame, world_T_cam, K, target_H, target_W):
        """
        输入:
            height_map: (H, W) float, 单位米
            mask: (H, W) uint8, 255=valid
        输出:
            pred_depth, pred_normal, hit_mask
        """
        # --- 1. 准备顶点 (Vertices) ---
        # 即使被 Mask 掉的点也保留占位，这样索引对得上
        z = height_map.flatten()
        x = (self.grid_x - self.W // 2 + 0.5) * self.ppmm / 1000.0
        y = (self.grid_y - self.H // 2 + 0.5) * self.ppmm / 1000.0
        vertices = np.stack([x, y, z], axis=-1) # (N, 3)
        
        # 变换到 Camera Frame
        # T_cam_frame = inv(T_world_cam) @ T_world_frame
        cam_T_world = np.linalg.inv(world_T_cam)
        cam_T_frame = cam_T_world @ world_T_frame
        
        R = cam_T_frame[:3, :3]
        t = cam_T_frame[:3, 3]
        vertices_cam = (R @ vertices.T).T + t # (N, 3)

        # --- 2. 动态过滤三角形 (Masking) ---
        # 只有当三角形的 3 个顶点都在 Mask 内时，才保留该三角形
        # 这样边缘会非常干净，不会有"拉丝"现象
        
        # 将 mask 展平: 0/False 为无效, 1/True 为有效
        flat_mask = (mask.flatten() > 128) # (N,) bool
        
        # 检查所有三角形的顶点是否有效
        # tri_indices: (Total_Tris, 3)
        # 我们使用 numpy 的广播机制快速检查
        # valid_tris: (Total_Tris,) bool
        
        # 获取三角形对应的 mask 值
        m1 = flat_mask[self.all_triangles[:, 0]]
        m2 = flat_mask[self.all_triangles[:, 1]]
        m3 = flat_mask[self.all_triangles[:, 2]]
        
        # 只有 3 个点都有效才保留
        valid_tri_mask = m1 & m2 & m3
        
        # 如果没有有效的三角形，直接返回空
        if np.sum(valid_tri_mask) == 0:
             return np.zeros((target_H, target_W), dtype=np.float32), \
                    np.zeros((target_H, target_W, 3), dtype=np.float32), \
                    np.zeros((target_H, target_W), dtype=bool)

        # 筛选出有效的三角形索引
        active_triangles = self.all_triangles[valid_tri_mask]
        
        # --- 3. 构建 Raycasting Scene ---
        mesh = o3d.t.geometry.TriangleMesh(self.device)
        mesh.vertex.positions = o3d.core.Tensor(vertices_cam.astype(np.float32), device=self.device)
        mesh.triangle.indices = o3d.core.Tensor(active_triangles, device=self.device)
        
        scene = o3d.t.geometry.RaycastingScene()
        # 添加 Mesh ID=0
        scene.add_triangles(mesh)
        
        # --- 4. 生成射线并渲染 ---
        intrinsic_matrix = K[:3, :3].astype(np.float64)
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(intrinsic_matrix),
            extrinsic_matrix=o3d.core.Tensor(np.eye(4)), # 已经是 Cam Frame 了
            width_px=target_W,
            height_px=target_H
        )
        
        ans = scene.cast_rays(rays)
        
        # --- 5. 获取结果 ---
        pred_depth = ans['t_hit'].numpy()
        pred_normal = ans['primitive_normals'].numpy()
        
        hit_mask = np.isfinite(pred_depth)
        
        # 清理背景
        pred_depth[~hit_mask] = 0.0
        pred_normal[~hit_mask] = 0.0
        
        return pred_depth, pred_normal, hit_mask
# ==========================================
# 1. Geometry & Visualization Utils (Optimized)
# ==========================================
def compute_normal_from_depth_vectorized(depth_image, K, foreground_mask):
    """
    使用 NumPy 向量化操作计算 Normal，比双重循环快 100 倍以上。
    """
    H, W = depth_image.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 1. 生成坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 2. 反投影到 3D (Z 已经在 depth_image 中)
    # 注意：这里只计算 mask 区域可以进一步加速，但全图计算通常也很快且代码更简洁
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 3. 准备 Shift 后的矩阵用于计算差分
    # 这里的 padding 保证形状一致，边界处设为 nan 或 0 不影响，因为后面会 mask 掉
    p = np.stack([x, y, z], axis=-1) # (H, W, 3)
    
    # 向右平移 (p_r) 和 向左平移 (p_l)
    p_r = np.pad(p, ((0,0), (0,1), (0,0)), mode='edge')[:, 1:]
    p_l = np.pad(p, ((0,0), (1,0), (0,0)), mode='edge')[:, :-1]
    
    # 向下平移 (p_d) 和 向上平移 (p_u)
    p_d = np.pad(p, ((0,1), (0,0), (0,0)), mode='edge')[1:, :]
    p_u = np.pad(p, ((1,0), (0,0), (0,0)), mode='edge')[:-1, :]
    
    # 4. 计算向量
    vec_h = p_r - p_l # 水平向量
    vec_v = p_d - p_u # 垂直向量
    
    # 5. 叉乘计算法向量
    # 注意 open3d 或传统视觉坐标系下，cross(x, y) 可能指向 z 正或负，取决于坐标系定义
    # 原代码是 cross(vec_h, vec_v)，这里保持一致
    normals = np.cross(vec_h, vec_v)
    
    # 6. 归一化
    norm_val = np.linalg.norm(normals, axis=-1, keepdims=True)
    # 避免除以 0
    norm_val[norm_val < 1e-9] = 1.0 
    normals = normals / norm_val

    # 7. 处理 Mask 和 边缘跳变 (Discontinuity Check)
    # 原始逻辑：depth > 0 且 foreground > 128
    valid_mask = (z > 1e-9) & (foreground_mask > 128)
    
    # Erode mask (与原代码一致)
    kernel = np.ones((3, 3), np.uint8)
    valid_mask_eroded = cv2.erode(valid_mask.astype(np.uint8), kernel, iterations=1) > 0
    
    # 深度跳变检测 (Vectorized)
    thresh = 0.005
    # 检查四个邻域的深度差
    z_diff_r = np.abs(z - np.pad(z, ((0,0),(0,1)), mode='edge')[:, 1:])
    z_diff_l = np.abs(z - np.pad(z, ((0,0),(1,0)), mode='edge')[:, :-1])
    z_diff_d = np.abs(z - np.pad(z, ((0,1),(0,0)), mode='edge')[1:, :])
    z_diff_u = np.abs(z - np.pad(z, ((1,0),(0,0)), mode='edge')[:-1, :])
    
    is_continuous = (z_diff_r < thresh) & (z_diff_l < thresh) & \
                    (z_diff_d < thresh) & (z_diff_u < thresh)
    
    # 最终 Mask
    final_valid = valid_mask_eroded & is_continuous
    
    # 应用 Mask，无效区域设为 0
    normals[~final_valid] = 0.0
    
    # 生成用于可视化的 mask (norm > 0.1)
    valid_normal_mask = (np.linalg.norm(normals, axis=-1) > 0.1).astype(np.uint8) * 255
    
    return normals.astype(np.float32), valid_normal_mask

# ==========================================
# 2. Optimized Projection (Replaces griddata)
# ==========================================
def fast_project_and_fill(uvs, depths, H, W):
    """
    使用整数索引映射 + 形态学闭运算代替 griddata。
    速度提升 100x。
    """
    # 1. 过滤在图像范围内的点
    u, v = uvs[:, 0], uvs[:, 1]
    mask_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[mask_bounds]
    v = v[mask_bounds]
    d = depths[mask_bounds]
    
    if len(d) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # 2. 转为整数坐标
    u_int = np.round(u).astype(np.int32)
    v_int = np.round(v).astype(np.int32)
    
    # 3. 处理遮挡：同一个像素如果有多个点，保留深度最小的（最近的）
    # 方法：按深度降序排序，然后赋值。这样小的深度会覆盖大的深度。
    sort_idx = np.argsort(d)[::-1] # 降序
    u_int = u_int[sort_idx]
    v_int = v_int[sort_idx]
    d_sorted = d[sort_idx]
    
    # 创建深度图，初始化为 0
    depth_map = np.zeros((H, W), dtype=np.float32)
    
    # 利用 NumPy 高级索引直接赋值 (后出现的覆盖前面的，所以最小的 z 最后写入)
    depth_map[v_int, u_int] = d_sorted
    
    # 4. 填补空洞 (Hole Filling)
    # 因为直接投影会有很多单像素空洞，使用简单的形态学闭运算填充
    # 定义 Mask：有数据的地方为 1
    valid_mask = (depth_map > 1e-9).astype(np.uint8)
    
    # 如果空洞较大，可以使用 cv2.inpaint (稍微慢一点点，但质量更好)
    # 这里使用简单的形态学闭运算 (Close) = Dilate -> Erode，能填补小黑点
    kernel = np.ones((3, 3), np.uint8)
    
    # 先膨胀扩散深度值
    dilated_depth = cv2.dilate(depth_map, kernel, iterations=1)
    dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
    
    # 哪里原本没有值，但膨胀后有值了，就填补进去
    hole_mask = (valid_mask == 0) & (dilated_mask == 1)
    depth_map[hole_mask] = dilated_depth[hole_mask]
    
    # 如果还需要更平滑，可以再加一次中值滤波，但通常为了 SDF 精度，不做过多平滑
    # depth_map = cv2.medianBlur(depth_map, 3)

    return depth_map

# ==========================================
# 1. Geometry & Visualization Utils
# ==========================================
def compute_normal_from_depth(depth_image, K, foreground_mask):
    """
    Compute normal map from depth image using camera intrinsics.
    Includes edge handling for depth discontinuities.
    """
    H, W = depth_image.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Valid mask: depth > 0 and inside foreground
    valid_mask = (depth_image > 1e-9) & (foreground_mask > 128)
    
    # Erode to avoid boundary artifacts
    kernel = np.ones((3, 3), np.uint8)
    valid_mask_eroded = cv2.erode(valid_mask.astype(np.uint8), kernel, iterations=1) > 0
    
    normal = np.zeros((H, W, 3), dtype=np.float32)
    
    # Compute normals (central difference + cross product)
    for i in range(1, H-1):
        for j in range(1, W-1):
            if not valid_mask_eroded[i, j]: continue
            
            # Neighbor check
            if not valid_mask[i-1:i+2, j-1:j+2].all(): continue
            
            p_c = np.array([x[i, j], y[i, j], z[i, j]])
            p_r = np.array([x[i, j+1], y[i, j+1], z[i, j+1]])
            p_l = np.array([x[i, j-1], y[i, j-1], z[i, j-1]])
            p_d = np.array([x[i+1, j], y[i+1, j], z[i+1, j]])
            p_u = np.array([x[i-1, j], y[i-1, j], z[i-1, j]])
            
            # Discontinuity check
            thresh = 0.005 # 5mm threshold
            if (abs(z[i,j]-z[i,j+1])>thresh or abs(z[i,j]-z[i,j-1])>thresh or
                abs(z[i,j]-z[i+1,j])>thresh or abs(z[i,j]-z[i-1,j])>thresh):
                continue
                
            vec_h = p_r - p_l
            vec_v = p_d - p_u
            n = np.cross(vec_h, vec_v)
            norm_val = np.linalg.norm(n)
            if norm_val > 1e-9:
                normal[i, j] = n / norm_val

    valid_normal_mask = (np.linalg.norm(normal, axis=-1) > 0.1).astype(np.uint8) * 255
    return normal, valid_normal_mask

def visualize_depth_and_normal(depth_image, normal_map, foreground_mask, save_prefix):
    # Depth Vis
    depth_vis = depth_image.copy()
    valid_d = depth_vis[foreground_mask > 128]
    if len(valid_d) > 0:
        d_min, d_max = valid_d.min(), valid_d.max()
        if d_max > d_min:
            depth_vis[foreground_mask > 128] = (depth_vis[foreground_mask > 128] - d_min) / (d_max - d_min)
        else:
            depth_vis[foreground_mask > 128] = 0.5
            
    d_color = cv2.applyColorMap((depth_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
    d_color[foreground_mask < 128] = 0
    cv2.imwrite(save_prefix + "_depth_vis.png", d_color)
    
    # Normal Vis
    n_vis = (normal_map + 1.0) / 2.0
    n_vis = np.clip(n_vis, 0, 1)
    n_vis_bgr = (n_vis * 255).astype(np.uint8)[:, :, [2, 1, 0]]
    valid_n_mask = np.linalg.norm(normal_map, axis=-1) > 0.1
    n_vis_bgr[~valid_n_mask] = 0
    cv2.imwrite(save_prefix + "_normal_vis.png", n_vis_bgr)

# ==========================================
# 2. Main Logic: Re-projection & Formatting
# ==========================================
def format_to_sdf(args):
    mesh_name = args.mesh_name
    # Input Paths
    align_path = os.path.join(args.data_root, mesh_name, "alignment", "contact_frames.npy")
    data_dir = os.path.join(args.data_root, mesh_name, "reconstruction")
    
    # Output Path
    final_output_dir = os.path.join(args.output_dir, mesh_name)
    os.makedirs(final_output_dir, exist_ok=True)
    
    print(f"=== Formatting SDF Data for {mesh_name} ===")
    print(f"Align Path: {align_path}")
    print(f"Data Dir:   {data_dir}")
    print(f"Output To:  {final_output_dir}")
    
    if not os.path.exists(align_path):
        raise FileNotFoundError(f"Alignment file not found: {align_path}")
    contact_frames = np.load(align_path, allow_pickle=True)
    
    # Find depth files
    search_pattern = os.path.join(data_dir, "*_pred_depth.npy")
    depth_files = sorted(glob.glob(search_pattern))
    
    if len(depth_files) == 0:
        raise FileNotFoundError(f"No predicted depth files: {search_pattern}")
    print(f"Found {len(depth_files)} frames to process.")
    
    # --- Parameters ---
    ppmm = 0.0634
    H, W = 240, 320
    
    # [Modified] Calculate scale based on args.size
    scale = args.size
    print(f"Using Scale Factor: {scale} (Target Size: {args.size})")

    # Virtual Camera Intrinsics
    focal_length = W
    K = np.array([
        [focal_length, 0, W / 2, 0],
        [0, focal_length, H / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    # Initialize Renderer
    # 注意：Renderer 依然在原始物理尺度下工作，缩放是后处理步骤
    renderer = MeshRenderer(H, W, ppmm)
    
    # Virtual Camera Extrinsics Offset (Pull back camera)
    frame_T_cam = np.eye(4)
    frame_T_cam[2, 3] = -focal_length * ppmm / 1000.0 

    meta_data = {
        "camera_model": "OPENCV",
        "height": H,
        "width": W,
        "has_mono_prior": True,
        "has_foreground_mask": True,
        "worldtogt": np.eye(4).tolist(),
        "frames": [],
    }

    all_pointcloud_world = []
    all_valid_depths = []
    
    # --- Processing Loop ---
    for i, d_path in enumerate(depth_files):
        # Parse ID
        basename = os.path.basename(d_path)
        idx_str = basename.replace("_pred_depth.npy", "")
        frame_idx = int(idx_str)
        
        if frame_idx >= len(contact_frames): continue
        
        # 1. Load Raw Data
        height_map_mm = np.load(d_path).astype(np.float32) * 1000
        height_pixel = height_map_mm / ppmm
        mask_path = d_path.replace("_depth.npy", "_mask.png")
        if not os.path.exists(mask_path): continue
        contact_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        rgb_out = np.zeros((H, W, 3), dtype=np.uint8) 
                
        time_start = time.time()
        
        height_map_m = height_map_mm / 1000.0
        world_T_frame = contact_frames[frame_idx]
        
        # Accumulate Point Cloud (Keep in ORIGINAL World Scale for now, scale at the end)
        if args.save_pcd or True: # Use True to ensure we calculate bounds
            ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            pointcloud = np.stack([xs - W // 2 + 0.5, ys - H // 2 + 0.5, height_pixel], axis=-1) * ppmm / 1000.0
            
            valid_indices = contact_mask > 128
            if np.sum(valid_indices) > 10:
                homo_pointcloud = np.concatenate([pointcloud, np.ones((H, W, 1))], axis=-1)
                homo_pointcloud = homo_pointcloud[valid_indices]
                homo_pointcloud_world = (world_T_frame @ homo_pointcloud.T).T
                all_pointcloud_world.append(homo_pointcloud_world[:, :3])

        # Calc Camera Pose
        world_T_cam = world_T_frame @ frame_T_cam
        
        # [Modified] Scale the camera translation
        # 这模拟了将整个世界缩放，相机位置也随之缩放
        world_T_cam[:3, 3] = world_T_cam[:3, 3] * scale
        
        # === Rendering ===
        # Renderer 使用原始米制单位渲染 (物理一致性)
        depth_image, normal_map, hit_mask = renderer.render(
            height_map_m, 
            contact_mask, 
            world_T_frame, 
            # 注意：Renderer 内部需要原始的 world_T_cam 来计算相对位置
            # 因为 vertices 是原始尺度的。
            # 所以这里我们要传入 *未缩放* 的 cam pose 给 renderer?
            # 这是一个关键点。Input 1 是先全部算好点云，然后缩放点云，再投影。
            # Input 2 是 Raycasting。
            # 逻辑修正：
            # 如果我们传入缩放后的 world_T_cam 给 Renderer，而 Renderer 内部的 vertices 是未缩放的，
            # 那么渲染结果会错（相机跑远了，物体没变大）。
            # 
            # 这里的正确做法是：
            # 1. Renderer 使用原始未缩放的 world_T_cam 进行渲染，得到真实的物理深度。
            # 2. 拿到 depth_image 后，乘以 scale。
            # 3. 保存到 json 中的 world_T_cam 应该是缩放后的。
            
            # 重新计算一个未缩放的 cam pose 给 renderer
            world_T_frame @ frame_T_cam, 
            K, 
            H, W
        )
        
        # [Modified] Scale the depth map
        depth_image = depth_image * scale
        
        valid_reproj_mask = (hit_mask).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        valid_reproj_mask = cv2.erode(valid_reproj_mask, kernel, iterations=1)
        
        depth_image[valid_reproj_mask < 128] = 0.0
        # Normal map 也就是方向向量，在均匀缩放 (Uniform Scaling) 下是不变的，
        # 所以 renderer 渲染出的 normal 不需要乘 scale。
        normal_map[valid_reproj_mask < 128] = 0.0
        
        if np.sum(valid_reproj_mask) > 0:
            all_valid_depths.append(depth_image[valid_reproj_mask > 128])
        
        # 5. Compute Consistent Normals
        # [Note] 因为 depth_image 已经 scaled 了，K 没变，
        # compute_normal_from_depth_vectorized 会根据 K 和 Scaled Z 计算出 Scaled X,Y。
        # 这样计算出的法向量是正确的。
        normal_map, valid_normal_mask = compute_normal_from_depth_vectorized(depth_image, K, valid_reproj_mask)
        
        final_foreground_mask = valid_reproj_mask

        # --- Save Outputs ---
        file_id = f"{i:06d}"
        
        cv2.imwrite(os.path.join(final_output_dir, f"{file_id}_rgb.png"), rgb_out)
        np.save(os.path.join(final_output_dir, f"{file_id}_depth.npy"), depth_image)
        np.save(os.path.join(final_output_dir, f"{file_id}_normal.npy"), normal_map)
        cv2.imwrite(os.path.join(final_output_dir, f"{file_id}_foreground_mask.png"), 
                    np.stack([final_foreground_mask]*3, axis=-1))
        
        # visualize_depth_and_normal(depth_image, normal_map, final_foreground_mask, 
                                #    os.path.join(final_output_dir, f"{file_id}"))
        
        # Metadata
        meta_data["frames"].append({
            "rgb_path": f"{file_id}_rgb.png",
            # [Modified] 保存缩放后的相机位姿
            "camtoworld": world_T_cam.tolist(),
            "intrinsics": K[:3, :3].tolist(),
            "mono_depth_path": f"{file_id}_depth.npy",
            "mono_normal_path": f"{file_id}_normal.npy",
            "foreground_mask": f"{file_id}_foreground_mask.png"
        })
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(depth_files)} frames...")

    # --- AABB & Metadata ---
    if len(all_pointcloud_world) > 0:
        all_pc = np.concatenate(all_pointcloud_world, axis=0)
        
        # Downsample for speed
        if all_pc.shape[0] > 100000:
            idx_rand = np.random.choice(all_pc.shape[0], 100000, replace=False)
            all_pc = all_pc[idx_rand]
            
        all_d = np.concatenate(all_valid_depths) if len(all_valid_depths) > 0 else np.array([0.1])
        
        min_bounds = np.min(all_pc, axis=0)
        max_bounds = np.max(all_pc, axis=0)
        bound_size = max_bounds - min_bounds # Original size
        
        # [Modified] Add padding and then SCALE bounds
        # 逻辑参考 gelslam_data2sdf_data
        padding = 0.25 * bound_size
        min_bounds_padded = min_bounds - padding
        max_bounds_padded = max_bounds + padding
        
        # Apply Scale to bounds
        min_bounds_padded = min_bounds_padded * scale
        max_bounds_padded = max_bounds_padded * scale
        
        radius = np.linalg.norm(max_bounds_padded - min_bounds_padded) / 2
        
        # [Modified] Scale near/far
        # all_d is already scaled inside the loop, so just take min/max
        scene_box = {
            "aabb": [min_bounds_padded.tolist(), max_bounds_padded.tolist()],
            "near": float(np.min(all_d) / 2),
            "far": float(np.max(all_d) * 2),
            "radius": float(radius),
            "collider_type": "near_far"
        }
        
        meta_data["scene_box"] = scene_box
        
        # 保存完整的 meta_data.json
        json_path = os.path.join(final_output_dir, "meta_data.json")
        with open(json_path, "w") as f:
            json.dump(meta_data, f, indent=4)
        print(f"Saved metadata to {json_path}")
        
        # 生成子集 JSON 文件
        SUBSETS = [10, 20, 40, 100, 300]
        total_frames = len(meta_data["frames"])
        
        for subset_size in SUBSETS:
            if subset_size > total_frames:
                print(f"Warning: Subset {subset_size} > Available {total_frames}. Skipping.")
                continue
            
            # 创建子集数据
            subset_meta_data = meta_data.copy()
            subset_meta_data["frames"] = meta_data["frames"][:subset_size]
            
            # 重新计算该子集的 scene_box (使用对应帧的点云)
            subset_pc = np.concatenate(all_pointcloud_world[:subset_size], axis=0)
            subset_min = np.min(subset_pc, axis=0)
            subset_max = np.max(subset_pc, axis=0)
            subset_size_vec = subset_max - subset_min
            subset_padding = 0.25 * subset_size_vec
            subset_min_padded = (subset_min - subset_padding) * scale
            subset_max_padded = (subset_max + subset_padding) * scale
            subset_radius = np.linalg.norm(subset_max_padded - subset_min_padded) / 2
            
            subset_meta_data["scene_box"] = {
                "aabb": [subset_min_padded.tolist(), subset_max_padded.tolist()],
                "near": float(np.min(all_d[:subset_size]) / 2) if len(all_d[:subset_size]) > 0 else 0.01,
                "far": float(np.max(all_d[:subset_size]) * 2) if len(all_d[:subset_size]) > 0 else 1.0,
                "radius": float(subset_radius),
                "collider_type": "near_far"
            }
            
            # 保存子集 JSON
            subset_json_name = f"{mesh_name}_{subset_size}.json"
            subset_json_path = os.path.join(final_output_dir, subset_json_name)
            with open(subset_json_path, "w") as f:
                json.dump(subset_meta_data, f, indent=4)
            print(f"Saved subset {subset_size} to {subset_json_name}")
        
        if args.save_pcd:
            # Save accumulated pointcloud (Optional: save scaled or unscaled? usually unscaled for debug, but let's match logic)
            # Input 1 doesn't save ply, but let's save the original scale one for debugging
            all_points = o3d.geometry.PointCloud()
            all_points.points = o3d.utility.Vector3dVector(all_pc) 
            all_points_scaled = o3d.geometry.PointCloud()
            all_points_scaled.points = o3d.utility.Vector3dVector(all_pc * scale)
            o3d.io.write_point_cloud(os.path.join(final_output_dir, "accumulated_pointcloud_orig_scale.ply"), all_points)
            o3d.io.write_point_cloud(os.path.join(final_output_dir, "accumulated_pointcloud_scaled.ply"), all_points_scaled)
        
    else:
        raise RuntimeError("No valid geometry found for export")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root dir (e.g., ./results)")
    parser.add_argument("--mesh_name", type=str, required=True, help="Object name (e.g., bowl)")
    parser.add_argument("--output_dir", type=str, required=True, help="Final output dir")
    parser.add_argument("--save_pcd", action="store_true", help="Whether to save point clouds")
    # [Modified] Added size argument and default value
    parser.add_argument("--size", type=float, default=9, help="Uniform scale factor applied to depths and camera translations (default: 9).")
    
    args = parser.parse_args()
    format_to_sdf(args)
