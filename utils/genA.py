import torch, h5py, torchvision
import scipy.io
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from skimage.draw import polygon2mask
from rich.progress import Progress


def ray_plane_intersect_y(ray_start, ray_end, plane_point, plane_normal):
    # Calculate ray direction
    ray_direction = ray_end - ray_start
    ray_direction = ray_direction / torch.norm(ray_direction, p=2)

    # Calculate intersection distance
    dist = (ray_start[1] - plane_point[1]) / -ray_direction[1]
    
    if dist > 0:
        point = ray_start + dist * ray_direction
    else:  # We don't intersect the plane, but project to get shadows
        point = ray_start - dist * ray_direction
        point = point + point[1] * plane_normal + plane_point

    return point


def simulate_occluder(wall_point, wall_vector_1, wall_vector_2, wall_normal, walln_points, light_source_pos, occ_corner):
    # Initialize the projection of occluder corners onto the wall
    occ_corner_proj = torch.zeros((occ_corner.size(0), 3), device=occ_corner.device)
    
    # Project each occluder corner onto the wall
    for i in range(occ_corner.size(0)):
        occ_corner_proj[i, :] = ray_plane_intersect_y(light_source_pos, occ_corner[i, :], wall_point, wall_normal)

    # Coordinates in terms of the FOV (Field of View)
    occ_image_coords_1 = (occ_corner_proj[:, 0] - (wall_point[0] - wall_vector_1[0])) * walln_points / (wall_vector_1[0] * 2)
    occ_image_coords_2 = (occ_corner_proj[:, 2] - (wall_point[2] - wall_vector_2[2])) * walln_points / (wall_vector_2[2] * 2)

    # Move coordinates to CPU to use with ConvexHull and polygon2mask
    occ_image_coords_1_np = occ_image_coords_1.cpu().numpy()
    occ_image_coords_2_np = occ_image_coords_2.cpu().numpy()

    # Use ConvexHull to get the boundary vertices
    k = ConvexHull(np.stack((occ_image_coords_1_np, occ_image_coords_2_np), axis=-1)).vertices

    # Generate the occluder mask using polygon2mask
    occluder_mask = polygon2mask((walln_points, walln_points), np.stack((occ_image_coords_1_np[k], occ_image_coords_2_np[k]), axis=-1))

    # Convert the mask back to a PyTorch tensor and invert it (from False/True to 1/0)
    occluder_image = torch.tensor(1 - occluder_mask, dtype=torch.float32, device=light_source_pos.device)

    return occluder_image


def ViewingAngleFactor(MonitorPixel_xyz, FOV_xdiscr, FOV_zdiscr, D):
    powexp = 18  # You can adjust this value as needed
    
    # Calculate angle for z direction
    ang_z = torch.atan((MonitorPixel_xyz[2] - FOV_zdiscr) / D)
    Mz = torch.cos(ang_z) ** powexp

    # Calculate angle for x direction
    ang_x = torch.atan((MonitorPixel_xyz[0] - FOV_xdiscr) / D)
    Mx = torch.cos(ang_x) ** 1  # Power of 1 has no effect, but included for completeness

    # Perform the matrix multiplication equivalent (outer product in this case)
    # M = torch.mm(Mx.unsqueeze(1), Mz.unsqueeze(0))
    Mx = Mx.unsqueeze(0)
    Mz = Mz.unsqueeze(0)

    M = (Mx.T @ Mz).t()

    return M

def calc(vec):
    return vec ** 2

def simulate_block(wall_mat, wall_point, wall_vector_1, wall_vector_2, wall_normal, walln_points, light_source_pos, occ_corner, MM):
    # 计算从光源位置到墙壁位置的向量
    vec = wall_mat[[0, 2], :].unsqueeze(2) - light_source_pos[:, [0, 2]].permute(1, 0).unsqueeze(1)
    
    # 计算距离的平方
    vs = calc(vec)
    
    # y 方向是常数，因此只需计算一次
    vs2 = torch.full((1, walln_points), (wall_mat[1, 0] - light_source_pos[0, 1]) ** 2)

    # 计算最终距离并归一化
    tmp = calc(vs[0, :, :] + vs2.unsqueeze(2) + vs[1, :, :].unsqueeze(1))

    # 计算最终的光强度
    intensity = MM * vs2.unsqueeze(2) / tmp
    
    # 初始化图像矩阵
    image = torch.zeros(walln_points, walln_points, device=wall_mat.device)

    # 计算遮挡物的位置
    if occ_corner.numel() > 0:
        for i in range(vec.shape[2]):
            occluder_image = simulate_occluder(wall_point, wall_vector_1, wall_vector_2, wall_normal, walln_points, light_source_pos[i, :], occ_corner[:, :, 0])
            for o in range(1, occ_corner.shape[2]):
                occluder_image *= simulate_occluder(wall_point, wall_vector_1, wall_vector_2, wall_normal, walln_points, light_source_pos[i, :], occ_corner[:, :, o])
            
            image += intensity[:, :, i] * occluder_image
    else:
        image = torch.sum(intensity, dim=2)

    return image


def simulate_A(wallparam, occ_corner, simuParams, Mon_xdiscr, Mon_zdiscr, Monitor_depth):
    # Extract wall parameters
    wall_matr = wallparam['wall_matr']
    wall_point = wallparam['wall_point']
    wall_vector_1 = wallparam['wall_vector_1']
    wall_vector_2 = wallparam['wall_vector_2']
    wall_normal = wallparam['wall_normal']
    walln_points = wallparam['walln_points']

    # Simulation parameters
    NumBlocks_row = simuParams['NumBlocks'][0]
    NumBlocks_col = simuParams['NumBlocks'][1]
    Ndiscr_mon = simuParams['Ndiscr_mon']

    blockcount = 0

    # Initialize A matrix on GPU
    A = torch.zeros((walln_points**2, NumBlocks_row * NumBlocks_col), device=wall_matr.device)
    with Progress() as ps:
        task = ps.add_task('[cyan]Simulating light transport matrix...', total=len(range(NumBlocks_col - 1, -1, -1))*len(range(NumBlocks_row)))
        # Loop through each scene patch
        for mc in range(NumBlocks_col - 1, -1, -1):
            for mr in range(NumBlocks_row):
                lightposy = Monitor_depth

                # View angle model
                if simuParams['viewAngleCorrection'] == 1:
                    MM = ViewingAngleFactor(
                        torch.tensor([Mon_xdiscr[mc] - simuParams['IlluminationBlock_Size'][0] / 2, lightposy, Mon_zdiscr[mr] - simuParams['IlluminationBlock_Size'][1] / 2], device=wall_matr.device),
                        wall_matr[0, :], 
                        torch.flip(wall_matr[2, :], dims=[0]), 
                        simuParams['D']
                    )
                else:
                    MM = torch.ones_like(wall_matr[0, :])

                step_x = simuParams['IlluminationBlock_Size'][0] / Ndiscr_mon
                step_z = simuParams['IlluminationBlock_Size'][1] / Ndiscr_mon

                # 生成网格
                lightposx = torch.arange(
                    Mon_xdiscr[mc], 
                    Mon_xdiscr[mc] - simuParams['IlluminationBlock_Size'][0] + step_x / 10, 
                    -step_x, device=wall_matr.device
                )

                lightposz = torch.arange(
                    Mon_zdiscr[mr], 
                    Mon_zdiscr[mr] - simuParams['IlluminationBlock_Size'][1] + step_z / 10, 
                    -step_z, device=wall_matr.device
                )

                lightposx, lightposz = torch.meshgrid(lightposx, lightposz, indexing='ij')

                # Simulate the block
                image = torch.flip(
                    simulate_block(
                        wall_matr, wall_point, wall_vector_1, wall_vector_2, wall_normal, walln_points, 
                        torch.cat((
                            lightposx.reshape(-1, 1),  # Use reshape instead of view
                            torch.ones(Ndiscr_mon**2, 1, device=wall_matr.device) * lightposy, 
                            lightposz.reshape(-1, 1)   # Use reshape instead of view
                        ), dim=1), 
                        occ_corner, 
                        MM.unsqueeze(2)
                    ), dims=[1]
                )

                # Add to A matrix as column
                A[:, blockcount] = image.T.reshape(-1)
                blockcount += 1
                ps.update(task, advance=1)

    return A

numPixels = 128
number = 64
# numPixels = 64
# number = 128
radio = 1
Ndiscr_mon = 6

downsamp_factor = 0
viewAngleCorrection = 1
useEstimatedOccPos = True

subblocksperaxis = torch.tensor([radio, radio])
NumBlocks_sim = torch.tensor([number, number]) * subblocksperaxis
NumBlocks_cal = torch.tensor([number, number])
D = 0.95


Occ_size = torch.tensor([0.035, 0, 0.175])
Occ_LLcorner = torch.tensor([1.41, D - 0.25, 0.835])
Occluder = torch.stack([Occ_LLcorner, Occ_LLcorner + Occ_size])


FOV_size = torch.tensor([0.375, 0.375])
FOV_LLCorner = torch.tensor([1.42, 0.77])
FOV_cord = torch.stack([FOV_LLCorner, FOV_LLCorner + FOV_size])

if number==64:
    Mon_Offset = torch.tensor([0.70, 0.42])
elif number==32:
    Mon_Offset = torch.tensor([0.70, 0.6])
elif number==96:
    Mon_Offset = torch.tensor([0.75, 0.45])
elif number==128:
    Mon_Offset = torch.tensor([0.70, 0.6])

NumBlocks_col = NumBlocks_cal[1]
NumBlocks_row = NumBlocks_cal[0]
ScreenSize = torch.tensor([0.34, 0.34])
ScreenResolution = torch.tensor([1080, 1080])
NumPixelsPerMonitorBlock = 35
PixelSize_m = ScreenSize / ScreenResolution

Mon_Offset[1] += PixelSize_m[1] * (ScreenResolution[1] % NumBlocks_cal[0])
Mon_Offset[0] += PixelSize_m[0] * (ScreenResolution[0] % NumBlocks_cal[1])

IlluminationBlock_Size = PixelSize_m * NumPixelsPerMonitorBlock / subblocksperaxis

simuParams = {
    "NumBlocks": NumBlocks_sim,
    "Ndiscr_mon": Ndiscr_mon,
    "numPixels": numPixels,
    "D": D,
    "FOV_cord": FOV_cord,
    "Occluder": Occluder,
    "viewAngleCorrection": viewAngleCorrection,
    "IlluminationBlock_Size": IlluminationBlock_Size,
    "Mon_Offset": Mon_Offset
}

wall_point = torch.tensor([FOV_LLCorner[0] + FOV_size[0] / 2, D, FOV_LLCorner[1] + FOV_size[1] / 2])
wall_vector_1 = torch.tensor([FOV_size[0] / 2, 0, 0])
wall_vector_2 = torch.tensor([0, 0, FOV_size[1] / 2])
wall_normal = torch.linalg.cross(wall_vector_1, wall_vector_2)
wall_normal = wall_normal / torch.norm(wall_normal)

walln_points = numPixels // (2 ** downsamp_factor)

wall_vec = torch.linspace(-1, 1, walln_points)
wall_matr = torch.zeros((3, walln_points))
wall_matr[0, :] = wall_point[0] + wall_vec * wall_vector_1[0] + wall_vec * wall_vector_2[0]
wall_matr[1, :] = wall_point[1] + wall_vec * wall_vector_1[1] + wall_vec * wall_vector_2[1]
wall_matr[2, :] = wall_point[2] + wall_vec * wall_vector_1[2] + wall_vec * wall_vector_2[2]

Monitor_xlim = torch.tensor([0, NumBlocks_col]) * IlluminationBlock_Size[0] + Mon_Offset[0]
Monitor_y = 0
Monitor_zlim = torch.tensor([0, NumBlocks_row]) * IlluminationBlock_Size[1] + Mon_Offset[1]

Mon_xdiscr = torch.linspace(Monitor_xlim[0].item(), Monitor_xlim[1].item(), NumBlocks_col)
Mon_zdiscr = torch.linspace(Monitor_zlim[1].item(), Monitor_zlim[0].item(), NumBlocks_row)

wallparam = {
    "wall_matr": wall_matr,
    "wall_point": wall_point,
    "wall_vector_1": wall_vector_1,
    "wall_vector_2": wall_vector_2,
    "wall_normal": wall_normal,
    "walln_points": walln_points
}

numPixels = numPixels // (2 ** downsamp_factor)

occ_corner = torch.zeros((4, 3, 4))

occ_corner[0, :, 0] = Occ_LLcorner
occ_corner[1, :, 0] = Occ_LLcorner + torch.tensor([0.035, 0, 0])
occ_corner[2, :, 0] = Occ_LLcorner + torch.tensor([0.035, 0, 0.175])
occ_corner[3, :, 0] = Occ_LLcorner + torch.tensor([0, 0, 0.175])

occ_corner[0, :, 1] = Occ_LLcorner + torch.tensor([0, -0.002, 0])
occ_corner[1, :, 1] = Occ_LLcorner + torch.tensor([0.035, 0, 0]) + torch.tensor([0, -0.002, 0])
occ_corner[2, :, 1] = Occ_LLcorner + torch.tensor([0.035, 0, 0.175]) + torch.tensor([0, -0.002, 0])
occ_corner[3, :, 1] = Occ_LLcorner + torch.tensor([0, 0, 0.175]) + torch.tensor([0, -0.002, 0])

occ_corner[0, :, 2] = Occ_LLcorner + torch.tensor([0.035, 0, 0.074])
occ_corner[1, :, 2] = Occ_LLcorner + torch.tensor([0.035, 0, 0.074]) + torch.tensor([0.065, 0, 0])
occ_corner[2, :, 2] = Occ_LLcorner + torch.tensor([0.035, 0, 0.074]) + torch.tensor([0.065, 0, 0.032])
occ_corner[3, :, 2] = Occ_LLcorner + torch.tensor([0.035, 0, 0.074]) + torch.tensor([0, 0, 0.032])

occ_corner[0, :, 3] = Occ_LLcorner + torch.tensor([0.035, 0, 0.074]) + torch.tensor([0, -0.002, 0])
occ_corner[1, :, 3] = Occ_LLcorner + torch.tensor([0.035, 0, 0.074]) + torch.tensor([0.065, 0, 0]) + torch.tensor([0, -0.002, 0])
occ_corner[2, :, 3] = Occ_LLcorner + torch.tensor([0.035, 0, 0.074]) + torch.tensor([0.065, 0, 0.032]) + torch.tensor([0, -0.002, 0])
occ_corner[3, :, 3] = Occ_LLcorner + torch.tensor([0.035, 0, 0.074]) + torch.tensor([0, 0, 0.032]) + torch.tensor([0, -0.002, 0])

occ_corner = occ_corner.clone().detach().to(torch.float64)

print(f'x_size:{number}   y_size:{numPixels}   Ndiscr_mon:{Ndiscr_mon}')
simA = simulate_A(wallparam, occ_corner, simuParams, Mon_xdiscr, Mon_zdiscr, 0).to(torch.float64)
print('simA shape: ',simA.shape)

save_sim = {'simA':simA.numpy()}

directory = "./simA"
filename = f'simA_{number}_{numPixels}.mat'
# file_path = os.path.join(args.simA, filename)
file_path = os.path.join(directory, filename)

if not os.path.exists(directory):
    os.makedirs(directory)

# with h5py.File(f'../simA/simA_{number}_{numPixels}.mat', 'w') as f:
with h5py.File(file_path, 'w') as f:
    for key, value in save_sim.items():
        f.create_dataset(key, data=value)