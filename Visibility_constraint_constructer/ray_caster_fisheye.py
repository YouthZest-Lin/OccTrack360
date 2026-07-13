import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
def my_ray_cast_gpu(voxel_grid_np, origin_np, max_steps=1000, step_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxel_grid = torch.tensor(voxel_grid_np, dtype=torch.float32, device=device)
    origin = torch.tensor(origin_np, dtype=torch.float32, device=device)
    mymask = torch.zeros_like(voxel_grid)

    
    x = torch.linspace(0, voxel_grid.shape[0] - 1, voxel_grid.shape[0], device=device)
    y = torch.linspace(0, voxel_grid.shape[1] - 1, voxel_grid.shape[1], device=device)
    z = torch.linspace(0, voxel_grid.shape[2] - 1, voxel_grid.shape[2], device=device)

    
    directions = []
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            z_max = torch.tensor(voxel_grid.shape[2] - 1, device=device)
            directions.append(torch.stack([x[i], y[j], z_max]) - origin)
            directions.append(torch.stack([x[i], y[j], torch.tensor(0, device=device)]) - origin)
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[2]):
            y_max = torch.tensor(voxel_grid.shape[1] - 1, device=device)
            directions.append(torch.stack([x[i], y_max, z[j]]) - origin)
            directions.append(torch.stack([x[i], torch.tensor(0, device=device), z[j]]) - origin)
    for i in range(voxel_grid.shape[1]):
        for j in range(voxel_grid.shape[2]):
            x_max = torch.tensor(voxel_grid.shape[0] - 1, device=device)
            directions.append(torch.stack([x_max, y[i], z[j]]) - origin)
            directions.append(torch.stack([torch.tensor(0, device=device), y[i], z[j]]) - origin)

    directions = torch.stack(directions)  # shape: (N, 3)
    directions = directions / torch.norm(directions, dim=1, keepdim=True) 

    
    origins = origin.unsqueeze(0).repeat(directions.shape[0], 1)  # shape: (N, 3)

    
    pos = origins.clone()
    last_hit_mask = torch.zeros(directions.shape[0]).to(device)#.long()
    occluded = torch.zeros(directions.shape[0]).to(device)#.long()
    
    for _ in range(max_steps):
        idx = torch.floor(pos).long()  # shape: (N, 3)
        #idx = torch.round(pos).long()  # shape: (N, 3)
        
        
        valid_mask = (
            (idx[:, 0] >= 0) & (idx[:, 0] < voxel_grid.shape[0]) &
            (idx[:, 1] >= 0) & (idx[:, 1] < voxel_grid.shape[1]) &
            (idx[:, 2] >= 0) & (idx[:, 2] < voxel_grid.shape[2])
        )
        if torch.unique(valid_mask)[-1] == False:
            break
        valid_idx = idx[valid_mask]
        
        valid_idx = torch.clamp(idx, min=0)
        valid_idx = torch.minimum(valid_idx[:, :3], torch.tensor([voxel_grid.shape[0] - 1, voxel_grid.shape[1] - 1, voxel_grid.shape[2] - 1]).to(device))
        
        if valid_idx.shape[0] == 0:
            pos += directions * step_size
            continue


        values = voxel_grid[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        #values = voxel_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        hit_mask = values > 0
        last_hit_mask = hit_mask
        hit_mask1 = torch.logical_and(hit_mask, torch.logical_not(occluded))
        occluded = torch.logical_or(hit_mask, occluded)
        #occluded = torch.logical_or(hit_mask, occluded)
        #last_hit_mask = hit_mask#.long()
        #hit_mask = hit_mask.unsqueeze(1)
        #hit_mask = hit_mask[valid_mask]
        #hit_mask = torch.logical_or(hit_mask, valid_mask)
        hit_idx = valid_idx[hit_mask1]
        '''
        if hit_idx.shape[0] != 0:
            print(1)
        '''
        mymask[hit_idx[:, 0], hit_idx[:, 1], hit_idx[:, 2]] = 1


        pos += directions * step_size

    return mymask.cpu().numpy()
def my_ray_cast_gpu1_fisheye(voxel_grid_np, origin_np, max_steps=1000, step_size=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    voxel_grid = torch.tensor(voxel_grid_np, dtype=torch.float32, device=device)
    left_origin_np = origin_np - 5 * np.array([0.87,0.16, -0.4])
    right_origin_np = origin_np - 5 * np.array([0.87,-0.16, -0.4])
    left_origin = torch.tensor(left_origin_np, dtype=torch.float32, device=device)
    right_origin = torch.tensor(right_origin_np, dtype=torch.float32, device=device)
    mymask = torch.zeros_like(voxel_grid)


    x = torch.linspace(0, voxel_grid.shape[0] - 1, voxel_grid.shape[0], device=device)
    y = torch.linspace(0, voxel_grid.shape[1] - 1, voxel_grid.shape[1], device=device)
    z = torch.linspace(0, voxel_grid.shape[2] - 1, voxel_grid.shape[2], device=device)


    #directions = []
    left_directions = []
    right_directions = []

    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            z_max = torch.tensor(voxel_grid.shape[2] - 1, device=device)
            left_directions.append(torch.stack([x[i], y[j], z_max]) - left_origin)
            left_directions.append(torch.stack([x[i], y[j], torch.tensor(0, device=device)]) - left_origin)
            right_directions.append(torch.stack([x[i], y[j], z_max]) - right_origin)
            right_directions.append(torch.stack([x[i], y[j], torch.tensor(0, device=device)]) - right_origin)
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[2]):
            y_max = torch.tensor(voxel_grid.shape[1] - 1, device=device)
            left_directions.append(torch.stack([x[i], y_max, z[j]]) - left_origin)
            right_directions.append(torch.stack([x[i], torch.tensor(0, device=device), z[j]]) - right_origin)
    for i in range(voxel_grid.shape[1]):
        for j in range(voxel_grid.shape[2]):
            x_max = torch.tensor(voxel_grid.shape[0] - 1, device=device)
            left_directions.append(torch.stack([x_max, y[i], z[j]]) - left_origin)
            left_directions.append(torch.stack([torch.tensor(0, device=device), y[i], z[j]]) - left_origin)
            right_directions.append(torch.stack([x_max, y[i], z[j]]) - right_origin)
            right_directions.append(torch.stack([torch.tensor(0, device=device), y[i], z[j]]) - right_origin)
    
    left_directions = torch.stack(left_directions)  # shape: (N, 3)
    left_directions = left_directions / torch.norm(left_directions, dim=1, keepdim=True)  
    right_directions = torch.stack(right_directions)  # shape: (N, 3)
    right_directions = right_directions / torch.norm(right_directions, dim=1, keepdim=True)  


    origins = left_origin.unsqueeze(0).repeat(left_directions.shape[0], 1)  # shape: (N, 3)


    pos = origins.clone()
    last_hit_mask = torch.zeros(left_directions.shape[0]).to(device)#.long()
    occluded = torch.zeros(left_directions.shape[0]).to(device)#.long()

    for _ in range(max_steps):
        idx = torch.floor(pos).long()  # shape: (N, 3)
        #idx = torch.round(pos).long()  # shape: (N, 3)
  
        
        valid_mask = (
            (idx[:, 0] >= 0) & (idx[:, 0] < voxel_grid.shape[0]) &
            (idx[:, 1] >= 0) & (idx[:, 1] < voxel_grid.shape[1]) &
            (idx[:, 2] >= 0) & (idx[:, 2] < voxel_grid.shape[2])
        )
        if torch.unique(valid_mask)[-1] == False:
            break
        valid_idx = idx[valid_mask]
        
        valid_idx = torch.clamp(idx, min=0)
        valid_idx = torch.minimum(valid_idx[:, :3], torch.tensor([voxel_grid.shape[0] - 1, voxel_grid.shape[1] - 1, voxel_grid.shape[2] - 1]).to(device))
        
        if valid_idx.shape[0] == 0:
            pos += directions * step_size
            continue

 
        values = voxel_grid[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        #values = voxel_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        hit_mask = values > 0
        last_hit_mask = hit_mask
        #mask1 = torch.logical_and(last_hit_mask, torch.logical_not(occluded))
        mask1 = torch.logical_not(occluded)
        #hit_mask1 = torch.logical_and(hit_mask, torch.logical_not(occluded))
        #occluded = torch.logical_or(last_hit_mask, occluded)
        occluded = torch.logical_or(hit_mask, occluded)
        #occluded = torch.logical_or(hit_mask, occluded)
        #last_hit_mask = hit_mask#.long()
        #hit_mask = hit_mask.unsqueeze(1)
        #hit_mask = hit_mask[valid_mask]
        #hit_mask = torch.logical_or(hit_mask, valid_mask)
        hit_idx = valid_idx[mask1]
        '''
        if hit_idx.shape[0] != 0:
            print(1)
        '''
        mymask[hit_idx[:, 0], hit_idx[:, 1], hit_idx[:, 2]] = 1



        pos += left_directions * step_size

    origins = right_origin.unsqueeze(0).repeat(right_directions.shape[0], 1)  # shape: (N, 3)
    

    pos = origins.clone()
    last_hit_mask = torch.zeros(right_directions.shape[0]).to(device)#.long()
    occluded = torch.zeros(right_directions.shape[0]).to(device)#.long()

    for _ in range(max_steps):
        idx = torch.floor(pos).long()  # shape: (N, 3)
        #idx = torch.round(pos).long()  # shape: (N, 3)

        
        valid_mask = (
            (idx[:, 0] >= 0) & (idx[:, 0] < voxel_grid.shape[0]) &
            (idx[:, 1] >= 0) & (idx[:, 1] < voxel_grid.shape[1]) &
            (idx[:, 2] >= 0) & (idx[:, 2] < voxel_grid.shape[2])
        )
        if torch.unique(valid_mask)[-1] == False:
            break
        valid_idx = idx[valid_mask]
        
        valid_idx = torch.clamp(idx, min=0)
        valid_idx = torch.minimum(valid_idx[:, :3], torch.tensor([voxel_grid.shape[0] - 1, voxel_grid.shape[1] - 1, voxel_grid.shape[2] - 1]).to(device))
        
        if valid_idx.shape[0] == 0:
            pos += directions * step_size
            continue

        values = voxel_grid[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        #values = voxel_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        hit_mask = values > 0
        last_hit_mask = hit_mask
        #mask1 = torch.logical_and(last_hit_mask, torch.logical_not(occluded))
        mask1 = torch.logical_not(occluded)
        #hit_mask1 = torch.logical_and(hit_mask, torch.logical_not(occluded))
        #occluded = torch.logical_or(last_hit_mask, occluded)
        occluded = torch.logical_or(hit_mask, occluded)
        #occluded = torch.logical_or(hit_mask, occluded)
        #last_hit_mask = hit_mask#.long()
        #hit_mask = hit_mask.unsqueeze(1)
        #hit_mask = hit_mask[valid_mask]
        #hit_mask = torch.logical_or(hit_mask, valid_mask)
        hit_idx = valid_idx[mask1]
        '''
        if hit_idx.shape[0] != 0:
            print(1)
        '''
        mymask[hit_idx[:, 0], hit_idx[:, 1], hit_idx[:, 2]] = 1



        pos += right_directions * step_size

    return mymask.cpu().numpy()
def my_ray_cast(voxel_grid, origin, max_steps=1000, step_size=1):
    mymask = np.zeros((200, 200, 16))
    x = np.linspace(0, voxel_grid.shape[0] - 1, voxel_grid.shape[0])  # 横向坐标
    y = np.linspace(0, voxel_grid.shape[1] - 1, voxel_grid.shape[1])  # 纵向坐标
    z = np.linspace(0, voxel_grid.shape[2] - 1, voxel_grid.shape[2])  # 纵向坐标

    
    directions = []

    xy_max = []
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            xy_max.append(np.stack([x[i], y[j], voxel_grid.shape[2] - 1]))
            direction = np.stack([x[i], y[j], voxel_grid.shape[2] - 1]) - origin
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)
    xz_max = []
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[2]):
            xz_max.append(np.stack([x[i], voxel_grid.shape[1] - 1,z[j]]))
            direction = np.stack([x[i], voxel_grid.shape[1] - 1,z[j]]) - origin
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)

    yz_max = []
    for i in range(voxel_grid.shape[1]):
        for j in range(voxel_grid.shape[2]):
            yz_max.append(np.stack([voxel_grid.shape[0] - 1, y[i], z[j]]))
            direction = np.stack([voxel_grid.shape[0] - 1, y[i], z[j]]) - origin
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)
    
    xy_min = []
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            xy_min.append(np.stack([x[i], y[j], 0]))
            direction = np.stack([x[i], y[j], 0]) - origin
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)
    xz_min = []
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[2]):
            xz_min.append(np.stack([x[i], 0,z[j]]))
            direction = np.stack([x[i], 0,z[j]]) - origin
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)

    yz_min = []
    for i in range(voxel_grid.shape[1]):
        for j in range(voxel_grid.shape[2]):
            yz_min.append(np.stack([0, y[i], z[j]]))
            direction = np.stack([0, y[i], z[j]]) - origin
            direction = direction / np.linalg.norm(direction)
            directions.append(direction)
    for direction in directions:
        pos = np.array(origin, dtype=np.float32)
        for _ in range(max_steps):
            idx = np.floor(pos).astype(int)
            if (0 <= idx[0] < voxel_grid.shape[0] and
                0 <= idx[1] < voxel_grid.shape[1] and
                0 <= idx[2] < voxel_grid.shape[2]):
                if voxel_grid[tuple(idx)] > 0:
                    mymask[tuple(idx)] = 1
                    break
            pos += direction * step_size
    return mymask
    '''
    xy_max = np.stack(xy_max)
    xz_max = np.stack(xz_max)
    yz_max = np.stack(yz_max)
    

    xy_min = np.stack(xy_min)
    xz_min= np.stack(xz_min)
    yz_min = np.stack(yz_min)
    directions = []
    '''
    

def ray_cast(voxel_grid, origin, direction, max_steps=1000, step_size=1):
    direction = direction / np.linalg.norm(direction)
    pos = np.array(origin, dtype=np.float32)

    for _ in range(max_steps):
        idx = np.floor(pos).astype(int)
        if (0 <= idx[0] < voxel_grid.shape[0] and
            0 <= idx[1] < voxel_grid.shape[1] and
            0 <= idx[2] < voxel_grid.shape[2]):
            if voxel_grid[tuple(idx)] > 0:
                return idx
        pos += direction * step_size
    return None


def generate_spherical_directions(num_theta=360, num_phi=180):
    directions = []
    for i in range(num_phi):
        phi = np.pi * (i + 0.5) / num_phi  # avoid poles
        for j in range(num_theta):
            theta = 2 * np.pi * j / num_theta
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            directions.append([x, y, z])
    return np.array(directions)



data_root0 = r'G:\sscbenchkitti360_voxel'
sequence_root = r"G:\mywork_instance"
sequence_list = os.listdir(sequence_root)
sequence_list = sequence_list[5:]
for sequence in sequence_list:
    data_root = os.path.join(data_root0, sequence, 'voxels')
    data_list = os.listdir(data_root)
    frame_cnt = 0
    for _ in range(0,len(data_list),3):
        if os.path.exists(
                        'G:/mywork_instance/'+sequence+'/center_voxel/' +str(frame_cnt).zfill(6) + ".npy")==0:
            frame_cnt += 5
            continue
        else:
            data = np.load('G:/mywork_instance/'+sequence+'/center_voxel/' +str(frame_cnt).zfill(6) + ".npy").astype(np.float32)
        if frame_cnt % 500 == 0:
            print(frame_cnt / len(data_list) / 5 * 3)

        #data = np.load(os.path.join(data_root, data_name))
        data = np.reshape(data,(256,256,32))
        save_root = os.path.join(sequence_root, sequence, 'fisheye_mask')

        grid = data != 0
        mymask = np.zeros((256, 256, 32))
        camera_pos = np.array([128, 128, 16])  
        voxel_grid = grid.astype(np.int32)
        camera_origin = camera_pos
        directions = generate_spherical_directions()
        mymask = my_ray_cast_gpu1_fisheye(voxel_grid, camera_origin, max_steps=1000, step_size=0.5)

        np.save(os.path.join(save_root, str(frame_cnt).zfill(6) + '.npy'), np.copy(mymask).astype(bool))
        frame_cnt += 5
        #np.save(os.path.join(save_root, data_name[:-3] + 'npy'), np.copy(mymask))
        #frame_cnt += 5
print('Done')
