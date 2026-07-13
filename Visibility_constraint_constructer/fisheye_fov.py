from corner_8_to_FOV import undistort_point
from phi_to_circumference import distort_point
import numpy as np
import cv2
from utils.voxels_from_directions import FoV_from_directions
# the intrinsic parameters of the left fisheye camera
xi = 2.2134047507854890
k1 = 1.6798235660113681e-02
k2 = 1.6548773243373522e+00
p1 = 4.2223943394772046e-04
p2 = 4.2462134260997584e-04
gamma1 = 1.3363220825849971e+03
gamma2 = 1.3357883350012958e+03
u0 = 7.1694323510126321e+02
v0 = 7.0576498308221585e+02
W, H = 1400, 1400
# the intrinsic parameters of the right fisheye camera
xi_R = 2.5535139132482758e+00
k1_R = 4.9370396274089505e-02
k2_R = 4.5068455478645308e+00
p1_R = 1.3477698472982495e-03
p2_R = -7.0340482615055284e-04
gamma1_R = 1.4854388981875156e+03
gamma2_R = 1.4849477411748708e+03
u0_R = 6.9888316784030962e+02
v0_R = 6.9814541887723055e+02

def T_Fish2Lidar(z_theta, t):
    R = pitch_R(z_theta)
    T = np.zeros([4,4])
    T[:3,:3] = R
    T[:3,3] = t
    T[3,3] = 1
    return T
def pitch_R(z_theta):
    R = [[np.cos(z_theta), -np.sin(z_theta), 0],
         [np.sin(z_theta), np.cos(z_theta), 0],
         [0,0,1]]
    R = np.stack(R)
    return R
def cal_directions(T, xi, k1, k2, p1, p2, gamma1, gamma2, u0, v0, W, H):
    I = [xi,k1,k2,p1,p2,gamma1,gamma2,u0,v0,W,H]
    directions = []
    R_pos = []
    error_pos = []
    max_R = -xi - 2
    min_R = xi + 2
    limit_of_a = np.sqrt(1 / (xi ** 2 - 1)) #- 0.005
    cnt = 500
    for a in np.linspace(0, limit_of_a, 100):
        val = 1 + a**2*(1 - xi**2)
        cos_theta = (-a**2*xi + np.sqrt(val)) / (1 + a**2)
        theta = np.arccos(cos_theta)
        #print(theta)
        for phi in np.linspace(0,2 * np.pi,360):
            x_u = a * np.cos(phi)
            y_u = a * np.sin(phi)
            z_u = np.sqrt(1 - a ** 2)
            #z_u = cos_theta - xi
            #z_shifted = z_u - xi
            x_d, y_d = distort_point(x_u, y_u, k1, k2 ,p1, p2)
            #x_d, y_d = x_u, y_u
            
            u = x_d * gamma1 + u0
            v = y_d * gamma2 + v0
            
            if u >= W:
                u = W - 1
            if u < 0:
                u = 0
            if v >= H:
                v = H - 1
            if v < 0:
                v = 0
            #x_d = (u - u0) / gamma1
            #y_d = (v - v0) / gamma2
            
            x_u_pi, y_u_pi= undistort_point(u, v, I, 100)
            r = np.sqrt(x_u_pi**2 + y_u_pi**2) - 1e-10
            val = 1 + r**2*(1 - xi**2)
            cos_theta = (-r**2*xi + np.sqrt(val)) / (1 + r**2)
            theta = np.arccos(cos_theta)
            
            z_u = cos_theta
            x_u = np.sin(theta)*np.cos(phi)
            y_u = np.sin(theta)*np.sin(phi)
            #print(cos_theta, theta, a, r)
            #print(cos_theta)
            #print(x_u, y_u, z_u, np.sqrt(x_u**2 + y_u ** 2 + z_u **2))
            if np.sqrt(x_u**2 + y_u ** 2 + z_u **2) > max_R:
                max_R = (np.sqrt(x_u**2 + y_u ** 2 + z_u **2))
                max_R_pos = [x_u, y_u, z_u]
            if np.sqrt(x_u**2 + y_u ** 2 + z_u **2) < min_R:
                min_R = np.sqrt(x_u**2 + y_u ** 2 + z_u **2)
                min_R_pos = [x_u, y_u, z_u]
            #points.append([x_d,y_d])
            # x = right, y = down, z = forward # camera coordinates
            # x = forward, y = left, z = up # velodyne coordinates
            direction = np.stack([z_u, -x_u, -y_u, 1])# remeber to transform the camera coordinates to velodyne coordinates
            #direction = np.stack([x_u, y_u, z_u, 1])
            direction = T @ direction
            direction = direction[:3]
            direction = direction / np.linalg.norm(direction)
            
            directions.append(direction)
    #points = np.stack(points)

    directions = np.stack(directions) 
    #print('Done!')
    print(f'max_R: {max_R} orrurs at error_pos: {max_R_pos}')
    print(f'min_R: {min_R} takes place at R_pos: {min_R_pos}')
    return directions
points = []
directions = []
if __name__=='__main__':
    FoV = np.zeros([256,256,32])

    T_left_LF = T_Fish2Lidar(np.pi / 2, 0 * np.array([-0.08, 0.16, 0.22]))
    #T_left_LF = T_Fish2Lidar(-np.pi / 2, np.array([-0, 0, 0]))
    #T_left_LF = T_Fish2Lidar(0, np.array([-0, 0, 0]))
    directions_left = cal_directions(T_left_LF, xi, k1, k2, p1, p2, gamma1, gamma2, u0, v0, W, H)
    left_F_in_space = np.array([128,128,16]) + 5 * np.array([0.08,0.16,0.22]) 
    FoV = FoV_from_directions(directions_left, FoV,0.5, 400,left_F_in_space)
    #FoV = FoV_from_directions(directions_left, FoV,0.5,20, left_F_in_space)
    #np.save("untransformed_normalized.npy",FoV)
    
    np.save("Left_FoV.npy", FoV)
    print("The Left is Done!")
    T_right_LF = T_Fish2Lidar(-np.pi / 2, 0 * np.array([-0.08, -0.16, 0.22]))
    directions_right = cal_directions(T_right_LF, xi_R, k1_R, k2_R, p1_R, p2_R, gamma1_R, gamma2_R, u0_R, v0_R, W, H)

    Right_FoV = np.zeros([256,256,32])
    Right_F_in_space = np.array([128,128,16]) + 5 * np.array([0.08,-0.16,0.22]) 
    Right_FoV = FoV_from_directions(directions_right, Right_FoV,0.5,400, Right_F_in_space)
    #np.save("Right_FoV.npy", Right_FoV)
    print("Done!")