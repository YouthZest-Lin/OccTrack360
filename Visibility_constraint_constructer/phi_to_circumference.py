from corner_8_to_FOV import undistort_point
import numpy as np
import cv2
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

def distort_point(x_u, y_u, k1, k2 ,p1, p2):
    r_square = x_u ** 2 + y_u ** 2
    L = 1 + k1 * r_square + k2 * r_square ** 2
    x_t = 2 * p1 * x_u * y_u + p2 * (r_square + 2 * x_u ** 2)
    y_t = 2 * p2 * x_u * y_u + p1 * (r_square + 2 * y_u ** 2)
    x_d = x_u * L + x_t
    y_d = y_u * L + y_t
    return x_d, y_d
if __name__=='__main__':
    
    limit_of_a = np.sqrt(1 / (xi ** 2 - 1)) #- 0.005
    cnt = 192
    while 1:
        points = []
        img = cv2.imread(r"G:\kitti\2013_05_28_drive_0000_sync_image_03\2013_05_28_drive_0000_sync\image_03\data_rgb" + "/" + str(cnt).zfill(10) + '.png') 
        for r in np.linspace(0.0*limit_of_a, limit_of_a, 50):
            for phi in np.linspace(0,2 * np.pi,3600):
                x_u = r * np.cos(phi)
                y_u = r * np.sin(phi)
                x_d, y_d = distort_point(x_u, y_u, k1, k2 ,p1, p2)
                #x_d, y_d = -1 * x_d, -1 * y_d
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
                
                img[int(u), int(v), :] = np.array([0,0,255])
        #img[int(u), int(v), :] = np.array([0,0,0])
        points.append([u,v])
        points = np.stack(points)
        max_idx = np.argmax(np.sum(points,axis = 1))
        print(f'rb points:{points[max_idx]}')
        min_idx = np.argmin(np.sum(points,axis = 1))
        print(f'lt points:{points[min_idx]}')
        #print(f'v_min:{points[:,1].min()}')
        #present_img = 0 * np.ones([2 * W, 2 * H, 3])
        #present_img[W - W // 2  : W + W // 2 , H - H // 2 : H + H // 2 ,:] = img
        #present_img[W // 2 + int(u), H // 2 + int(v), :] = np.array([0,0,255])
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        #present_img = cv2.resize(present_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
        cnt += 1
    
    print('Done!')
