import numpy as np

CLASS_NAME_TO_ID = {
'Pedestrian': -100,
'Car': 1,
'Cyclist': -100,
'Van': 2,
'Truck': 3,
'Person_sitting': -100,
'Tram': -100,
'Misc': -100,
'DontCare': -100}

# cal mean from train set
R0=np.array([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]])
R0_inv=np.linalg.inv(R0)

Tr_velo_to_cam=np.array([
    [7.49916597e-03,-9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0,0,0,1]])

def inverse_rigid_trans(Tr):
    inv_Tr=np.zeros_like(Tr)
    inv_Tr[0:3,0:3]=np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3]=np.dot(-np.transpose(Tr[0:3,0:3]),Tr[0:3,3])
    return (inv_Tr)

def camera_to_lidar(x,y,z,V2C=None,R0=None,P2=None):
    p=np.array([x,y,z,1])
    if V2C is None or R0 is None:
        p=np.matmul(R0_inv,p)
        p=np.matmul(Tr_velo_to_cam,p)
    else:
        R0_i=np.zeros((4,4))
        R0_i[:3,:3]=R0
        R0_i[3,3]=1
        p=np.matmul(np.linalg.inv(R0_i),p)
        p=np.matmul(inverse_rigid_trans(V2C),p)
    p=p[0:3]
    return (tuple(p))

def camera_to_lidar_box(boxes,V2C=None,R0=None,P2=None):
    ret=[]
    for box in boxes:
        x,y,z,h,w,l,ry=box
        (x,y,z),h,w,l,rz=camera_to_lidar(x,y,z,V2C=V2C,R0=R0,P2=P2),h,w,l,-ry-np.pi/2
        ret.append([x,y,z,h,w,l,rz])
    return (np.array(ret).reshape(-1,7))

def lidar_to_camera(x,y,z,V2C=None,R0=None,P2=None):
    p=np.array([x,y,z,1])
    if V2C is None or R0 is None:
        p=np.matmul(Tr_velo_to_cam,p)
        p=np.matmul(R0,p)
    else:
        p=np.matmul(V2C,p)
        p=np.matmul(R0,p)
    p=p[0:3]
    return (tuple(p))

def lidar_to_camera_box(boxes,V2C=None,R0=None,P2=None):
    ret=[]
    for box in boxes:
        x,y,z,h,w,l,rz=box
        (x,y,z),h,w,l,ry=lidar_to_camera(x,y,z,V2C=V2C,R0=R0,P2=P2),h,w,l,-rz-np.pi/2
        ret.append([x,y,z,h,w,l,ry])
    return (np.array(ret).reshape(-1,7))

def roty(angle):
    c=np.cos(angle)
    s=np.sin(angle)
    return (np.array([[c,0,s],[0,1,0],[-s,0,c]]))

def compute_diagonal_3d(dim,location,ry):
    R=roty(ry)
    h,w,l=dim
    x_corners=[-l / 2, l / 2]
    y_corners=[0, -h]
    z_corners=[0, 0]
    corners=np.array([x_corners,y_corners,z_corners],dtype=np.float32)
    corners_3d=np.dot(R,corners)
    corners_3d=corners_3d+np.array(location,dtype=np.float32).reshape(3,1)
    return (corners_3d.transpose(1,0))

def project_to_image(pts_3d,P):
    pts_3d_homo=np.concatenate([pts_3d,np.ones((pts_3d.shape[0],1),dtype=np.float32)],axis=1)
    pts_2d=np.dot(P,pts_3d_homo.transpose(1,0)).transpose(1,0)
    pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
    return (pts_2d.astype(np.int_))

def compute_relative_angle_2d(dim, location, ry, P2):
    corners_3d=compute_diagonal_3d(dim,location,ry)
    corners_2d=project_to_image(corners_3d,P2)

    pt1_x, pt1_y = corners_2d[0][0], corners_2d[0][1]
    pt2_x, pt2_y = corners_2d[1][0], corners_2d[1][1]

    pt3_x, pt3_y = corners_2d[0][0] - 50, corners_2d[0][1]

    angle1 = np.arctan2(pt2_y - pt1_y, pt2_x - pt1_x) * 180/np.pi
    angle2 = np.arctan2(pt3_y - pt1_y, pt3_x - pt1_x) * 180/np.pi

    rel_angle = 360 - (angle2 - angle1)
    return np.radians(rel_angle) if 0<=ry<=3.14 else np.radians(180 + rel_angle)

class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo 
        #(veldoyne->camera->rectification->image)
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
    '''

    def __init__(self, calib_filepath):
        self.file = self.read_calib_file(calib_filepath)
        calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P2 = calibs['P2']
        self.P2 = np.reshape(self.P2, [3, 4])
        self.P3 = calibs['P3']
        self.P3 = np.reshape(self.P3, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo2cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P2[0, 2]
        self.c_v = self.P2[1, 2]
        self.f_u = self.P2[0, 0]
        self.f_v = self.P2[1, 1]
        self.b_x = self.P2[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P2[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        return {'P2': P2.reshape(3, 4),
                'P3': P3.reshape(3, 4),
                'R_rect': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3,4)}