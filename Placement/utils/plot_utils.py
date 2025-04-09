import os
import math
import numpy as np
import cv2
import torch
import re
import torch.utils as utils
from torch.utils.data import Dataset
# from data_process.kitti_data_utils import gen_hm_radius, compute_radius, Calibration, get_filtered_lidar
# from data_process.kitti_bev_utils import makeBEVMap, drawRotatedBox, get_corners
# from data_process import transformation
# import config.kitti_config as cnf


def sort_nicely(l):
    convert=lambda text:int(text) if text.isdigit() else text
    alphanum_key=lambda key:[convert(c) for c in re.split('([0-9]+)',key)]
    l.sort( key=alphanum_key)
    return (l)

boundary = {
    "minX": 0,
    "maxX": 50,
    "minY": -25,
    "maxY": 25,
    "minZ": -2.73,
    "maxZ": 1.27
}

CLASS_NAME_TO_ID = {
    'Pedestrian': -100,
    'Car': 1,
    'Cyclist': -100,
    'Van': 1,
    'Truck': 3,
    'Person_sitting': -100,
    'Tram': -100,
    'Misc': -100,
    'DontCare': -100}

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l


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

def compute_box_3d(dim,location,ry):
    R=roty(ry)
    h,w,l=dim
    x_corners=[l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners=[0, 0, 0, 0, -h, -h, -h, -h]
    z_corners=[w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners=np.array([x_corners,y_corners,z_corners],dtype=np.float32)
    corners_3d=np.dot(R,corners)
    corners_3d=corners_3d+np.array(location,dtype=np.float32).reshape(3,1)
    return (corners_3d.transpose(1,0))

def project_to_image(pts_3d,P):
    pts_3d_homo=np.concatenate([pts_3d,np.ones((pts_3d.shape[0],1),dtype=np.float32)],axis=1)
    pts_2d=np.dot(P,pts_3d_homo.transpose(1,0)).transpose(1,0)
    pts_2d=pts_2d[:,:2]/pts_2d[:,2:]
    return (pts_2d.astype(np.int))

def create_2d_box(box_2d):
    corner1_2d=box_2d[0]
    corner2_2d=box_2d[1]
    pt1=corner1_2d
    pt2=(corner1_2d[0],corner2_2d[1])
    pt3=corner2_2d
    pt4=(corner2_2d[0],corner1_2d[1])
    return (pt1, pt2, pt3, pt4) 


def plot_2d_box(img, box_2d):
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)
    cv2.line(img, pt1, pt2, (255,0,0), 2)
    cv2.line(img, pt2, pt3, (255,0,0), 2)
    cv2.line(img, pt3, pt4, (255,0,0), 2)
    cv2.line(img, pt4, pt1, (255,0,0), 2)
    return img


def draw_box_3d(image, corners, color=(255, 255, 255)):
    ''' Draw 3d bounding box in image
        corners: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    '''

    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), color, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                     (corners[f[2], 0], corners[f[2], 1]), color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                     (corners[f[3], 0], corners[f[3], 1]), color, 1, lineType=cv2.LINE_AA)

    return image




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

    def cart2hom(self, pts_3d):
        pts_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)))
        return (pts_hom)

class kitti_dataset(utils.data.Dataset):
    def __init__(self,dataset_dir,input_size,hm_size,mode="train",hflip_prob=None,num_samples=None):
        self.dataset_dir=dataset_dir
        self.input_size=input_size
        self.hm_size=hm_size
        assert mode in ["train","val","test"],"Invalid mode:{}".format(mode)    
        self.mode=mode
        self.is_test=(self.mode=="test")
        sub_folder="training"
        self.image_dir=os.path.join(self.dataset_dir,sub_folder,"image_2")
        # self.image3_dir=os.path.join(self.dataset_dir,sub_folder,"image_3")
        self.label_dir=os.path.join(self.dataset_dir,sub_folder,"label_2")
        # self.twodlabel_dir=os.path.join(self.dataset_dir,sub_folder,"2dlabel_2")
        self.calib_dir=os.path.join(self.dataset_dir,sub_folder,"calib")
        split_txt_path=os.path.join(self.dataset_dir,"ImageSets","{}.txt".format(mode))
        self.sample_id_list=[int(x.strip()) for x in open(split_txt_path).readlines()]
        if num_samples is not None:
            self.sample_id_list=self.sample_id_list[:num_samples]
        self.num_samples=len(self.sample_id_list)

    def __len__(self):
        return (len(self.sample_id_list))
    
    def get_image(self,idx):
        img_path=os.path.join(self.image_dir,'{:06d}.png'.format(idx))
        img=cv2.imread(img_path)
        # print(img_path)
        return (img_path,img)
    
    def get_image3(self,idx):
        img_path=os.path.join(self.image3_dir,'{:06d}.png'.format(idx))
        img=cv2.imread(img_path)
        return (img_path,img)
    
    def load_img(self,idx):
        sample_id=int(self.sample_id_list[idx])
        img_path,img_rgb=self.get_image(sample_id)
        metadatas={"img_path":img_path}
        return (metadatas,img_rgb)

    def get_calib(self,idx):
        calib_file=os.path.join(self.calib_dir,'{:06d}.txt'.format(idx))
        return (Calibration(calib_file))
    
    
    def get_label(self,idx):
        labels=[]
        bboxes=[]
        label_path=os.path.join(self.label_dir,'{:06d}.txt'.format(idx))
        # print(label_path)
        for line in open(label_path,"r"):
            line=line.rstrip()
            line_parts=line.split(' ')
            obj_name=line_parts[0]
            # print(obj_name)
            cat_id=int(CLASS_NAME_TO_ID[obj_name])
            if cat_id<=-99:
                continue
            truncated=int(float(line_parts[1]))
            occluded=int(line_parts[2])
            alpha=float(line_parts[3])#angle
            # xmin, ymin, xmax, ymax
            bbox=np.array([float(line_parts[4]),float(line_parts[5]),float(line_parts[6]),float(line_parts[7])])
            # height,width,length(h,w,l)
            h,w,l=float(line_parts[8]),float(line_parts[9]),float(line_parts[10])
            #location (x,y,z) in camera coord.
            x,y,z=float(line_parts[11]),float(line_parts[12]),float(line_parts[13])
            ry=float(line_parts[14])#yaw angle(around Y-axis in camera coordinates)[-pi..pi]
            object_label=[cat_id,x,y,z,h,w,l,ry]
            labels.append(object_label)
            bboxes.append(bbox)
        if len(labels)==0:
            labels=np.zeros((1,8),dtype=np.float32)
            has_labels=False
        else:
            labels=np.array(labels,dtype=np.float32)
            has_labels=True
        return (bboxes,labels,has_labels)

    def load_img_with_targets(self,idx):
        sample_id=int(self.sample_id_list[idx])
        img_path=os.path.join(self.image_dir,'{:06d}.png'.format(sample_id))
        calib = Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
        # calib=self.get_calib(sample_id)
        bboxes,labels,has_labels=self.get_label(sample_id)
        lidar_labels=labels
        if has_labels:
            lidar_labels[:,1:]=camera_to_lidar_box(lidar_labels[:,1:],calib.V2C,calib.R0,calib.P3)
    
    def get_img_with_label(self,idx):
        sample_id=int(self.sample_id_list[idx])
        img_path,img_rgb=self.get_image(sample_id)
        _,img_rgb3=self.get_image3(sample_id)
        calib=Calibration(img_path.replace(".png", ".txt").replace("image_2", "calib"))
        # calib=self.get_calib(idx)
        bboxes,labels,has_labels=self.get_label(sample_id)
        if has_labels:
            labels[:,1:]=camera_to_lidar_box(labels[:,1:],calib.V2C,calib.R0,calib.P3)  
        minX=boundary['minX']
        maxX=boundary['maxX']
        minY=boundary['minY']
        maxY=boundary['maxY']
        minZ=boundary['minZ']
        maxZ=boundary['maxZ']
        label_x=(labels[:, 1] >= minX) & (labels[:, 1] < maxX)
        label_y=(labels[:, 2] >= minY) & (labels[:, 2] < maxY)
        label_z=(labels[:, 3] >= minZ) & (labels[:, 3] < maxZ)
        mask_label=label_x & label_y & label_z
        # labels=labels[mask_label]
        labels[:,1:]=lidar_to_camera_box(labels[:,1:],calib.V2C,calib.R0,calib.P3)
        # targets={"rgb":img_rgb,"path":img_path,"p2":calib.P2}
        # "bbox":bboxes,"labels":label,
        return(bboxes,labels,img_rgb,img_path,calib.P3,img_rgb3,calib,sample_id)

    def __getitem__(self,idx):
        if self.is_test:
            return(self.get_img_with_label(idx))
        else:
            return(self.get_img_with_label(idx))
    





# for idx in range(len(dataset)):
#     bboxes=dataset[idx][0]
#     cam_labs=dataset[idx][1]
#     imgs=dataset[idx][2]
#     path=dataset[idx][3]
#     P2=dataset[idx][4]
#     image_3=dataset[idx][5]
#     calibration_mat=dataset[idx][6]
#     id=dataset[idx][7]
#     # print(len(bboxes))
#     # print(calibration_mat.file.keys())
#     # print(cam_labs)
#     # print(id)

#     for box_idx in range(len(cam_labs)):
#         cls_id,location,dim,ry=cam_labs[box_idx][0],cam_labs[box_idx][1:4],cam_labs[box_idx][4:7],cam_labs[box_idx][7]
#         # if location[2] < 2.0:  # The object is too close to the camera, ignore it during visualization
#         #     continue
#         if cls_id < 0:
#             continue

#         x,y,z=cam_labs[box_idx][1:4]
#         h,w,l=cam_labs[box_idx][4:7]
#         corners_3d=compute_box_3d(dim,location,ry)
#         corners_2d=project_to_image(corners_3d,P2)
#         imgs=draw_box_3d(imgs,corners_2d)
#         print("saving")
#         cv2.imwrite("",imgs)
