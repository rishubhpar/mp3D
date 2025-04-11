import os
import math
import numpy as np
import sys
#from render_cars_on_scene import render_scene
INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
# if INSIDE_BLENDER:
#   try:
#     import utils
#   except ImportError as e:
#     print("\nERROR")
#     print("Running render_images.py from Blender and cannot import utils.py.") 
#     print("You may need to add a .pth file to the site-packages of Blender's")
#     print("bundled python with a command like this:\n")
#     print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
#     print("\nWhere $BLENDER is the directory where Blender is installed, and")
#     print("$VERSION is your Blender version (such as 2.78).")
#     sys.exit(1)
import random
import math
# import cv2
# from PIL import Image

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

NUSC_CLASS_NAME_TO_ID = {
    'car': 1,
    'truck': 2,
    'trailer':3,
    'barrier':4,
    'pedestrian':5,
    'traffic_cone':6,
    'bus':7,
    'construction_vehicle':8,
    'bicycle':9,
    'motorcycle':10,
    'Cyclist': -100,
    'Van': 1,
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

def render(savepath, filename, idx ,orientation, scene_blend = 'base_scene.blend'):
    # folder_path = "./ShapeNetCore.v2"
    folder_path = "/data3/rishubh/blender/car_objs"
    #print(len(os.listdir(folder_path)))


    file_paths = []
    for path, subdirs, files in os.walk(folder_path):
        for name in files:
            if(name.endswith(".obj")):# and ("/models" in path)):
                file_paths.append(os.path.join(path, name))
    #print(len(file_paths))
    
    # Load the main blendfile
    #bpy.ops.wm.open_mainfile(filepath=scene_blend)

    if bpy.app.version < (2, 78, 0):
      bpy.context.user_preferences.system.compute_device_type = 'CUDA'
      bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

    ctx = bpy.context.scene
    ctx.render.image_settings.file_format='PNG'
    ctx.render.resolution_x = 1024
    ctx.render.resolution_y = 1024
    ctx.render.engine='CYCLES'
    ctx.render.use_antialiasing = False
    ctx.cycles.feature_set = 'EXPERIMENTAL'
    ctx.cycles.device='GPU'
    ctx.cycles.use_denoising=True
    ctx.cycles.use_adaptive_sampling=True
    ctx.cycles.tile_size = 256
    ctx.cycles.film_transparent=True
    ctx.render.alpha_mode='TRANSPARENT'
    ctx.cycles.transparent_min_bounces = 32
    ctx.cycles.transparent_max_bounces = 40
    ctx.cycles.max_bounces = 50

    for o in ctx.objects:
        if o.name != 'Plane':
            o.select=True
    bpy.ops.object.delete()

    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bg.inputs[1].default_value = 1.0

    light_data = bpy.data.lamps.new(name="Shadow_Light", type='SUN')
    light_data.energy = 1
    light_object = bpy.data.objects.new(name="Shadow_Light", object_data=light_data)
    light_object.location = (3,3,100)
    ctx.objects.link(light_object)

    # plane = ob.copy()
    # plane.data = ob.data.copy()
    # plane.location = (0,0,0)
    # plane.scale = (10,10,1)
    # ctx.objects.link(plane)

    render_folder = savepath + str(filename)
    os.makedirs(render_folder, exist_ok=True)

    file_path = random.choice(file_paths)
    print("Path is: ", file_path)

    avg_dimension = [0.387, 0.279, 0.873]
    obj_dimension = [100.0, 100.0, 100.0]
    imp_obj = None

    while sum([x > 1.5*y for x,y in zip(obj_dimension, avg_dimension)])==3:
        bpy.ops.import_scene.obj(filepath = file_path)
        imp_obj = bpy.context.selected_objects[0]
        obj_dimension = list(imp_obj.dimensions)
        print('Selecting object of lower dimension..')


    imp_obj.location = (0.0, 0.0, 0.0)
    imp_obj.scale  = (5.0, 5.0, 5.0)
    imp_obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(-90) + math.radians(orientation))

    # # shadow
    # bpy.ops.mesh.primitive_plane_add()
    # plane = bpy.data.objects['Plane']
    # plane.scale = (10,10,1)
    
    # # shadow
    # plane.location = (0, 0, -imp_obj.dimensions[1]/2)
    # plane.cycles.is_shadow_catcher = True
    # imp_obj.cycles_visibility.camera = False
    # imp_obj.cycles_visibility.diffuse = False

    cam1_data = bpy.data.cameras.new("Camera")
    cam1_data.lens = 64
    cam_obj1 = bpy.data.objects.new("Camera", cam1_data)
    ctx.objects.link(cam_obj1)
    ctx.camera = cam_obj1

    diag_length = math.sqrt((obj_dimension[0])**2 + (obj_dimension[1])**2 + (obj_dimension[2])**2)
    print(11*diag_length)

    camera_location = [11*diag_length,0,1.6]   
    cam_obj1.location = tuple(camera_location)
    cam_obj1.rotation_euler = (math.radians(90.0 - 10.0), 0, math.radians(90.0))
    
    ctx.render.filepath=os.path.join(render_folder,str(idx)+".png")
    bpy.ops.render.render(write_still=True)
    
    

class iou_dataset():
    def __init__(self,root='/raid/rishubh/interns/sarthakv/clevr-dataset-gen/KITTI/label_2'):
        
        self.dataset_dir="/data3/rishubh/MonoDTR/data/KITTI/object"
        self.image_dir="/data3/rishubh/MonoDTR/data/KITTI/object/training/image_2"
        # self.dataset_dir="/data3/rishubh/blender/KITTI/nusc_kitti_mini"
        # self.image_dir="/data3/rishubh/blender/KITTI/nusc_kitti_mini/image_2"
        split_txt_path=os.path.join(self.dataset_dir,"{}.txt".format("train"))
        # split_txt_path=os.path.join(self.dataset_dir,"{}.txt".format("mini_train"))
        self.id_list=[int(x.strip()) for x in open(split_txt_path).readlines()]
        # self.id_list=[x.strip() for x in open(split_txt_path).readlines()]
        #self.id_list=[120,171,253]
        self.root=root
        # self.label_setting=label_setting
        self.file_path=os.path.join(self.root)
        self.files_list=os.listdir(self.file_path)
        # print(self.files_list)
    
    def len(self):
        length=len(self.files_list)
        print(length)
        return (length)

    def getitem(self,idx):
        idx=self.id_list[idx]
        # print(idx)
        img_path=os.path.join(self.image_dir,'{:06d}.png'.format(idx))
        # img_path=os.path.join(self.image_dir,'{}.png'.format(idx))
        calib=Calibration(img_path.replace(".png",".txt").replace("image_2","calib"))
        # print(idx)
        # print(img_path)
        label_path=os.path.join(self.file_path,'{:06d}.txt'.format(idx))
        # label_path=os.path.join(self.file_path,'{}.txt'.format(idx))

        # print(label_path)
        assert (os.path.exists(label_path))
        labels=[]
        bboxes=[]
        # label_path=os.path.join(self.label_dir,'{:06d}.txt'.format(idx))
        # # print(label_path)
        for line_idx,line in enumerate(open(label_path,"r")):
            line=line.rstrip()
            line_parts=line.split(' ')
            obj_name=line_parts[0]

            # if int(float(line_parts[-1])) == 1:
            #     continue

            # print(obj_name)
            cat_id=int(CLASS_NAME_TO_ID[obj_name])
            # cat_id =int(NUSC_CLASS_NAME_TO_ID[obj_name])
            # print(cat_id)
            # if cat_id<=-99:
            #     continue
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
            # score=float(line_parts[15])
            object_label=[cat_id,x,y,z,h,w,l,ry]
            labels.append(object_label)
            bboxes.append(bbox)
            
            if line_idx > 10:
                break

        if len(labels)==0:
            labels=np.ones((1,8),dtype=np.float32)
            has_labels=False
        else:
            labels=np.array(labels,dtype=np.float32)
            has_labels=True
        if has_labels:
            labels[:,1:]=camera_to_lidar_box(labels[:,1:],calib.V2C,calib.R0,calib.P2)
            labels[:,1:]=lidar_to_camera_box(labels[:,1:],calib.V2C,calib.R0,calib.P2)
        return (labels,idx)
    

if __name__=='__main__':

    parser=ArgumentParser(description="Path for images and label files")
    parser.add_argument('--root',type=str, default="/data/srinjay/datasets/KITTI/label_2/")
    parser.add_argument('--render_path',type=str, default="/data/mp3d/smart4_vae_car_combined")
    parser.add_argument('--start_idx',type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    print(args)

    os.makedirs(args.render_path,exist_ok=True)
    dataset=iou_dataset(args.root)
    
    # start_idx = 2065
    print(dataset.id_list)
    start_idx = dataset.id_list.index(args.start_idx)
    print("start idx:",args.start_idx)

    for idx in range(start_idx, dataset.len()):
    # for idx in data_idxs:
    #for idx in range(start_idx, end_idx):
        cam_labs = dataset.getitem(idx)[0]
        data_idx = dataset.getitem(idx)[1]

        filename = '{:06d}'.format(data_idx)
        # filename = '{}'.format(data_idx)
        print("filename:",filename)
        for box_idx in range(len(cam_labs)):
            # if box_idx!=1:
            #     continue
            cls_id,location,dim,ry=cam_labs[box_idx][0],cam_labs[box_idx][1:4],cam_labs[box_idx][4:7],cam_labs[box_idx][7]
            # if location[2] < 2.0:  # The object is too close to the camera, ignore it during visualization
            #     continue

            if cls_id < 0:
                continue

            x,y,z=cam_labs[box_idx][1:4]
            h,w,l=cam_labs[box_idx][4:7]

            rel_angle = math.degrees(math.atan(x/z))
            
            orientation = -math.degrees(ry) + 90 + rel_angle # 75deg -> 105deg, 
            render(args.render_path, filename, box_idx, orientation)
            print("saved")


        
        




