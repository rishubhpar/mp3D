import importlib
import fire
import os
import copy
import torch

#local imports
from preprocess.path_init_ import *
from networks.utils.registry import DETECTOR_DICT, DATASET_DICT, PIPELINE_DICT
from utils.utils import cfg_from_file
import data.kitti.dataset
from networks.pipelines import evaluate_kitti_obj
from visualization import *


print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(config:str="config/config.py",gpu:int=0,checkpoint_path:str="retinanet_79.pth",split_to_test:str='validation'):
    config="config/config.py"
    split_to_test="training"
    cfg=cfg_from_file(config)
    cfg.trainer.gpu=0#1,2,3
    torch.cuda.set_device(cfg.trainer.gpu)

    #dataset and dataloader
    split="validation"
    dataset=DATASET_DICT[cfg.data.train_dataset](cfg)#3769
    print(len(dataset))
    # #model
    # # Create the model and load weights
    # if cfg.train_setting=="smart4":
    #     checkpoint_path=""#path for weight file of smart4 setting
    # elif cfg.train_setting=="smart4_vae":
    #     checkpoint_path=""#path for weight file of smart4_vae setting

    detector=DETECTOR_DICT[cfg.detector.name](cfg.detector)
    state_dict=torch.load(checkpoint_path,map_location="cuda")
    new_dict=state_dict.copy()
    detector.load_state_dict(new_dict,strict=False)
    detector=detector.cuda()
    detector.eval()

    print(checkpoint_path)

    # eval
    if 'evaluate_func' in cfg.trainer:
        evaluate_detection=PIPELINE_DICT[cfg.trainer.evaluate_func]
        print("Found evaluate function")
    else:
        raise KeyError("evluate_func not found in Config")

    print(evaluate_detection)
    evaluate_detection(cfg,detector,dataset,None,0,result_path_split=split_to_test)

    # #viz
    # if cfg.viz:
    #     split_txt_path="/scratch/kitti-detection/ImageSets/val.txt"
    #     image_idx=[int(x.strip()) for x in open(split_txt_path).readlines()]
    #     dataset_dir="/scratch/kitti-detection/"
    #     sub_folder="training"
    #     calib_dir=os.path.join(dataset_dir,sub_folder,"calib")


    #     for idx in range(len(os.listdir(os.path.join(cfg.path.preprocessed_path,split,'data')))):
    #         # print(idx)
    #         img=dataset[idx]["image"]
    #         P2=dataset[idx]["original_P"]
    #         sample_id=int(image_idx[idx])
    #         calib_file=os.path.join(calib_dir,'{:06d}.txt'.format(sample_id))
    #         calib=Calibration(calib_file)
    #         label_path=os.path.join(os.path.join(cfg.path.preprocessed_path,split,'data'),'{:06d}.txt'.format(idx))
    #         bboxes=[]
    #         labels=[]
    #         for line in open(label_path,"r"):
    #             line=line.rstrip()
    #             line_parts=line.split(' ')
    #             obj_name=line_parts[0]
    #             # print(obj_name)
    #             cat_id=int(CLASS_NAME_TO_ID[obj_name])
    #             if cat_id<=-99:
    #                 continue
    #             truncated=int(float(line_parts[1]))
    #             occluded=int(line_parts[2])
    #             alpha=float(line_parts[3])#angle
    #             # xmin, ymin, xmax, ymax
    #             bbox=np.array([float(line_parts[4]),float(line_parts[5]),float(line_parts[6]),float(line_parts[7])])
    #             # height,width,length(h,w,l)
    #             h,w,l=float(line_parts[8]),float(line_parts[9]),float(line_parts[10])
    #             #location (x,y,z) in camera coord.
    #             x,y,z=float(line_parts[11]),float(line_parts[12]),float(line_parts[13])
    #             ry=float(line_parts[14])#yaw angle(around Y-axis in camera coordinates)[-pi..pi]
    #             object_label=[cat_id,x,y,z,h,w,l,ry]
    #             labels.append(object_label)
    #             bboxes.append(bbox)
    #         if len(labels)==0:
    #             labels=np.zeros((1,8),dtype=np.float32)
    #         else:
    #             labels=np.array(labels,dtype=np.float32)
    #         # print(labels)

    #         labels[:,1:]=camera_to_lidar_box(labels[:,1:],calib.V2C,calib.R0,P2)  
    #         minX=boundary['minX']
    #         maxX=boundary['maxX']
    #         minY=boundary['minY']
    #         maxY=boundary['maxY']
    #         minZ=boundary['minZ']
    #         maxZ=boundary['maxZ']
    #         label_x=(labels[:, 1] >= minX) & (labels[:, 1] < maxX)
    #         label_y=(labels[:, 2] >= minY) & (labels[:, 2] < maxY)
    #         label_z=(labels[:, 3] >= minZ) & (labels[:, 3] < maxZ)
    #         mask_label=label_x & label_y & label_z
    #         labels=labels[mask_label]
    #         labels[:,1:]=lidar_to_camera_box(labels[:,1:],calib.V2C,calib.R0,P2)
    #         cam_labs=labels


    #         for box_idx in range(len(cam_labs)):
    #             # print(len(cam_labs))
    #             cls_id,location,dim,ry=cam_labs[box_idx][0],cam_labs[box_idx][1:4],cam_labs[box_idx][4:7],cam_labs[box_idx][7]
    #             # if location[2] < 2.0:  # The object is too close to the camera, ignore it during visualization
    #             #     continue
    #             if cls_id < 0:
    #                 continue
    #             x,y,z=cam_labs[box_idx][1:4]
    #             h,w,l=cam_labs[box_idx][4:7]
    #             corners_3d=compute_box_3d(dim,location,ry)
    #             corners_2d=project_to_image(corners_3d,calib.P2)
    #             img=draw_box_3d(img,corners_2d)
    #             # black_imgs=draw_box_3d(black_imgs,corners_2d)
    #         # print(cfg.path.viz_path+"/"+str(idx)+".png")
    #         cv2.imwrite(cfg.path.viz_path+"/"+str(idx)+".png",img)
            
    print('finish')

if __name__ == '__main__':
    fire.Fire(main)