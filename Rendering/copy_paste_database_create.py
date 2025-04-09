import numpy as np
from PIL import Image
import os
import torch
from detectron2.structures import Boxes, pairwise_iou
from torchvision.utils import save_image
import csv

from utils import Calibration, camera_to_lidar_box, lidar_to_camera_box, CLASS_NAME_TO_ID, compute_relative_angle_2d
from instance_segmentation import get_predictor, run_predictor


class dataset():
    def __init__(self,root, dataset_dir="", image_dir=""):
        self.dataset_dir=dataset_dir
        self.image_dir=image_dir
        split_txt_path=os.path.join(self.dataset_dir,"{}.txt".format("train"))
        self.id_list=[int(x.strip()) for x in open(split_txt_path).readlines()]
        self.root=root
        self.file_path=os.path.join(self.root)
        self.files_list=os.listdir(self.file_path)
    
    def len(self):
        length=len(self.files_list)
        return (length)

    def getitem(self,idx):
        #idx=self.id_list[idx]
        img_path=os.path.join(self.image_dir,'{:06d}.png'.format(idx))
        calib=Calibration(img_path.replace(".png",".txt").replace("image_2","calib"))
        label_path=os.path.join(self.file_path,'{:06d}.txt'.format(idx))
        assert (os.path.exists(label_path))
        labels=[]
        bboxes=[]
        for line in open(label_path,"r"):
            line=line.rstrip()
            line_parts=line.split(' ')
            obj_name=line_parts[0]
        
            cat_id=int(CLASS_NAME_TO_ID[obj_name])
            truncated=float(line_parts[1])
            occluded=int(line_parts[2])
            alpha=float(line_parts[3])#angle

            if cat_id!=1 or truncated>0:
                continue

            # xmin, ymin, xmax, ymax
            bbox=[int(float(line_parts[4])),int(float(line_parts[5])),int(float(line_parts[6])),int(float(line_parts[7]))]

            # height,width,length(h,w,l)
            h,w,l=float(line_parts[8]),float(line_parts[9]),float(line_parts[10])

            #location (x,y,z) in camera coord.
            x,y,z=float(line_parts[11]),float(line_parts[12]),float(line_parts[13])

            ry=float(line_parts[14])#yaw angle(around Y-axis in camera coordinates)[-pi..pi]

            object_label=[cat_id,x,y,z,h,w,l,ry]
            labels.append(object_label)
            bboxes.append(bbox)
        if len(labels)==0:
            labels=np.ones((1,8),dtype=np.float32)
            has_labels=False
        else:
            labels=np.array(labels,dtype=np.float32)
            has_labels=True
        if has_labels:
            labels[:,1:]=camera_to_lidar_box(labels[:,1:],calib.V2C,calib.R0,calib.P2)
            labels[:,1:]=lidar_to_camera_box(labels[:,1:],calib.V2C,calib.R0,calib.P2)
        return (labels,bboxes,idx, calib.P2)

if __name__=='__main__':
    
    ## DEFINE PATHS
    # root = ""          # Path for Ground-Truth KITTI3D label folder
    # save_path = ""             # Path to save the Copy-Paste Car Database
    # dataset_dir = ""                    # Path to KITTI3D Dataset (should have calib_2, image_2, label_2 folders) 
    # image_dir = ""    # Path to KITTI3D Dataset image_2 folder
    ## DEFINE PATHS 

    car_directory = os.path.join(save_path, 'cars')
    mask_directory = os.path.join(save_path, 'mask')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(car_directory, exist_ok=True)
    os.makedirs(mask_directory, exist_ok=True)


    dataset=dataset(root, dataset_dir, image_dir)

    # Initialize instance segmentation
    predictor = get_predictor()

    # Define csv fields
    input_field = ['car_path', 'mask_path', 'orientation', 'relative orientation', 'scale']

    #scale area
    image_scale_area = 375*1242

    with open(os.path.join(save_path, 'data_stats.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(input_field)

        for idx in range(0, dataset.len()):
            cam_labs, bboxes, data_idx, P2 = dataset.getitem(idx)

            if len(bboxes)==0:
                continue

            filename = '{:06d}'.format(data_idx)
            image_path = os.path.join(image_dir, f'{filename}.png')
            image = np.array(Image.open(image_path))

            # Run detectron2 Instance Segmentation on the whole image
            pred_masks, boxes = run_predictor(predictor, image_path)

            for box_idx in range(len(cam_labs)):
                cls_id,location,dim,ry=cam_labs[box_idx][0],cam_labs[box_idx][1:4],cam_labs[box_idx][4:7],cam_labs[box_idx][7]
                depth = location[2]

                bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bboxes[box_idx]

                # if (bbox_xmax - bbox_xmin)>50 and (bbox_ymax - bbox_ymin)>50:
                if depth < 15:

                    query_box = Boxes([bboxes[box_idx]])
                    iou = pairwise_iou(query_box, boxes)[0]
                    match_index = torch.argmax(iou)

                    matched_pred_mask = pred_masks[match_index].unsqueeze(0).to(torch.float32)
            
                    query_area = torch.sum(matched_pred_mask).item() / query_box.area().item()
                    query_iou = iou[match_index].item()

                    if query_iou>0.85 and query_area>0.6:

                        print(f'Area:{query_area}, IOU:{iou[match_index].item()}, car code :{filename}_{box_idx}')

                        car_bbox = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]
                        matched_pred_mask = matched_pred_mask[:, bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]

                        # Compute car scale 
                        scale = depth

                        #compute relative orientation
                        rel_angle = compute_relative_angle_2d(dim, location, ry, P2)

                        car_bbox_path = os.path.join(car_directory, f'{filename}_{box_idx}.png')
                        car_bbox_mask_path = os.path.join(mask_directory, f'{filename}_{box_idx}.png')

                        car_bbox = Image.fromarray(car_bbox)
                        car_bbox.save(car_bbox_path)
                        save_image(matched_pred_mask, car_bbox_mask_path)


                        # save csv data
                        writer.writerow([car_bbox_path, car_bbox_mask_path, ry, rel_angle, scale])

    
        print(f'Done Image {filename}')

