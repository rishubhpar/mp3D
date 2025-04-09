import numpy as np
import os
import cv2  
from PIL import Image
import pandas as pd
import numpy as np
import math
from utils import compute_relative_angle_2d, Calibration, CLASS_NAME_TO_ID

def search(rel_orientation, depth, search_data):
    # Setting orientation search threshold
        epsilon = math.radians(5.0)
        sampled_orientation_data = []
        
        while len(sampled_orientation_data)==0:
            range_min = max(rel_orientation-epsilon, 0.0)  # Clipping the values beyond the orientation range
            range_max = min(rel_orientation+epsilon, 6.28)

            search_space = np.logical_and(search_data[:,1]>range_min, search_data[:,1]<=range_max)
            search_idxs = np.where(search_space)
            sampled_orientation_data = search_data[search_idxs]
            epsilon += math.radians(2.0)


        # finding stratified random closest match in scale
        k = 5
        if len(sampled_orientation_data)>5:
            match_idx = np.argpartition(np.abs(sampled_orientation_data[:,2] - depth), k)[:k]
            match_idx = np.random.choice(match_idx)
            final_idx = search_idxs[0][match_idx]
        else:
            final_idx = np.random.choice(search_idxs[0])

        return final_idx

# This function will load the label txt file and read the sequential 2D bounding box location to be used for object placement
def load_2D_box_locations(label_pt, calib_pt, database_path):
    boxes2d=[]
    box_paths=[]
    depths=[]

    calib = Calibration(calib_pt)
    P2 = calib.P2

    df = pd.read_csv(database_path)
    search_data = np.array(df.iloc[:,2:])

    # Iterating over bounding boxes present in a input image 
    idx = 0
    for line in open(label_pt,"r"): 
        line=line.rstrip()
        line_parts=line.split(' ')
        obj_name=line_parts[0]
        cat_id=int(CLASS_NAME_TO_ID[obj_name])

        if int(float(line_parts[-1]))==1:
            continue

        idx += 1

        if cat_id!=1:
            continue

        dim = float(line_parts[8]),float(line_parts[9]),float(line_parts[10])
        location = float(line_parts[11]),float(line_parts[12]),float(line_parts[13])
        ry=float(line_parts[14])
        depth = float(line_parts[13])

        # Compute relative 3D orientation
        rel_orientation = compute_relative_angle_2d(dim, location, ry, P2) 

        # Search for the matching car with nearest orientation
        final_idx = search(rel_orientation, depth, search_data)  

        # y_min, y_max, x_min, x_man
        bbox = np.array([float(line_parts[5]), float(line_parts[7]), float(line_parts[4]), float(line_parts[6])])

        ## LOAD paths
        car_path = df.loc[final_idx]['car_path']
        mask_path = df.loc[final_idx]['mask_path']
        ## LOAD paths

        depths.append(depth) 
        boxes2d.append(bbox)
        box_paths.append((car_path, mask_path)) 

    return boxes2d, box_paths, depths

# This function will process the car by removing the background and tightly cropping 2d bounding boxes 
def process_cars(img, img_mask):
    h,w = img_mask.shape[0], img_mask.shape[1]

    y_min_flag = False
    x_min, y_min, x_max, y_max = h, w, -1, -1 
    
    for yid in range(0, h):
        x_min_flag = False
        for xid in range(0, w):
            pix = img_mask[yid, xid] # ITerating over the mask and selecting the best index for masking 
            if (pix > 0):
                # Along the row  
                if (x_min_flag == False and xid < x_min):
                    x_min = xid
                    x_min_flag = True
                elif (xid > x_max):
                    x_max = xid

                # Along the columns
                if (y_min_flag == False):
                    y_min = yid
                    y_min_flag = True
                elif (yid > y_max):
                    y_max = yid     

    img_crop = img[y_min:y_max, x_min:x_max, :]
    single_mask = img_mask[y_min:y_max, x_min:x_max]  
    single_mask = single_mask > 120
    
    img_mask = np.zeros(img_crop.shape)
    img_mask[:,:,0] = single_mask
    img_mask[:,:,1] = single_mask
    img_mask[:,:,2] = single_mask 

    return img_crop, img_mask


# This function will load the cars from the given path and place the objects at the location 
def load_car_projections(car_mask_paths):
    loaded_cars = []

    # Iterating over all the boxes and loading cars for each of the corresponding box 
    for id in range(0, len(car_mask_paths)):
        car_path, car_mask_path = car_mask_paths[id]

        print("car image path: {}".format(car_path))
        car_loaded = np.array(Image.open(car_path))
        car_mask_loaded = np.array(Image.open(car_mask_path).convert('L'))
        
        print(f'car_img : {car_loaded.shape}, car_mask_image : {car_mask_loaded.shape}')
        car_img, car_mask = car_loaded, car_mask_loaded  
        car_img = cv2.cvtColor(car_img, cv2.COLOR_RGB2BGR)
        img_crop, img_mask = process_cars(car_img, car_mask) 
        loaded_cars.append([img_crop, img_mask])

    return loaded_cars 

# This function will place cars on the input image using the processed bounding boxes and car images 
def place_cars(img_base, bbox2d_list, car_imgs, depths):
    # If there are no cars to place, just return the same image as it is 
    if (len(bbox2d_list) == 0):
        return img_base
    
    img_edit = img_base.copy()

    # Sorting the boxes based on the depth information for each of the boxes, so we will start from the max depth and come towards the front for placement 
    depths, bbox2d_list, car_imgs = zip(*sorted(zip(depths, bbox2d_list, car_imgs), reverse=True, key = lambda x : x[0]))
    for id in range(0, len(bbox2d_list)):
        box_location = bbox2d_list[id]
        car_data = car_imgs[id]

        car_rgb, mask = car_data[0], car_data[1]
        x_min, x_max, y_min, y_max = int(box_location[0]), int(box_location[1]), int(box_location[2]), int(box_location[3]) 
        print("x min {}, x max {}, y min {}, y max {}".format(x_min, x_max, y_min, y_max))

        box_x_dim, box_y_dim = x_max-x_min, y_max-y_min
        if box_y_dim<=0 or box_x_dim<=0 or x_min<0 or y_min<0:
            print('Skipping this box location. Out of Image!')
            continue
    
        print("car rgb shape: {}, car mask shape: {}".format(car_rgb.shape, mask.shape))
        print("box x shape: {}, box y shape: {}".format(box_x_dim, box_y_dim))

        car_rgb = cv2.resize(car_rgb, (box_y_dim, box_x_dim))
        mask = cv2.resize(mask, (box_y_dim, box_x_dim))
        print("car image reshaped: {}, mask image reshaped: {}".format(car_rgb.shape, mask.shape))

        img_edit[x_min:x_max, y_min:y_max, :] = mask * car_rgb + (1-mask) * img_edit[x_min:x_max, y_min:y_max, :]

    return img_edit

# This function will render cars at the given location by loading the label paths and adding a bounding box on top of it 
def render_cars_main(label_path, image_path, image_save_path, calib_file_path, database_path, start_idx=0):

    img_ids = [filename.rstrip() for filename in open('./train.txt',"r")]
    start_idx = img_ids.index(f'{start_idx:06d}')
    
    # This function will iterate over all the images in the given folder and load the text file followed by placing cars in the inpainted images 
    for id in range(start_idx, len(img_ids)):
        img_pt = os.path.join(image_path, img_ids[id]+'.png')
        label_pt = os.path.join(label_path, img_ids[id]+ '.txt')
        calib_pt = os.path.join(calib_file_path, img_ids[id]+ '.txt')       
        img_save_pt = os.path.join(image_save_path, img_ids[id]+'.png')

        # Loading image 
        img = cv2.imread(img_pt) 

        # Loading bounding boxes
        bbox2d_list, car_mask_paths, depths = load_2D_box_locations(label_pt, calib_pt, database_path)

        if car_mask_paths==[]:
            cv2.imwrite(img_save_pt, img)
            print("Saving image at path: {}".format(img_save_pt))
            continue

        print("bbox list shape: {}".format(len(bbox2d_list))) 
        # Loading the cars after preprocessing 

        car_imgs = load_car_projections(car_mask_paths)

        # Editing the image by placing cars on the source image 
        img_edit = place_cars(img, bbox2d_list, car_imgs, depths)

        cv2.imwrite(img_save_pt, img_edit)
        print("Saving image at path: {}".format(img_save_pt))



def run_main():
    # Start rendering loop from a predefined image index
    start_idx=0
    
    ## Define Paths
    label_path = # Path to predicted label file from the placement model
    img_path =  # Path to KITTI3D image_2 folder
    img_save_path = # Define path for output directory

    calib_file_path =  # Define the path for calib folder in KITTI3D dataset
    database_path =  # Define the path for cutpaste .csv database 
    ##

    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    render_cars_main(label_path, img_path, img_save_path, calib_file_path, database_path, start_idx=start_idx)

if __name__ == "__main__":
    run_main()
