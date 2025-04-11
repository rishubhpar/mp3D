import numpy as np
import os
import cv2  
import sys
from PIL import Image
from scipy.ndimage import gaussian_filter
import argparse
from argparse import ArgumentParser, Namespace

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

# This function will load the label txt file and read the sequential 2D bounding box location to be used for object placement
def load_2D_box_locations(label_pt):
    boxes2d=[]
    box_idxs=[]
    depths=[]

    # Iterating over bounding boxes present in a input image 
    idx = 0
    for line in open(label_pt,"r"): 
        line=line.rstrip()
        line_parts=line.split(' ')
        obj_name=line_parts[0]
        cat_id=int(CLASS_NAME_TO_ID[obj_name])


        if int(float(line_parts[-1])) == 1:    
            continue

        idx += 1

        if cat_id!=1:
            continue
        
        # if cat_id<0:
        #     continue

        truncated=float(line_parts[1])
        occluded=int(line_parts[2])
        alpha=float(line_parts[3])#angle

        # ###
        # if truncated>0.7:
        #     print(' Truncated, skipping label')
        #     continue
        # ###
        
        # y_min, y_max, x_min, x_man
        bbox = np.array([float(line_parts[5]), float(line_parts[7]), float(line_parts[4]), float(line_parts[6])])
        
        # xmax, ymax, xmin, ymin
        # bbox=np.array([float(line_parts[6]),float(line_parts[7]),float(line_parts[4]),float(line_parts[5])])
        # print("bbox values: {}".format(bbox))
        # height,width,length(h,w,l)
        depth = float(line_parts[13])
        depths.append(depth) 
        boxes2d.append(bbox)

        # if idx>11:
        #     break

        box_idxs.append(str(idx-1) + '.png') 

        # # 
        # if idx==6:
        #     break
        # #

    #print(box_idxs)
    return boxes2d, box_idxs, depths

# This function will process the car by removing the background and tightly cropping 2d bounding boxes 
def process_cars(img, img_mask,img_mask_shadow):
    
    
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

    # print("x min {}, x max {}, y min {}, y max {}".format(x_min, x_max, y_min, y_max))
    img_crop = img[y_min:y_max, x_min:x_max, :]
    # print("img crop shape: {}".format(img_crop.shape))
    single_mask = img_mask[y_min:y_max, x_min:x_max]  #this one is mask from car image
    # single_mask = img_mask_shadow[y_min:y_max, x_min:x_max]  # this one is mask from car+shadow image 
    single_mask = single_mask > 1
    
    img_mask = np.zeros(img_crop.shape)
    img_mask[:,:,0] = single_mask
    img_mask[:,:,1] = single_mask
    img_mask[:,:,2] = single_mask 
    # center_locs = ((x_min+x_max)/2,(y_min+y_max)/2)
    center_locs = [(x_min+x_max)/2,(y_min+y_max)/2]
    
    return img_crop, img_mask , center_locs


# This function will load the cars from the given path and place the objects at the location 
def load_car_projections(render_car_path, render_transformed_cars, car_index_names , cars_shadow_fld):
    print("render car path: {}".format(render_car_path))
    loaded_cars = []
    # Iterating over all the boxes and loading cars for each of the corresponding box 
    for id in range(0, len(car_index_names)):
        # if id!=2:
        #     continue
        car_shapenet_path = os.path.join(render_car_path, car_index_names[id]) 
        ###CHANGE HERE
        car_transformed_path = os.path.join(render_transformed_cars, car_index_names[id])
        car_shadow_path = os.path.join(cars_shadow_fld , car_index_names[id])

        print("car image path: {}".format(car_shapenet_path))
        car_loaded_shapenet = np.array(Image.open(car_shapenet_path))
        car_loaded_tform = np.array(Image.open(car_transformed_path))
        # car_mask_shadows = car_loaded_tform[...,3] 
        print ("#######",car_shadow_path)
        car_shadow = np.array(Image.open(car_shadow_path)) [:,:,3]
        
        
        print(car_shadow.shape)

        car_shadow_img = np.zeros((car_shadow.shape[0],car_shadow.shape[1],3))
        car_shadow_img[:,:,0] = car_shadow
        car_shadow_img[:,:,1] = car_shadow
        car_shadow_img[:,:,2] = car_shadow 
        
        # car_shadow = car_shadow.astype(np.uint8)
        # car_shadow = cv2.cvtColor(car_shadow,cv2.COLOR_RGB2BGR)
        
        print("############",np.unique(car_shadow_img))
        print("############",car_shadow_img.shape)
        # car_shadow_save = car_shadow.astype(np.uint8) 
        car_loaded_tform = cv2.cvtColor(car_loaded_tform, cv2.COLOR_RGB2BGR)
        print(f'car_img : {car_loaded_shapenet.shape}, car_transformed : {car_loaded_tform.shape}')
        # car_img, car_mask  = car_loaded_tform, car_loaded_shapenet[...,3] 
        car_img, car_mask  = car_loaded_tform, car_loaded_shapenet[...,3] 
        # car_img = cv2.cvtColor(car_img, cv2.COLOR_RGB2BGR)
        
        
        # car mask = car loaded shapenet = car shapenet path = render car path 
        # car img = car_loaded_tform = car transformed path = render_transformed_cars



        # cv2.imwrite('./debug/car_img.png', car_img)
        # cv2.imwrite('./debug/car_mask.png', car_mask)
        # exit()
        # cv2.imwrite('./debug/car_shapenet.png', car_loaded_shapenet)
        # print("loaded car image shape: {}".format(car_img.shape))
        # img_crop, img_mask , center_locs = process_cars(car_img, car_mask, car_mask_shadows)
        img_crop, img_mask , center_locs = process_cars(car_img, car_mask, car_shadow_img)
        # cv2.imwrite('./debug/car_img_crop.png', img_crop)
        # cv2.imwrite('./debug/car_mask_crop.png', img_mask * 255) 
        # cv2.imwrite('./debug/car_loade_tform.png',car_shadow_img)
        
        print("center locs:",center_locs)
        # exit()
        loaded_cars.append([img_crop, img_mask , center_locs ,car_shadow_img ])
        # loaded_cars.append([img_crop, img_mask , center_locs ,car_loaded_tform ])
    return loaded_cars 

# def load_car_projections(car_mask_paths):
#     loaded_cars = []

#     # Iterating over all the boxes and loading cars for each of the corresponding box 
#     for id in range(0, len(car_mask_paths)):
#         car_path, car_mask_path = car_mask_paths[id]

#         print("car image path: {}".format(car_path))
#         car_loaded = np.array(Image.open(car_path))
#         car_mask_loaded = np.array(Image.open(car_mask_path).convert('L'))
        
#         print(f'car_img : {car_loaded.shape}, car_mask_image : {car_mask_loaded.shape}')
#         car_img, car_mask = car_loaded, car_mask_loaded  
#         car_img = cv2.cvtColor(car_img, cv2.COLOR_RGB2BGR)
#         img_crop, img_mask = process_cars(car_img, car_mask) 
#         loaded_cars.append([img_crop, img_mask])

#     return loaded_cars 


# This function will place cars on the input image using the processed bounding boxes and car images 
def place_cars(img_base, bbox2d_list, car_imgs, depths):
    # If there are no cars to place, just return the same image as it is 
    if (len(bbox2d_list) == 0):
        return img_base
    
    img_edit = img_base.copy()

    # Sorting the boxes based on the depth information for each of the boxes, so we will start from the max depth and come towards the front for placement 
    #temp = sorted(zip(depths, bbox2d_list, car_imgs))
    print(depths)
    depths, bbox2d_list, car_imgs = zip(*sorted(zip(depths, bbox2d_list, car_imgs), reverse=True, key = lambda x : x[0]))
    #ids = [-1,-2]
    for id in range(0, len(bbox2d_list)):
        if id!=1:
            continue
    #for idx in ids:
        #id = 1
        #id = ids[idx]
        box_location = bbox2d_list[id]
        car_data = car_imgs[id]

        car_rgb, mask , global_center_locs , only_shadow = car_data[0], car_data[1] ,car_data[2] ,car_data[3]
        
        # cv2.imwrite('./debug/only_shadow.png',only_shadow)
        #applying gaussian to only_shadow image
        # print(only_shadow.shape)
        sigma=4
        only_shadow=gaussian_filter(only_shadow,sigma=sigma)
        # cv2.imwrite("./debug/only_shadow_gaussian_"+str(sigma)+".png",only_shadow)
        
        # exit()
        
        x_min, x_max, y_min, y_max = int(box_location[0]), int(box_location[1]), int(box_location[2]), int(box_location[3]) 
        print("x min {}, x max {}, y min {}, y max {}".format(x_min, x_max, y_min, y_max))

        y_min += 80
        y_max += 80
       
     
        
        y_min -= 50
        y_max -= 50
        box_center=[(x_min+x_max)/2,(y_min+y_max)/2]
        y_min += 50
        y_max += 50
        x_min -= 30
        x_max -= 30

     

        
        
        box_x_dim, box_y_dim = x_max-x_min, y_max-y_min
        #print(box_x_dim, box_y_dim)
        if box_x_dim<=0 or box_y_dim<=0:
            print('Skipping this bounding box - Out of Image coordinates')
            continue

        print("car rgb shape: {}, car mask shape: {}".format(car_rgb.shape, mask.shape))
        print("box x shape: {}, box y shape: {}".format(box_x_dim, box_y_dim))
        print("only shadow:" , only_shadow.shape)
        
        resize_factor = (box_y_dim/car_rgb.shape[1] , box_x_dim/car_rgb.shape[0])
        print("resize factor:" , resize_factor)
        
        print("gc anf box cent:",global_center_locs,box_center)
        
        start_shadow_loc = [ int(box_center[0])-int(global_center_locs[0]* resize_factor[0]),int(box_center[1])-int(global_center_locs[1]* resize_factor[1])]

        car_rgb = cv2.resize(car_rgb, (box_y_dim, box_x_dim))
        mask = cv2.resize(mask, (box_y_dim, box_x_dim))
        only_shadow = cv2.resize(only_shadow , (int(only_shadow.shape[0]*resize_factor[1]) , int(only_shadow.shape[1]*resize_factor[0])))
        
        global_shadow_image = np.zeros((img_edit.shape))
        
        x_min_shadow = max(0 , start_shadow_loc[0]) 
        x_max_shadow = only_shadow.shape[0] + start_shadow_loc[0]
        y_min_shadow = max(0 , start_shadow_loc[1])
        y_max_shadow = only_shadow.shape[1] + start_shadow_loc[1]
        
        print("limits:",x_min_shadow,x_max_shadow,y_min_shadow,y_max_shadow)
        print("corners of the only shadow iamge:",-min (0 , start_shadow_loc[0]) , -min (0 , start_shadow_loc[1]), only_shadow.shape, global_shadow_image.shape)
        
        print("region of inserting shdaow :",global_shadow_image[x_min_shadow:x_max_shadow,y_min_shadow:y_max_shadow].shape)  
        print("taking only shadow prt of image :",only_shadow[-min (0 , start_shadow_loc[0]) : , -min (0 , start_shadow_loc[1]) : ] .shape)
        
        # global_shadow_image[x_min_shadow:x_max_shadow,y_min_shadow:y_max_shadow] = only_shadow[-min (0 , start_shadow_loc[0]) : , -min (0 , start_shadow_loc[1]) : ]
        
        try:
            global_shadow_image[x_min_shadow:x_max_shadow,y_min_shadow:y_max_shadow] = only_shadow[-min (0 , start_shadow_loc[0]) : , -min (0 , start_shadow_loc[1]) : ] 
        except:
            only_shadow = only_shadow[-min (0 , start_shadow_loc[0]) : , -min (0 , start_shadow_loc[1]) : ] 
            # print("123:",only_shadow.shape)
            # print(x_max_shadow-x_min_shadow,y_max_shadow-y_min_shadow)
            # print("124:",only_shadow[0:(x_max_shadow-x_min_shadow),0:(y_max_shadow-y_min_shadow)].shape)
            gsr=global_shadow_image[x_min_shadow:x_max_shadow,y_min_shadow:y_max_shadow]
            # print(only_shadow [0:global_shadow_image.shape[0]-1,0:global_shadow_image.shape[1]-1].shape)
            global_shadow_image[x_min_shadow:x_max_shadow,y_min_shadow:y_max_shadow] = only_shadow [0:gsr.shape[0],0:gsr.shape[1]]
            
        
        global_shadow_mask=global_shadow_image.copy()
        global_shadow_mask[global_shadow_mask > 0]=255
        global_shadow_mask /= 255
        # print("###",np.unique(global_shadow_mask))
        # print("###",np.unique(global_shadow_image))
        # print("###",np.unique(only_shadow))
        
        
        
        
        # img_edit = global_shadow_mask * global_shadow_image + (1-global_shadow_mask) * img_edit
        global_shadow_image /= 255
        global_shadow_image *= 1
        img_edit = global_shadow_image * np.zeros(img_edit.shape) + (1-global_shadow_image) * img_edit
        
        
        print("car image reshaped: {}, mask image reshaped: {}".format(car_rgb.shape, mask.shape))

        # cv2.imwrite('./debug/in_car_rgb.png', car_rgb)
        # cv2.imwrite('./debug/in_mask.png', mask * 100)


        # print("img base shape: {}".format(img_edit.shape))
        # img_edit[x_min:x_max, y_min:y_max, :] = mask * car_rgb + (1-mask) * img_base[x_min:x_max, y_min:y_max, :]
        # print("mask unique: ", np.unique(mask))
        # img_edit[x_min:x_max, y_min:y_max, :] = (1-mask) * img_base[x_min:x_max, y_min:y_max, :]
        img_edit[x_min:x_max, y_min:y_max, :] = mask * car_rgb + (1-mask) * img_edit[x_min:x_max, y_min:y_max, :]
        
        # cv2.imwrite('./debug/global_shadow_image.png', global_shadow_image)
        # cv2.imwrite('./debug/global_shadow_mask.png', global_shadow_mask)
        # cv2.imwrite('./debug/img_edit.png', img_edit)
        print("saved")
        
        # exit()
        
        #img_edit[x_min:x_max, y_min:y_max, :] = mask + (1-mask) * img_edit[x_min:x_max, y_min:y_max, :]

    return img_edit

# This function will render cars at the given location by loading the label paths and adding a bounding box on top of it 
def render_cars_main(label_path, image_path, image_save_path, render_car_path, render_transformed_cars_path,shadow_cars_path, start_idx):

    img_ids = [filename.rstrip() for filename in open('../train.txt',"r")]
    start_idx = img_ids.index(f'{start_idx:06d}')
    
    # print("#####################",start_idx)

    # data_idxs = [44,120,141,286]
    # img_ids = ['{:06d}'.format(idx) for idx in data_idxs]

    
    for id in range(start_idx, len(img_ids)):
    #for id in range(0, len(img_ids)):
        img_pt = os.path.join(image_path, img_ids[id]+'.png')
        label_pt = os.path.join(label_path, img_ids[id]+ '.txt')
        
        cars_img_fld = os.path.join(render_car_path, img_ids[id])
        cars_img_transformed_fld = os.path.join(render_transformed_cars_path, img_ids[id])
        
        cars_shadow_fld= os.path.join(shadow_cars_path , img_ids[id])
        
        img_save_pt = os.path.join(image_save_path, img_ids[id]+'.png')

        # Loading image 
        img = cv2.imread(img_pt) 

        # Loading bounding boxes
        bbox2d_list, car_index_names, depths = load_2D_box_locations(label_pt)

        print(bbox2d_list)

        if car_index_names==[]:
            cv2.imwrite(img_save_pt, img)
            print("Saving image at path: {}".format(img_save_pt))
            continue

        print("bbox list shape: {}".format(len(bbox2d_list))) 
        # Loading the cars after preprocessing 
        car_imgs = load_car_projections(cars_img_fld, cars_img_transformed_fld, car_index_names , cars_shadow_fld)
        # Editing the image by placing cars on the source image 
        img_edit = place_cars(img, bbox2d_list, car_imgs, depths)

        cv2.imwrite(img_save_pt, img_edit)
        print("Saving image at path: {}".format(img_save_pt))

        exit()


def main(args):
    # # start_idx=5097

    # #label_path = './label_2'
    # label_path = '/data3/rishubh/blender/smart4_vae_car_combined'

    # img_path = '/data3/rishubh/MonoDTR/data/KITTI2/object/training/image_2'
    # #img_path = '/data3/rishubh/MonoDTR/data/KITTI/object/training/image_2_refined'

    # #render_car_path = './rendered_cars'
    
    # # render_car_path = '/data3/rishubh/blender/rendered_cars_cutpaste_smart4_vae_car_combined/mask'
    # render_car_path = '/data3/rishubh/blender/rendered_cars_smart4_vae_car_combined_rectified'
    # # render_car_path = '/data3/rishubh/blender/controlnet/rendered_cars_smart4_vae_car_combined'
    
    
    # #change this for controlnet and copy paste
    # # render_transformed_cars_path = '/data3/rishubh/blender/rendered_cars_smart4_vae_car_combined_rectified' # same as render_car_path
    # # render_transformed_cars_path = '/data3/rishubh/blender/controlnet/rendered_cars_smart4_vae_car_combined_rectified'
    # # render_transformed_cars_path = '/data3/rishubh/blender/rendered_cars_cutpaste_smart4_vae_car_combined/cars'
    # render_transformed_cars_path= '/data3/rishubh/blender/rendered_cars_smart4_vae_car_combined_rectified'
    
    
    # shadow_cars_path = '/data3/rishubh/blender/rendered_cars_shadow_smart4_vae_labels_no_artifacts'
    
    # # render_transformed_cars_path = '/data3/rishubh/blender/rendered_cars_smart4_vae_car_combined_rectified'
    # # render_transformed_cars_path= '/data3/rishubh/blender/rendered_shapenet_shadowcars_smart4_vae_labels'
    # #render_transformed_cars_path = './controlnet/rendered_cars_smart4_car_combined_rectified'
    # #render_transformed_cars_path = './controlnet/rendered_cars_smart4_car_combined_rectified'

    # img_save_path = '/data3/rishubh/blender/KITTI/fig-12/shapenet/'
    # #img_save_path = './controlnet/image_2_edited_smart4_car_combined_rectified/'

    # # label_path = "/data3/rishubh/blender/paper_figs/fixed_slide_8/labels_2"
    # # img_path = "/data3/rishubh/MonoDTR/data/KITTI2/object/training/image_2"
    # # render_car_path = "/data3/rishubh/blender/paper_figs/slide_8_eccv_viz/rendered_cars_labels_2"
    # # render_transformed_cars_path = "/data3/rishubh/blender/paper_figs/slide_8_eccv_viz/controlnet/rendered_cars_labels_2"
    # # shadow_cars_path = "/data3/rishubh/blender/paper_figs/slide_8_eccv_viz/rendered_cars_shadow_labels_2"
    # # img_save_path = "/data3/rishubh/blender/paper_figs/slide_8_eccv_viz/rendered_images_labels_2"

   

    # if not os.path.exists(args.img_save_path):
    #     os.mkdir(args.img_save_path)

    render_cars_main(args.label_path, args.img_path, args.img_save_path, args.render_car_path, args.render_transformed_cars_path, args.shadow_cars_path, start_idx=args.start_idx)

if __name__ == "__main__":
    
    parser=ArgumentParser(description="Path for images and shadows files")
    parser.add_argument('--label_path',type=str, default="/data/mp3d/smart4_vae_car_combined")
    parser.add_argument('--img_path',type=str, default="/data/mp3d/smart4_vae_car_combined")
    parser.add_argument('--render_car_path',type=str, default="/data/mp3d/rendered_cars_smart4_vae_car_combined_rectified")
    parser.add_argument('--render_transformed_cars_path',type=str, default="/data/mp3d/rendered_cars_smart4_vae_car_combined_rectified")
    parser.add_argument('--img_save_path',type=str, default="/data/mp3d/rendered_cars_smart4_vae_car_combined_rectified")
    parser.add_argument('--start_idx',type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    print(args)

    main(args)
