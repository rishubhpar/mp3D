This document lists the steps to train and evaluate our placement code.

INSTALLATION

#Environment Prerequisites
Ubuntu 16.04+
Python 3.6+
NumPy 1.19
PyTorch (tested on 1.13.1) with cuda 11.7

#Dependencies
Install all Prerequisites with pip install -r requirement.txt

#Build ops
Run the command "bash make.sh"


DATA
Please download the KITTI dataset from "https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d". Make sure the data directory has the following structure. 

#ROOT
|data/
|KITTI/
			
|training/
|calib/
|image_2/
|label_2/
|velodyne/
|train.txt
|val.txt   


The KITTI dataset has the validation and training samples in the same directory. Please refer to train.txt and val.txt for the training and validation sample indices.



MODEL TRAIN
1: Change all the paths in the config/config.py (for train/val split) and run the preprocessing script. Please add the config file path in "imdb_precompute_3d.py"
2: Obtain the inpainted images by using the stable diffusion model inpainting model from HuggingFace and save them in a separate folder with the same naming convention as the KITTI images. This step is will give us the unpainted version of KITTI training dataset
3: Extract the depth images for the inpainted images using Depth Prediction Transformer (DPT). Resize the depth images resize them to (320,72) to keep in accordance with the depth transformer module , save them according to the path mentioned in  "data/kitti/dataset/monodataset.py". Please keep the naming convention as "P2000000.png" to "P2003711.png" for the 3712 train images as depth is only required during training.
4: Please change the path of "self.image_2" to the path of inpainted images in "kitti/kittidata.py". Run "preprocess/imdb_precompute_3d.py" to get the preprocssed data.
5: Run "train.py" after changing the save paths according to your directory structure.


MODEL EVALUATION
1: In order to evaluate the placement method first run the training images through the preprocessing script by changing the path of "self.image_2" to "training/image_2" in "kitti/kittidata.py" and then running "preprocess/imdb_precompute_3d.py". Change the "checkpoint_path" in "eval.py" accordingly.
2: Change the score threshold and NMS threshold values to change the number of output bounding box. We have included the default value of the score threshold  and NMS threshold in the config file.
3: During inference, the model will save the label file with new boxes at the path defined in config.py.   