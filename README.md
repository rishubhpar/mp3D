# MonoPlace3D

**MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection** (CVPR 2025) [[paper](https://arxiv.org/abs/2504.06801)]\
Rishubh Parihar, Srinjay Sarkar, Sarthak Vora, Jogendra Kundu, R. Venkatesh Babu.

<img src="assets/road-scene-teaser-fig.jpg" alt="vis" style="zoom:50%;" />


## Setup

Please refer to [INSTALL.md](./Placement/INSTALL.md) for installation and to [DATA.md](./Placement/DATA.md) for data preparation.


## Train Placement Network

Move to root and train the network with `$EXP_NAME`:

```sh
 cd Placement #MonoPlace3D_ROOT
 CUDA_VISIBLE_DEVICES=$GPUS python scripts/train.py --config=$CONFIG_PATH --experiment_name=$EXP_NAME
```

## Eval Placement Network

To evaluate on the validation set using checkpoint `$CHECKPOINT_PATH`:

```sh
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/eval.py --config=$CONFIG_PATH --gpu=0 --checkpoint_path=$CKPT_PATH --split_to_test=$SPLIT 
```
## Download Blender and ShapNet dataset 
```sh
wget -c https://download.blender.org/release/Blender2.78/blender-2.78-linux-glibc219-x86_64.tar.bz2

wget -c https://download.blender.org/release/Blender2.78/blender-2.78-linux-glibc219-x86_64.tar.bz2
```

## Render ShapeNet Cars
```sh
cd Rendering
blender <PATH TO BLENDER FILE> --background --python KITTI_shapenet_render_cars.py --root=$LABEL_PATH --render_path=$RENDERED_IMAGE_PATH --start_idx 0
```
## Render Car Shadows
```sh
cd Rendering
blender <PATH TO BLENDER FILE> --background --python KITTI_shapenet_render_shadows.py --root=$LABEL_PATH --render_path=$RENDERED_IMAGE_PATH --start_idx 0
```

## Place rendered ShapeNet/ControlNet Cars with Shadow in the scene
```sh
cd Rendering
python render_cars_with_shadow.py --label_path=$LABEL_PATH --img_path=$IMAGE_2_PATH --render_car_path=$RENDER_CAR_PATH --render_transformed_cars_path=$RENDER_CARS_TRANS_PATH --img_save_path=$IMG_PATH --start_idx 0
```

## Train Object Detection Network
Please refer to the code of [MonoDLE](https://github.com/xinzhuma/monodle) and [GUPNet](https://github.com/SuperMHP/GUPNet).

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{rishubh2025monoplace3D,
      title={MonoPlace3D: Learning 3D-Aware Object Placement for 3D Monocular Detection},
      author={Rishubh Parihar,Srinjay Sarkar,Sarthak Vora,Jogendra Kundu,R. Venkatesh Babu},
      journal={Conference on Computer Vision and Pattern Recognition},      
      year={2025}, 
}
 ```

## License

This project is released under the MIT License.
