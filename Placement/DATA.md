## Data Preparation

Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:


```
#mp3D_ROOT
  |data/
    |KITTI/
      |object/			
        |training/
          |calib/
          |image_2/
          |label_2/
          |velodyne/
        |testing/
          |calib/
          |image_2/
```

You can modify the path in ./Placement/config/config.py (for train / val split), and then run the preparation script:

```sh
python scripts/imdb_precompute_3d.py --config=$CONFIG_PATH
python scripts/depth_gt_compute_3d.py --config=$CONFIG_PATH
```