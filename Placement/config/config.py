from easydict import EasyDict as edict
import os 
import numpy as np

cfg = edict()
cfg.obj_types = ['Car']
#cfg.obj_types = ['Car', 'Pedestrian', 'Cyclist']
cfg.jitter = True
cfg.threshold = 0.40
cfg.orientations = [[80,100], [-100, -80]]
cfg.orientation_thresh = [-5,5]
cfg.box_thresh = 6


## trainer
trainer = edict(
    gpu = 0,
    max_epochs = 120,
    disp_iter = 100,
    save_iter = 5,
    test_iter = 10,
    training_func = "train_mono_det",   
    test_func = "test_mono_det",
    evaluate_func = "evaluate_kitti_obj",
)

cfg.trainer=trainer

## path
path = edict()
path.data_path="" #path for KITTI training dataset
path.test_path="" #path for KITTI validation dataset
path.visualDet3D_path="" #project path
path.project_path="" #project path 
# print(path)


if not os.path.isdir(path.project_path):
    os.mkdir(path.project_path)
path.project_path = os.path.join(path.project_path)

# print(path.project_path)
# if not os.path.isdir(path.project_path):
#     os.mkdir(path.project_path)

path.log_path = os.path.join(path.project_path, "log")
if not os.path.isdir(path.log_path):
    os.mkdir(path.log_path)
# print("log path:",path.log_path)

path.checkpoint_path = os.path.join(path.project_path,"smart4_vae")
if not os.path.isdir(path.checkpoint_path):
    os.mkdir(path.checkpoint_path)
# print("checkpoint path:",path.checkpoint_path)

path.preprocessed_path=#path for preprocessed data
# print(path.preprocessed_path)
if not os.path.isdir(path.preprocessed_path):
    os.mkdir(path.preprocessed_path)
# print("preprocessed path:",path.preprocessed_path)

path.train_imdb_path=os.path.join(path.preprocessed_path,"training")
if not os.path.isdir(path.train_imdb_path):
    os.mkdir(path.train_imdb_path)
# print("train imdb path:",path.train_imdb_path)

path.val_imdb_path = os.path.join(path.preprocessed_path,"validation")
if not os.path.isdir(path.val_imdb_path):
    os.mkdir(path.val_imdb_path)
# print("val imdb path:",path.val_imdb_path)

path.viz_path=os.path.join(path.project_path,"viz_res")
if not os.path.isdir(path.viz_path):
    os.mkdir(path.viz_path)

cfg.path=path
cfg.viz=True


#vae huber detr
cfg.train_with_vae=False
cfg.train_with_huber=False
cfg.train_with_huber3d=False
cfg.train_with_detr=False
cfg.train_with_huberjitter=True

cfg.train_setting="smart4_vae"
assert cfg.train_setting in ["smart4","smart4_vae"]


## optimizer
optimizer = edict(
    type_name = 'adam',
    keywords = edict(
        lr        = 1e-4,
        weight_decay = 0,
    ),
    clipped_gradient_norm = 0.1
)
cfg.optimizer = optimizer
## scheduler
scheduler = edict(
    type_name = 'CosineAnnealingLR',
    keywords = edict(
        T_max     = cfg.trainer.max_epochs,
        eta_min   = 5e-6,
    )
)
cfg.scheduler = scheduler

## data
data = edict(
    batch_size = 12,
    num_workers = 8,
    rgb_shape = (288, 1280, 3),
    train_dataset = "kitti_monodataset",
    val_dataset   = "kitti_monodataset",
    test_dataset  = "kitti_monotestdataset",
    train_split_file = #path for training split file,
    val_split_file   = #path for validation split file),
)

data.augmentation = edict(
    rgb_mean = np.array([0.485, 0.456, 0.406]),
    rgb_std  = np.array([0.229, 0.224, 0.225]),
    cropSize = (data.rgb_shape[0], data.rgb_shape[1]),
    crop_top = 100,
)
data.train_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='PhotometricDistort', keywords=edict(distort_prob=1.0, contrast_lower=0.5, contrast_upper=1.5, saturation_lower=0.5, saturation_upper=1.5, hue_delta=18.0, brightness_delta=32)),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='RandomMirror', keywords=edict(mirror_prob=0.5)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]

# data.train_augmentation=[]

#Use this augmentation only when performing validation on actual train images not inpainted images.
# data.train_augmentation = [
#     edict(type_name='ConvertToFloat'),
#     edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
#     edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
#     edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
# ]


data.test_augmentation = [
    edict(type_name='ConvertToFloat'),
    edict(type_name='CropTop', keywords=edict(crop_top_index=data.augmentation.crop_top)),
    edict(type_name='Resize', keywords=edict(size=data.augmentation.cropSize)),
    edict(type_name='Normalize', keywords=edict(mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std))
]
cfg.data = data

## networks
detector = edict()
detector.obj_types = cfg.obj_types
detector.name = "monodtr"
detector.mono_backbone=edict(
)
head_loss = edict(
    fg_iou_threshold = 0.5,
    bg_iou_threshold = 0.4,
    L1_regression_alpha = 5 ** 2,
    focal_loss_gamma = 2.0,
    balance_weight   = [20.0],
    #balance_weight   = [20.0, 40, 40],
    regression_weight = [1, 1, 1, 1, 1, 1, 12, 1, 1, 0.5, 0.5, 0.5, 1], #[x, y, w, h, cx, cy, z, sin2a, cos2a, w, h, l]
)
head_test = edict(
    score_thr=0.45,
    cls_agnostic = False,
    nms_iou_thr=0.10,
    post_optimization=False
)

anchors = edict(
        {
            'obj_types': cfg.obj_types,
            'pyramid_levels':[3],
            'strides': [2 ** 3],
            'sizes' : [24],
            'ratios': np.array([0.5, 1, 2.0]),
            'scales': np.array([2 ** (i / 4.0) for i in range(16)]),
        }
    )

head_layer = edict(
    num_features_in=256,
    num_cls_output=len(cfg.obj_types)+1,
    num_reg_output=12,
    cls_feature_size=256,
    reg_feature_size=256,
)
detector.head = edict(
    num_regression_loss_terms=13,
    preprocessed_path=path.preprocessed_path,
    num_classes     = len(cfg.obj_types),
    anchors_cfg     = anchors,
    layer_cfg       = head_layer,
    loss_cfg        = head_loss,
    test_cfg        = head_test
)
detector.anchors = anchors
detector.loss = head_loss
cfg.detector = detector
