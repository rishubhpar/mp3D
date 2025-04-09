from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

#get the predictor
def get_predictor():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor

def run_predictor(predictor, im_path):
    im = cv2.imread(im_path)

    #get prediction
    outputs = predictor(im)
    
    #get boxes and masks
    ins = outputs["instances"]
    pred_masks = ins.pred_masks.cpu()
    boxes = ins.pred_boxes.to('cpu')

    return pred_masks, boxes   
