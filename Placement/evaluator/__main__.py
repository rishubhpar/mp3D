from fire import Fire
from .evaluate import evaluate

def main(evaluator='kitti_obj', **kwargs):
    if evaluator.lower() == 'kitti_obj':
        texts = evaluate(**kwargs)
        for text in texts:
            print(text)
        return 
Fire(main)