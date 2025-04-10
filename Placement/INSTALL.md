## Installation

### Prerequisites

- Ubuntu 16.04+
- Python 3.6+
- NumPy 1.19
- PyTorch (tested on 1.4.0)

### Installation

Our code is based on [MonoDTR](https://github.com/KuanchihHuang/MonoDTR), you can refer to [setup](https://github.com/KuanchihHuang/MonoDTR) for details. This repo is mainly developed with a single A5000 GPU on our local environment (python=3.8, cuda=11.7, pytorch=1.13), and we recommend you to use anaconda to create a vitural environment:

```bash
conda create -n mp3d python=3.8
conda activate mp3d
```

Install PyTorch:

```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

and other requirements:
```bash
pip install -r requirement.txt
```

Lastly, build ops (deform convs and iou3d)
```bash
cd Placement
./make.sh
```