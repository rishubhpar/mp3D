set -e

TORCH_VER=$(python3 -c "import torch;print(torch.__version__)")
CUDA_VER=$(python3 -c "import torch;print(torch.version.cuda)")


pushd ./networks/lib/ops/dcn
sh make.sh
rm -r build
popd

pushd ./networks/lib/ops/iou3d
sh make.sh
rm -r build
popd
