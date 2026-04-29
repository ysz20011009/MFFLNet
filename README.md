# MFFLNet

- A Multi-Scale Fire Feature Learning Network for Video Fire Recognition

## Installation 

- To prepare the environment, please follow the following instructions.

  ```shell
  conda create --name openmmlab python=3.8 -y
    conda activate open-mmlab
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    pip install -U openmim
    mim install mmengine==0.10.5
    mim install mmcv==2.2.0
    mim install mmdet  # 可选
    mim install mmpose  # 可选
    git clone https://github.com/YSZ20011009/MFFLNet.git
    cd MFFLNet
    pip install -v -e .
    pip install pytorchvideo==0.1.5
  ```

## Datasets

- The used datasets are provided in [FSVR](https://github.com/ysz20011009/FSVR) and [LFVR](https://github.com/yunyi9/LFVR). The train/test splits in both two datasets follow the official procedure. 

## Model

- We provided the original pre-trained weights of ViT-B and the model weights of MFFLNet on the FSVR and LFVR datasets. Please visit the following website [link](https://pan.baidu.com/s/1OoTHk9-y_IUENzMLZKjlqg?pwd=mffl).

## Train

- The MFFLNet model file is located at `/MFFLNet/mmaction2-main/mmaction/models/backbones/mfflnet.py`. The configuration file is located at `/MFFLNet/mmaction2-main/configs/recognition/mfflnet-base-p16-res224_clip_8xb32-u8_kinetics710-rgb.py`

- Before training, please update the pretrained weight path and dataset path in the configuration file. 


- The model can be trained with the following command.

  ```shell
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  PORT=29209 CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_train.sh ./configs/recognition/mfflnet-base-p16-res224_clip_8xb32-u8_kinetics710-rgb.py 2 --work-dir ./workdir
  ```

## Test

- The model can be tested with the following command,change the path below.

  ```shell
  export CUBLAS_WORKSPACE_CONFIG=":4096:8"
  PORT=28756 CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_test.sh ./configs/recognition/mfflnet-base-p16-res224_clip_8xb32-u8_kinetics710-rgb.py ./workdir/epoch_x.pth 2 --work-dir ./workdir/test
  ```
- After downloading the fine-tuned MFFLNet model weights for the FSVR and LFVR datasets, you can reproduce the results reported in the paper using the following evaluation command.
  
  **FSVR**
  ```shell
  PORT=28756 CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_test.sh ./configs/recognition/mfflnet-base-p16-res224_clip_8xb32-u8_kinetics710-rgb.py ./epoch_fsvr.pth 2 --work-dir ./workdir/test_fsvr
  ```
  **LFVR**
  ```shell
  PORT=28756 CUDA_VISIBLE_DEVICES=0,1 bash ./tools/dist_test.sh ./configs/recognition/mfflnet-base-p16-res224_clip_8xb32-u8_kinetics710-rgb.py ./epoch_lfvr_split1.pth 2 --work-dir ./workdir/test_lfvr
  ```
- If you would like to learn more about the training or testing command arguments, please visit this [link](https://mmaction2.readthedocs.io/zh-cn/latest/user_guides/train_test.html).

## Acknowledgement

- This project is based on [MMAction2](https://github.com/open-mmlab/mmaction2). Thanks to the OpenMMLab team for their great work.
