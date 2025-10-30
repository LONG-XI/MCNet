
# install env
conda create -n MCNet python=3.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv-python
pip install timm
pip install easydict
pip install transforms3d
pip install h5py
pip install open3d



# run this for using our MCNet
CUDA_VISIBLE_DEVICES=0 python main-MCNet.py --test --output_points 10000 --num_multi_completion 100 --lamda 0.6 --ckpts ./experiments/trained_model/ckpt-MCNet.pth --config ./cfgs/ShapeNet34_models/MCNet.yaml --exp_name test_MCNet


# Note that: 

# changing output_points for different point resoultions.
# output_points can be any number. e.g. 16, 32, 64, 128, 5000, 8000, 10000, 300000, etc.


# changing num_multi_completion for different number of completion results.
# num_multi_completion can be any number. e.g. 5, 10, 20, 50, 100, 200, etc.


# changing lamda for different level of completion diversity.
# lamda is from 0.0 to 1.0.   e.g. 0.2, 0.54, 0.28 ,0.3, 0.4, 0.5, 0.62, etc. 
