## HydroGAN

First you need to prepare data as explained in `datasets` folder

### Training Augmented CycleGAN model
`CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --name augcgan_model`