## HydroGAN

First you need to prepare data as explained in `datasets` folder

### Training Augmented CycleGAN model
`CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --name augcgan_model`

###short
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --name augcgan_model --batchSize 32 --grid_size 64 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp

###access tbviz
tensorboard --logdir=tbviz/temp/




