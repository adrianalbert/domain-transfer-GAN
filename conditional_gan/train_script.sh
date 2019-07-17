#!/bin/bash
# A sample Bash script by Ryan

for lambdaHeight in 0 1
do
  CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANlossLambdaHeight$lambdaHeight  --lambda_Height $lambdaHeight --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/LambdaHeight$LambdaHeight
done

for lambdaOcean in 0 1000
do
  CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANlossLambdaOcean$lambdaOcean  --lambda_Ocean $lambdaOcean --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/LambdaOcean$lambdaOcean
done

for lambdaSnow in 0 0.0001
do
  CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANlossLambdaSnow$lambdaSnow  --lambda_Snow $lambdaSnow --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/LambdaSnow$lambdaSnow
done


CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANfields0  --f0 0 --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/fields0
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANfields1  --f1 0 --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/fields1
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANfields2  --f2 0 --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/fields2
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANfields3  --f3 0 --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/fields3
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANfields4  --f4 0 --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/fields4
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANfields5  --f5 0 --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/fields5

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANbatch16  --batchSize 16 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/batch8
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANbatch64  --batchSize 64 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/cGANbatch64

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANngf8  --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 8 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/ngf8
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANngf32  --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 32 --nef 16 --ndf 32 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/ngf32

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANndf8  --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 8 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/ndf8
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ../datasets/livneh/ --checkpoints_dir ./checkpoints/ablation/ --name cGANndf16  --batchSize 32 --grid_size 64 --input_nc 6 --output_nc 1 --ngf 16 --nef 16 --ndf 16 --nlatent 16 --display_freq 1000 --tbpath tbviz/temp/ablation/ndf16
