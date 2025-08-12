#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
	configs/vit3d.yaml \
	--model_name="ViT3D"
