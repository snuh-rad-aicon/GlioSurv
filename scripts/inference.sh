#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py \
	configs/gliosurv.yaml \
	--model_name="GlioSurv" \
	--pretrain="path/to/gliosurv_model/checkpoint.pth.tar" \
	--gpu=0 \
	--vis_batch_size=1