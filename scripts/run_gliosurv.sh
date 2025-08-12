#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
	configs/gliosurv.yaml \
	--model_name="GlioSurv"
