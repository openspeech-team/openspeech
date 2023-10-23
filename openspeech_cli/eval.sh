#!/bin/bash

python3 hydra_eval.py \
    tokenizer=kspon_character \
    model=deepspeech2 \
    audio=mfcc \
#    eval.checkpoint_path=/home/patrick/openspeech/openspeech_cli/outputs/2023-10-11/18-16-11/logs/default/version_0/checkpoints/0_0.ckpt
