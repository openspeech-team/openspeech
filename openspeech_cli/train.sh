#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 hydra_train.py \
    dataset=ksponspeech \
    tokenizer=kspon_character \
    model=deepspeech2 \
    audio=mfcc \
    lr_scheduler=warmup_reduce_lr_on_plateau \
    trainer=gpu \
    trainer.batch_size=48 \
    criterion=ctc
