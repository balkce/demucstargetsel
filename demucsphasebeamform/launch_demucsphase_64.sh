#!/bin/bash
# based on make_dns. with Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss, adefossez and balkce

#run the model 1 epochs of each user sub-set
python train.py \
  dset=dns_full \
  demucs.causal=1 \
  demucs.hidden=64 \
  demucs.resample=4 \
  bandmask=0.2 \
  shift=8000 \
  shift_same=True \
  noise_normalize=True \
  stft_loss=True \
  stft_sc_factor=0.1 stft_mag_factor=0.1 \
  segment=4.5 \
  paap=False \
  acoustic_loss=False \
  acoustic_loss_only=False \
  ac_loss_weight=1 \
  stft_loss_weight=0.3 \
  stride=0.5 \
  epochs=200 \
  dummy=demucsphase64 \
  num_prints=75 \
  batch_size=32 \
  eval_every=1

