#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss, adefossez, balkce

dnspath=/opt/DNS/DNS-Challenge

path=egs/dns
if [[ ! -e $path ]]; then
    mkdir -p $path
fi
python3 -m denoiser.audio $dnspath/corpus_multimic/noisy $path "train"
python3 -m denoiser.audio $dnspath/corpus_multimic_validtest/noisy $path "valid,test"

mv $path/test.json $path/test_full.json
mv $path/train.json $path/train_full.json
mv $path/valid.json $path/valid_full.json

python3 make_userinfo.py dset=dns dummy=dns_userinfo

python make_mini_dataset_specDNS.py dset=dns dummy=dns_spec

