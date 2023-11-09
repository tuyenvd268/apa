#!/usr/bin/env bash
. ./path.sh

cd ..
python src/run_gop.py \
    --config configs/gop_kaldi_labels.yaml
