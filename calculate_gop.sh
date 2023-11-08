#!/usr/bin/env bash
. ./path.sh

echo $KALDI_ROOT
python src/run_gop.py --config configs/gop_kaldi_labels.yaml
