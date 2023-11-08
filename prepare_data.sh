#!/usr/bin/env bash
. ./path.sh

echo $KALDI_ROOT
python src/run_prepare_data.py --config configs/dataprep.yaml
