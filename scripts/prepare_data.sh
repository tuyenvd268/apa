#!/usr/bin/env bash
. ./path.sh

cd ..
python src/run_prepare_data.py \
    --config configs/dataprep.yaml
