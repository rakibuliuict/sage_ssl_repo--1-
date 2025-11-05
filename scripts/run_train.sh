#!/usr/bin/env bash
set -e
DATA_ROOT=${1:-/path/to/DATA_ROOT}
OUTDIR=${2:-outputs/exp1}
python -m src.train --data_root ${DATA_ROOT} --outdir ${OUTDIR} --epochs 200 --labeled_fraction 0.1
