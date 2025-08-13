#!/bin/bash

# get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# change to the parent directory
cd "$SCRIPT_DIR/.."

# create distribution folder and copy contents
echo "Creating archive: polar_segment_dist.tar.gz"
rm -rf .polar_segment_dist
mkdir -p .polar_segment_dist/{segment_models,configs,cases,ckpts}
cp infer.py .polar_segment_dist/
cp horao_dataset.py .polar_segment_dist/
cp requirements.txt .polar_segment_dist/
cp install.sh .polar_segment_dist/
cp README.md .polar_segment_dist/
cp segment_models/unet.py .polar_segment_dist/segment_models/unet.py
cp configs/infer.yml .polar_segment_dist/configs/infer.yml
cp -r docs .polar_segment_dist/
cp -r utils .polar_segment_dist/
cp -r mm_torch .polar_segment_dist/
rm -rf .polar_segment_dist/mm_torch/.git

# cases
cp cases/val2_600_b_npp_ht_tumor.txt .polar_segment_dist/cases/val2_600_b_npp_ht_tumor.txt
cp cases/k1_600_b_npp_ht_tumor_imbalance.txt .polar_segment_dist/cases/k1_600_b_npp_ht_tumor_imbalance.txt
cp cases/k2_600_b_npp_ht_tumor_imbalance.txt .polar_segment_dist/cases/k2_600_b_npp_ht_tumor_imbalance.txt
cp cases/k3_600_b_npp_ht_tumor_imbalance.txt .polar_segment_dist/cases/k3_600_b_npp_ht_tumor_imbalance.txt

# models
cp ckpts/600nm_lc_model.pt .polar_segment_dist/ckpts/
cp ckpts/600nm_mmff_model.pt .polar_segment_dist/ckpts/

# create archive file and clean up
tar -czf polar_segment_dist.tar.gz .polar_segment_dist/
rm -rf .polar_segment_dist
echo "Archive created successfully"