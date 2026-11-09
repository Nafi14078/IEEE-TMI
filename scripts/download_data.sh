#!/usr/bin/env bash
# Downloads the open datasets into data/raw. CHECK EVERY LICENCE (docs/DATA_AND_LICENCES.md) first.
set -euo pipefail
mkdir -p data/raw/isic2019 data/raw/pad_ufes20
cd data/raw/isic2019
# ISIC 2019 challenge training data (~9 GB), CC BY-NC. Source page: https://challenge.isic-archive.com/data/#2019
for f in ISIC_2019_Training_Input.zip ISIC_2019_Training_GroundTruth.csv ISIC_2019_Training_Metadata.csv; do
  [[ -f $f ]] || wget -c "https://isic-challenge-data.s3.amazonaws.com/2019/$f"
done
[[ -d ISIC_2019_Training_Input ]] || unzip -q ISIC_2019_Training_Input.zip
cd ../pad_ufes20
cat <<'MSG'
PAD-UFES-20 (CC BY 4.0) must be downloaded by hand from Mendeley Data:
  https://data.mendeley.com/datasets/zr7vgbcyr2/1
Put metadata.csv here (data/raw/pad_ufes20/) and unzip all image parts into data/raw/pad_ufes20/images/
MSG
