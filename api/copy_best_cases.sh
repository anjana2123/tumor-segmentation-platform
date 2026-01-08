#!/bin/bash
SOURCE_DIR="data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
DEPLOY_DIR="data/brats_deploy/MICCAI_BraTS2020_TrainingData"

cases=(
  "BraTS20_Training_001"
  "BraTS20_Training_037"
  "BraTS20_Training_075"
  "BraTS20_Training_009"
  "BraTS20_Training_026"
  "BraTS20_Training_020"
  "BraTS20_Training_040"
  "BraTS20_Training_069"
  "BraTS20_Training_071"
  "BraTS20_Training_053"
  "BraTS20_Training_091"
  "BraTS20_Training_098"
  "BraTS20_Training_062"
  "BraTS20_Training_006"
  "BraTS20_Training_100"
  "BraTS20_Training_047"
  "BraTS20_Training_067"
  "BraTS20_Training_089"
  "BraTS20_Training_016"
  "BraTS20_Training_015"
)

for case in "${cases[@]}"; do
  echo "Copying $case..."
  cp -r "$SOURCE_DIR/$case" "$DEPLOY_DIR/"
done

echo "Done! Check size:"
du -sh "$DEPLOY_DIR"
