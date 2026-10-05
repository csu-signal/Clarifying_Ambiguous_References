#!/usr/bin/env bash
# Fetch the SIMMC 2.1 files the pipeline reads (about 85 MB) into data/simmc2.
# Needs git and git-lfs. Scene images are not needed.
set -euo pipefail

DEST="${1:-data/simmc2}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/facebookresearch/simmc2.git "$TMP/simmc2"
cd "$TMP/simmc2"
git lfs pull --include "data/fashion_prefab_metadata_all.json,data/furniture_prefab_metadata_all.json,data/simmc2_scene_jsons_dstc10_public.zip"
cd - > /dev/null

mkdir -p "$DEST"
for f in simmc2.1_dials_dstc11_train.json simmc2.1_dials_dstc11_dev.json simmc2.1_dials_dstc11_devtest.json \
         fashion_prefab_metadata_all.json furniture_prefab_metadata_all.json simmc2_scene_jsons_dstc10_public.zip; do
    cp "$TMP/simmc2/data/$f" "$DEST/"
done
echo "SIMMC 2.1 files in $DEST"
