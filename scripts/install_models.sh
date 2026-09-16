#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p third_party
git clone https://github.com/xinyu1205/recognize-anything.git third_party/recognize-anything
git -C third_party/recognize-anything checkout 7cb804a8609e9f4b1a50b7f31436d2df40bb9481
git -C third_party/recognize-anything apply ../../patches/ram-transformers.patch
python -m pip install --no-deps -e third_party/recognize-anything

git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 checkout b26a5f330e05d321afb39d01d3d4881f258f65ff
python -m pip install --no-deps -e third_party/sam3
