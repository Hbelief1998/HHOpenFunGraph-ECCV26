# Copy to configs/paths.local.sh, edit the paths, then source it from the repo root.
# Keep credentials in your environment; do not add them to this file.
export FUNGRAPH3D_ROOT="${FUNGRAPH3D_ROOT:-$PWD/data/OpenFunGraph}"
export RAMPP_REPO="${RAMPP_REPO:-$PWD/third_party/recognize-anything}"
export SAM3_REPO="${SAM3_REPO:-$PWD/third_party/sam3}"
export SAM3_BPE_PATH="${SAM3_BPE_PATH:-$SAM3_REPO/sam3/assets/bpe_simple_vocab_16e6.txt.gz}"
export RAMPP_CHECKPOINT="${RAMPP_CHECKPOINT:-$PWD/checkpoints/ram_plus_swin_large_14m.pth}"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$PWD/checkpoints/sam3.pt}"
export CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-$PWD/checkpoints/open_clip_model.safetensors}"
export LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-$PWD/checkpoints/llava-v1.6-mistral-7b-hf}"
export HH_OFG_RUNTIME_PYTHON="${HH_OFG_RUNTIME_PYTHON:-$(command -v python)}"
# Set this to the absolute interpreter path of the separate evaluation environment.
# export HH_OFG_EMBEDDING_PYTHON=/path/to/hhofg-eval/bin/python
# Optional local snapshots instead of downloading evaluation text models:
# export HH_OFG_CLIP_EVAL_MODEL=/path/to/clip-vit-base-patch16
# export HH_OFG_BERT_EVAL_MODEL=/path/to/bert-base-uncased
