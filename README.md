<div align="center">

# HHOpenFunGraph

### Hierarchical and Holistic Open-Vocabulary Functional 3D Scene Graphs for Indoor Spaces

Xinggang Hu · Chenyangguang Zhang · Alexandros Delitzas · Xiangkui Zhang<br>
Marc Pollefeys · Francis Engelmann · Xiangyang Ji

[![arXiv](https://img.shields.io/badge/arXiv-2605.15753-b31b1b.svg)](https://arxiv.org/abs/2605.15753)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffd21e.svg)](https://huggingface.co/datasets/xg-123/HHOpenFunGraph)

**[Installation](#installation) · [Data](#data-preparation) · [Run](#running-the-pipeline) · [Evaluation](#evaluation) · [Citation](#citation)**

</div>

Official implementation of **HHOpenFunGraph**. Given posed RGB-D observations, the pipeline builds functional 3D scene graphs with objects, functional carriers, interactive units, and their relationships. The accompanying benchmark extends indoor annotations with dense tabletop objects and explicit hierarchical functional relations.

## Installation

Linux, Python 3.11, CUDA 12.4, and a 24 GB NVIDIA GPU are recommended.

```bash
git clone https://github.com/Hbelief1998/HHOpenFunGraph-ECCV26.git
cd HHOpenFunGraph-ECCV26
conda create -n hhofg python=3.11 -y
conda activate hhofg
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/runtime.txt
pip install -e .
bash scripts/install_models.sh
```

The setup script installs pinned RAM++ and SAM 3 source revisions, including the RAM++ compatibility patch. Download these model checkpoints separately:

| Model | Checkpoint |
|---|---|
| [RAM++](https://github.com/xinyu1205/recognize-anything) | `ram_plus_swin_large_14m.pth` |
| [SAM 3](https://huggingface.co/facebook/sam3) | `sam3.pt` |
| [OpenCLIP ViT-H-14](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | `open_clip_model.safetensors` |
| [LLaVA v1.6 Mistral 7B](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | Complete model directory |

```bash
cp configs/paths.example.sh configs/paths.local.sh
# Set dataset and checkpoint paths in configs/paths.local.sh.
source configs/paths.local.sh
read -rsp 'DEEPSEEK_API_KEY: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY
```

RAM++ tags guide LLM scene recognition and semantic atlas generation. Responses are cached; newly observed concepts are handled incrementally. Camera poses are supplied as inputs.

## Data preparation

Download RGB-D observations and camera poses from [FunGraph3D](https://huggingface.co/datasets/OpenFunGraph/FunGraph3D) and the extended annotations from [HHOpenFunGraph](https://huggingface.co/datasets/xg-123/HHOpenFunGraph). Set `FUNGRAPH3D_ROOT` to the directory below:

```text
OpenFunGraph/
├── annotations.json
├── relations.json
├── RootGT_Eval/
│   ├── all_labels.json
│   └── all_edges.json
└── FunGraph3D/
    ├── OpenFunGraph_split.txt
    └── <scene>/<video>/             # rgb/, depth/, images.txt, cameras.txt
```

Use `compat/FunGraph3D/FunGraph3D.annotations.json` and `FunGraph3D.relations.json` from the annotation release, renamed as above. Copy its label/relation vocabularies into `RootGT_Eval/` and its FunGraph3D split from `splits/` into `FunGraph3D/`.

Prepare the evaluation embeddings once in a separate environment:

```bash
conda create -n hhofg-eval python=3.11 -y
conda activate hhofg-eval
pip install -r requirements/eval.txt
export HH_OFG_EMBEDDING_PYTHON="$(command -v python)"
python scripts/encode_eval_text.py --kind clip --model openai/clip-vit-base-patch16 \
  --input "$FUNGRAPH3D_ROOT/RootGT_Eval/all_labels.json" \
  --output "$FUNGRAPH3D_ROOT/RootGT_Eval/all_labels_clip_embeddings.npy"
python scripts/encode_eval_text.py --kind bert --model google-bert/bert-base-uncased \
  --input "$FUNGRAPH3D_ROOT/RootGT_Eval/all_edges.json" \
  --output "$FUNGRAPH3D_ROOT/RootGT_Eval/all_edges_bert_embeddings.npy"
conda activate hhofg
```

Keep `HH_OFG_EMBEDDING_PYTHON` set to this interpreter when running evaluation.

## Running the pipeline

Run all 14 sequences and evaluate the resulting graphs:

```bash
./run_fungraph3d_background.sh --dry-run
./run_fungraph3d_background.sh
./run_fungraph3d_background.sh --status outputs/fungraph3d_runs/latest
```

The server job continues after SSH/VS Code disconnects. Outputs and logs are saved under `outputs/fungraph3d_runs/`. Each sequence produces a `final_hierarchical_graph.json` and `.ply` in its `predictions/<scene>_<video>_hierarchy_lifting_full/mapping3d/map/` directory.

For a single sequence: `./run_full_sequence.sh 3kitchen/video0`.

## Evaluation

The full runner evaluates automatically. To evaluate existing predictions:

```bash
OUTPUTS_ROOT=outputs/fungraph3d_runs/latest/predictions ./evaluate_fungraph3d.sh
```

Evaluation reports node recall and directed functional-triplet recall over the complete split, using 3D box overlap, CLIP/BERT semantic matching, and one-to-one assignment. Hierarchical U-C-O paths contribute U-O relations; remote edges follow source-to-target direction. Structural edges without functional predicates are excluded.

## Acknowledgments

We thank [OpenFunGraph](https://huggingface.co/datasets/OpenFunGraph/FunGraph3D), [Functional-SLAM](https://github.com/Hbelief1998/Functional-SLAM-CoRL_2026), and the model providers above. See [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md) for source attribution and dependency revisions.

## Citation

```bibtex
@article{hu2026hierarchical,
  title={Hierarchical and Holistic Open-Vocabulary Functional 3D Scene Graphs for Indoor Spaces},
  author={Hu, Xinggang and Zhang, Chenyangguang and Delitzas, Alexandros and Zhang, Xiangkui and Pollefeys, Marc and Engelmann, Francis and Ji, Xiangyang},
  journal={arXiv preprint arXiv:2605.15753},
  year={2026}
}
```
