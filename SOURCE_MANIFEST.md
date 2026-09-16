# Source acknowledgments

HHOpenFunGraph includes dataset and semantic frontend components adapted from earlier research implementations. The adapters and graph pipeline are maintained in this repository; running it does not require a Functional-SLAM or OpenFunGraph source checkout.

| Component | Origin | Location in this release |
|---|---|---|
| RGB-D readers, COLMAP pose conventions, evaluation conventions | OpenFunGraph | `src/hhofg/data/`, `src/hhofg/eval/` |
| Semantic prompts, scene locking, RAM++ integration, SAM 3 role-aware processing | [Functional-SLAM](https://github.com/Hbelief1998/Functional-SLAM-CoRL_2026) | `src/hhofg/frontend2d/` |

External models are installed separately:

| Dependency | Revision used for release validation | Local adaptation |
|---|---|---|
| [recognize-anything](https://github.com/xinyu1205/recognize-anything) | `7cb804a8609e9f4b1a50b7f31436d2df40bb9481` | `patches/ram-transformers.patch`: Transformers compatibility, local tokenizer resolution, and checkpoint label-embedding alignment |
| [SAM 3](https://github.com/facebookresearch/sam3) | `b26a5f330e05d321afb39d01d3d4881f258f65ff` | No source patch; the wrapper accepts the upstream fixed-resolution builder |

The RAM++ patch is derived from the Apache-2.0 recognize-anything sources. Its upstream license is retained in `patches/LICENSE-recognize-anything`. Other dependencies, model weights, and datasets retain their respective upstream licenses and access terms. No model checkpoints or sensor data are redistributed here.
