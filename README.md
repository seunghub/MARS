<div align="center">

# MARS  
### Residual-Guided Expert Specialization for Incomplete Multimodal Learning

**ECCV 2026**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-ee4c2c.svg)](#installation)
[![Task](https://img.shields.io/badge/Tasks-Classification%20%7C%20Segmentation-7b61ff.svg)](#benchmark-results)
[![License](https://img.shields.io/badge/License-See%20LICENSE-lightgrey.svg)](#license)

<br>

<img src="./assets/mars_concept.png" width="92%">

<br>
<div align="left">

<em> MARS compares full- and partial-modality representations to identify missingness-induced residuals, enabling residual-guided expert specialization for robust incomplete multimodal learning. </em>

## ✨ Highlights

- **Residual-guided specialization.** MARS compares complete and incomplete multimodal representations during training and uses their residual to expose how missing modalities reshape the task representation.
- **Dual-router MoE.** A privileged residual router supervises expert specialization during training, while a deployable feature router imitates its routing behavior using only available modalities at inference.
- **Discrepancy-aware robustness.** Routing noise is regularized according to the disagreement between residual and feature routers, making experts robust to train-test routing mismatch.
- **Adaptive modality sampling.** MARS prioritizes modality combinations where routers disagree most, improving learning under difficult missing-modality cases.
- **Broad task coverage.** The code covers multimodal classification and segmentation across CASIA-SURF, CREMA-D, UPMC Food-101, and MCubeS.

## 🧠 Method at a Glance

MARS is designed for incomplete multimodal learning, where all modalities are available during training but arbitrary subsets may be missing at test time.

Given a full representation $z^{full}$ and a partial representation $z^{partial}$, MARS computes a residual:

```math
z^{res} = z^{full} - z^{partial}
```

The residual router uses $z^{res}$ as privileged information to assign each sample to experts specialized for the corresponding missingness-induced deviation. Since $z^{res}$ cannot be computed at inference, a feature router learns to mimic the residual router from $z^{partial}$ alone.

```math
\mathcal{L}_{total}
= \lambda_{task}\mathcal{L}_{task}
+ \lambda_{LB}\mathcal{L}_{LB}
+ \lambda_{distill}\mathcal{L}_{distill}
+ \lambda_{noise}\mathcal{L}_{noise}
```

## 🏆 Benchmark Results

### Classification and segmentation summary

| Dataset | Task | Modalities | Metric | Best baseline | MARS |
|---|---:|---|---:|---:|---:|
| CASIA-SURF | Spoof classification | RGB / Depth / IR | ACER ↓ | 3.58 | **2.43** |
| CREMA-D | Emotion classification | Audio / Visual | Accuracy ↑ | 61.35 | **65.52** |
| UPMC Food-101 | Food classification | Image / Text | Accuracy ↑ | 84.81 | **91.59** |
| MCubeS | Material segmentation | RGB / AoLP / DoLP / NIR | mIoU ↑ | 0.4683 | **0.4773** |

## 📁 Repository Structure

```text
MARS/
├── CASIA-SURF/          # Spoof classification with RGB, Depth, and IR
│   ├── classification/
│   │   ├── main.py
│   │   ├── models/our_model.py
│   │   ├── train_func.py
│   │   └── test_func.py
│   └── requirements.txt
├── CREMA-D/             # Audio-visual emotion classification
│   ├── main.py
│   ├── valid.py
│   ├── cramed.sh
│   ├── models/
│   ├── dataset/
│   └── loss/
├── MCubeS/              # Multimodal material segmentation
│   ├── train.py
│   ├── test.py
│   ├── main_train_multimodal.sh
│   ├── main_test_multimodal.sh
│   ├── modeling/
│   └── dataloaders/
├── UPMC Food-101/       # Image-text food classification
│   └── train.py
└── assets/              # Figures used by this README
```

## ⚙️ Installation

Different benchmarks are based on different public codebases, so using a separate environment per benchmark is recommended.

```bash
# Example base environment
conda create -n mars python=3.8 -y
conda activate mars

# Install PyTorch matching your CUDA version first, then install common packages
pip install torch torchvision torchaudio
pip install numpy pandas scipy scikit-learn pillow tqdm matplotlib opencv-python tensorboard
pip install transformers librosa
```

For MCubeS, the original implementation was tested with Python 3.6.12 and PyTorch 1.7.1. You can also use its local requirement file:

```bash
cd MCubeS
pip install -r requirements.txt
```

For CASIA-SURF, the provided requirement file is a Conda environment export:

```bash
cd CASIA-SURF
conda create -n mars-casia --file requirements.txt
conda activate mars-casia
```

## 📦 Dataset Preparation

### CASIA-SURF

Download CASIA-SURF and place it as:

```text
CASIA-SURF/
└── data/
    └── CASIA-SURF/
        ├── train_list.txt
        ├── test_private_list.txt
        └── ...
```

The default training script expects the dataset at `CASIA-SURF/data/CASIA-SURF` when launched from `CASIA-SURF/classification`.

### CREMA-D

Prepare audio wav files and extracted visual frames as:

```text
CREMA-D/
├── dataset/data/CREMAD/
│   ├── train.csv
│   └── test.csv
└── data/CREMA-D/
    ├── AudioWAV/
    └── Image-01-FPS/
```

Frame extraction utilities are provided under `CREMA-D/dataset/data/CREMAD/`.

### MCubeS

Download and extract the MCubeS dataset to the path expected by `MCubeS/mypath.py`, typically:

```text
/dataset/multimodal_dataset/
├── polL_color/
├── polL_aolp_sin/
├── polL_aolp_cos/
├── polL_dolp/
├── NIR_warped/
├── GT/
└── list_folder/
```

### UPMC Food-101

Prepare image and title CSV files as expected by `UPMC Food-101/train.py`:

```text
UPMC Food-101/
└── content/
    ├── images/
    └── texts/
        ├── train_titles.csv
        └── test_titles.csv
```

## 🚀 Training

### CASIA-SURF

```bash
cd CASIA-SURF/classification
python main.py \
  --name classification \
  --anchor_task 20 \
  --sampled_task 20 \
  --load_balance 1 \
  --gate_distill 20 \
  --var_order 0.2 \
  --prob_epoch 20 \
  --temperature 0.06
```

### CREMA-D

```bash
cd CREMA-D
bash cramed.sh
```

Equivalent command:

```bash
python main.py \
  --train \
  --ckpt_path results/cramed/pme \
  --alpha 1e-3 \
  --dataset CREMAD \
  --modulation Normal \
  --pe 0 \
  --gpu_ids 0 \
  --beta 0.20
```

### MCubeS

```bash
cd MCubeS
sh main_train_multimodal.sh
```

Default script command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --is-multimodal \
  --checkname MCubeSNet \
  --use-pretrained-resnet \
  --name experiment_35 \
  --weight_task 0.1 \
  --weight_lb 0.001 \
  --weight_distill 0.005 \
  --weight_order 0.01 \
  --prob_epoch 30
```

### UPMC Food-101

```bash
cd "UPMC Food-101"
python train.py
```

Training constants such as `TRAIN_CSV`, `TEST_CSV`, `IMAGE_DIR`, `BATCH_SIZE`, `EPOCHS`, and `SAVE_PATH` can be edited at the top of `train.py`.

## 🔎 Evaluation

### CASIA-SURF

CASIA-SURF evaluation is run inside the training loop using `test_private_list.txt`. To test a specific missing-modality setting, set `--miss_modal` in `CASIA-SURF/classification/main.py`.

### CREMA-D

```bash
cd CREMA-D
python valid.py
```

### MCubeS

```bash
cd MCubeS
sh main_test_multimodal.sh
```

To evaluate a checkpoint manually:

```bash
python test.py \
  --backbone resnet_adv \
  --dataset multimodal_dataset \
  --pth-path ./run/multimodal_dataset/MCubeSNet/experiment_35/checkpoint.pth.tar \
  --list-folder list_folder \
  --is-multimodal \
  --use-dolp \
  --use-aolp \
  --use-nir
```


## ✅ Practical Tips

- Adjust `CUDA_VISIBLE_DEVICES`, `--gpu`, or `--gpu_ids` to match your machine.
- Check dataset paths before launching each benchmark; the four benchmark folders follow slightly different path conventions.
- For missing-modality experiments, verify the modality mask or command flag used by each dataset implementation.
- Start with a single benchmark, confirm the dataloader works, then scale to the full experiment suite.

## 🙏 Acknowledgements

This repository builds on several public multimodal learning and segmentation codebases, including DMRNet-style classification code, the official MCubeS material segmentation baseline, and UPMC Food-101 image-text fusion resources.

## 📚 Citation

If this repository is useful for your research, please cite:

```bibtex
@inproceedings{baek2026mars,
  title  = {Residual-Guided Expert Specialization for Incomplete Multimodal Learning},
  author = {Baek, Seunghun and Park, Jihwan and Sim, Jaeyoon and Jeong, Minjae and Lee, Hoseok and Kim, Won Hwa},
  year   = {2026},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}
```

## License

Please refer to the repository `LICENSE` file for licensing details.
