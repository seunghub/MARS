# ECCV 2026 Paper2562 Classification
Our codebase is built upon the recent baseline [DMRNet](https://github.com/shicaiwei123/ECCV2024-DMRNet)
We follow the original training and evaluation settings.


## dataset preparation
download CAISIA-SURF to data/CASIA-SURF

## run

```bash
    cd classification
    python main.py --name=classification --anchor_task=20 --sampled_task=20 --load_balance=1 --gate_distill=20 --var_order=0.2 --prob_epoch=20 --temperature=0.06
```