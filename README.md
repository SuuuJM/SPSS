# SPSS

### Introduction

This is the Implementation of《Self-Paced Sample Selection for Barely-Supervised Medical Image Segmentation》

### Usage

1. Put the data in `data/LA/2018LA_Seg_Training Set` and `data/LA/processed_h5`.

2. Train the model

```
python train_spss.py --exp model_name
```
3. Test the model

```
python test.py --model model_name
```
