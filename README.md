# Automatic Intermediate Generation with Deep Reinforcement Learning for Robust Two-exposure Image Fusion

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbigmms%2Freinforcement_learning_hdr&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
![](https://img.shields.io/badge/Tensorflow-1.14.0-yellow)
![](https://img.shields.io/badge/Cuda-10.0-blue)

![](./demo/framework1.png)

## Introduction
XXXXXXXX

XXXXXXXX

**Authors**: Jia-Li Yin, Bo-Hao Chen, Yan-Tsung Peng, and Hau Hwang

**Paper**: Automatic Intermediate Generation with Deep Reinforcement Learning for Robust Two-exposure Image Fusion


## Requirements
### Dependencies
* Python 3.5+
* CUDA 10.0 
* Tensorflow 1.14.0
* Keras 2.2.5
* Numpy 1.16.1
* Scipy 1.0.0
* Opencv 4.1.1.26
* Opencv-contrib-python 4.1.1.26

You can install the required libraries by the command `pip install -r requirements.txt`.

### Model
* Pre-trained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1iqkGTl8sqoVEaVFo4uoAJiLFtce_f8cu?usp=sharing) or [baidu drive](https://pan.baidu.com/s/1nLrWmgkYNffSJHB1Fsr0Gw) (password: 2wrw).

### It was tested and runs under the following OSs:
* Windows 10 with GeForce GTX 1060 GPU
* Ubuntu 16.04 with NVIDIA GTX 1080 Ti GPU

Might work under others, but didn't get to test any other OSs just yet.

## Getting Started:
### Usage
```
python agent_test.py --model_path /your/checkpoints/path --data_dir /your/LDR/images/path --output_dir /results/to/save
```
### Demo
To test this code
```
python agent_test.py --model_path ./checkpoints/test_run.ckpt-700 --data_dir ./test/Images/ --output_dir ./result/
```

## Results

![](./demo/results_2.png)

## License + Attribution
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this code in a scientific publication, please cite the following [paper](https://ieeexplore.ieee.org/document/9357944):
```
@ARTICLE{ChenTITS2021,
  author={B. -H. {Chen} and S. {Ye} and J. -L. {Yin} and H. -Y. {Cheng} and D. {Chen}},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Deep Trident Decomposition Network for Single License Plate Image Glare Removal}, 
  year={2021},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TITS.2021.3058530}}
```
