# Automatic Intermediate Generation With Deep Reinforcement Learning for Robust Two-Exposure Image Fusion

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbigmms%2Freinforcement_learning_hdr&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
![](https://img.shields.io/badge/Tensorflow-1.14.0-yellow)
![](https://img.shields.io/badge/Cuda-10.0-blue)

![](./demo/framework1.png)

## Introduction
Fusing low dynamic range (LDR) for high dynamic range (HDR) images has gained a lot of attention, especially to achieve real-world application significance when the hardware resources are limited to capture images with different exposure times. However, existing HDR image generation by picking the best parts from each LDR image often yields unsatisfactory results due to either the lack of input images or well-exposed contents. To overcome this limitation, we model the HDR image generation process in two-exposure fusion as a deep reinforcement learning problem and learn an online compensating representation to fuse with LDR inputs for HDR image generation. Moreover, we build a two-exposure dataset with reference HDR images from a public multiexposure dataset that has not yet been normalized to train and evaluate the proposed model. By assessing the built dataset, we show that our reinforcement HDR image generation significantly outperforms other competing methods under different challenging scenarios, even with limited well-exposed contents. More experimental results on a no-reference multiexposure image dataset demonstrate the generality and effectiveness of the proposed model. To the best of our knowledge, this is the first work to use a reinforcement-learning-based framework for an online compensating representation in two-exposure image fusion.

**Authors**: Jia-Li Yin, Bo-Hao Chen, Yan-Tsung Peng, and Hau Hwang

**Paper**: [PDF](https://ieeexplore.ieee.org/document/9466369)


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
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this code in a scientific publication, please cite the following [paper](https://ieeexplore.ieee.org/document/9466369):
```
@ARTICLE{YinTNNLS21,
author={Yin, Jia-Li and Chen, Bo-Hao and Peng, Yan-Tsung and Hwang, Hau},  
journal={IEEE Transactions on Neural Networks and Learning Systems},  
title={Automatic Intermediate Generation With Deep Reinforcement Learning for Robust Two-Exposure Image Fusion},   
year={2022}, 
volume={33}, 
number={12},
pages={7853-7862}}
```
