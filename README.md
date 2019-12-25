# Robust Two-exposure Image Fusion
Implementation for robust two-exposure image fusion, CVPR 2020 (under review).

## Introduction
In this repository, we provide
* Our model architecture description (HDR_RL)
* Demo code
* Trained models
* Fusion examples

## Architecture

![](./framework1.png)

Please refer to our paper for details.

## Dependencies
* Python 3
* [Tensorflow >= 1.14.0](https://www.tensorflow.org/) (CUDA version >= 10.0 if installing with CUDA. [More details](https://www.tensorflow.org/install/gpu/))
* Python packages:  `pip install -r requirement.txt`

Our code is tested under Ubuntu 14.04 and 16.04 environment with Titan X GPUs (12GB VRAM).

## Test models
1. Clone this github repo. 
```
git clone https://github.com/bigmms/reinforcement_learning_hdr
cd reinforcement_learning_hdr
```
2. Place your own **LDR images** in `./test/Images` folder. (There are several sample images there).
3. Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1iqkGTl8sqoVEaVFo4uoAJiLFtce_f8cu?usp=sharing). Place the trained model in `./checkpoints`. 
4. Run test. We provide the trained model and you can config in the `agent_test.py`.
```
python agent_test.py
```
5. The results are in `./test/test_run/results` folder.

## Dataset
If you want to train or evaluate our models with DIV2K or Flickr2K dataset, please download the dataset from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).
Place the tar file to the location you want. **(We recommend /var/tmp/dataset/)**  <U>If the dataset is located otherwise, **you have to change the optional argument -dataset for training and test.**</U>

* [**DIV2K**](http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf) from [**NTIRE2017**](http://www.vision.ee.ethz.ch/ntire17/)
    ```bash
    makeData = /var/tmp/dataset/ # We recommend this path, but you can freely change it.
    mkdir -p $makeData/; cd $makedata/
    tar -xvf DIV2K.tar
    ```
    You should have the following directory structure:

    `/var/tmp/dataset/DIV2K/DIV2K_train_HR/0???.png`<br>
    `/var/tmp/dataset/DIV2K/DIV2K_train_LR_bicubic/X?/0???.png`<br>
    `/var/tmp/dataset/DIV2K/DIV2K_train_LR_unknown/X?/0???.png`<br>
