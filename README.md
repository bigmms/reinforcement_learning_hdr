# Robust Two-exposure Image Fusion
Implementation for robust two-exposure image fusion, CVPR 2020 (under review).

## Introduction
In this repository, we provide
* Our model architecture description (HDR_RL)
* Demo code
* Trained models
* Fusion examples


## Test models
1. Clone this github repo. 
```
git clone https://github.com/bigmms/reinforcement_learning_hdr
cd reinforcement_learning_hdr
```
2. Place your own **LDR images** in `./test/Images` folder. (There are several sample images there).
3. Download pretrained models from [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) or [Baidu Drive](https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ). Place the trained model in `./checkpoints`. 
4. Download pretrained vgg16.npy from [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) or [Baidu Drive](https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ). Place the trained model in the root directory. 
5. Run test. We provide the trained model and you can config in the `agent_test.py`.
```
python agent_test.py
```
6. The results are in `./test/test_run/results` folder.
