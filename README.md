# Robust Two-exposure Image Fusion
![](./demo/framework1.png)

**Paper**: Robust Two-exposure Image Fusion 

**Authors**: Bo-Hao Chen, Jia-Li Yin

Implementation for robust two-exposure image fusion, CVPR 2020 (under review).

## Requirements
### Dependencies
* Python 3.5+
* CUDA 10.0 
* Tensorflow 1.14.0
* Keras 2.2.5
* Numpy 1.16.1
* Scipy 1.0.0
* Opencv 4.1.1.26
* opencv-contrib-python 4.1.1.26

You can install the required libraries by the command `pip install -r requirements.txt`.

### Model
* Pre-trained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1iqkGTl8sqoVEaVFo4uoAJiLFtce_f8cu?usp=sharing).
### It was tested and runs under the following OSs:
* Windows 10 with GeForce GTX 1060 GPU (6GB VRAM)

Might work under others, but didn't get to test any other OSs just yet.

## Getting Started:
### usage
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
