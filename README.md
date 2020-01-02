# Robust Two-exposure Image Fusion
Implementation for robust two-exposure image fusion, CVPR 2020 (under review).

## Introduction
In this repository, we provide
* Our model architecture description (HDR_RL)
* Demo code
* Trained models
* Fusion examples

## Architecture

![](./demo/framework1.png)

Please refer to our paper for details.

## Dependencies
* Python 3
* [Tensorflow >= 1.14.0](https://www.tensorflow.org/) (CUDA version >= 10.0 if installing with CUDA. [More details](https://www.tensorflow.org/install/gpu/))
* Python packages:  `pip install -r requirement.txt`

Our code is tested under Windows 10 environment with GeForce GTX 1060 GPU (6GB VRAM). Might work under others, but didn't get to test any other OSs just yet.

## Test models
1. Clone this github repo. 
```
git clone https://github.com/bigmms/reinforcement_learning_hdr
cd reinforcement_learning_hdr
```
2. Place your own **LDR images** in `./test/Images` folder. (There are several sample images there).
3. Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1iqkGTl8sqoVEaVFo4uoAJiLFtce_f8cu?usp=sharing). Place the trained model in `./checkpoints`. 
4. Run test. We provide the trained model and you can config in the `agent_test.py`. You can run different models and testing datasets by changing input arguments.
```
# To run with different models, set -model_path as your model path.
# To run for different testing dataset, you need to set -data_dir as your data path.

cd $makeReposit/reinforcement_learning_hdr

# Test model
python agent_test.py --model_path ./checkpoints/test_run.ckpt-700 --data_dir ./test/Images/
```
    

5. The results are in `./test/test_run/results` folder.

## Results

![](./demo/results_2.png)
