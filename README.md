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

Our code is tested under Windows 10 environment with GeForce GTX 1060 GPU (6GB VRAM). Might work under others, but didn't get to test any other OSs just yet.

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

   **You can run different models and scales by changing input arguments.**
```
    # To run for scale 2, 3, or 4, set -scale as 2, 3, or 4
    # To run EDSR+ and MDSR+, you need to set -selfEnsemble as true

    cd $makeReposit/NTIRE2017/demo

    # Test EDSR (scale 2)
    th test.lua -model EDSR_x2 -selfEnsemble false

    # Test EDSR+ (scale 2)
    th test.lua -model EDSR_x2 -selfEnsemble true

    # Test MDSR (scale 2)
    th test.lua -model MDSR -scale 2 -selfEnsemble false

    # Test MDSR+ (scale 2)
    th test.lua -model MDSR -scale 2 -selfEnsemble true
```
(Note: To run the **MDSR**, model name should include `multiscale` or `MDSR`. e.g. `multiscale_blahblahblah.t7`)
    

5. The results are in `./test/test_run/results` folder.

## Results

![](./results_2.png)
