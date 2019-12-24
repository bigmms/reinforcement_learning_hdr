# Robust Two-exposure Image Fusion

## Introduction
In this repository, we provide
* Our model architecture description (EDSR, MDSR)
* NTIRE2017 Super-resolution Challenge Results
* Demo & Training code
* Trained models (EDSR, MDSR) 
* Datasets we used (DIV2K, Flickr2K)
* Super-resolution examples

The code is based on Facebook's Torch implementation of ResNet ([facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)). <br>

### Test models
1. Clone this github repo. 
```
git clone https://github.com/bigmms/reinforcement_learning_hdr
cd reinforcement_learning_hdr
```
2. Place your own **low-resolution images** in `./LR` folder. (There are two sample images - baboon and comic).
3. Run test. We provide ESRGAN model and RRDB_PSNR model and you can config in the `test.py`.
```
python test.py
```
4. The results are in `./results` folder.
