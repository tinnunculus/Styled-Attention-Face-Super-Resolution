# Styled-Attention-Face-Super-Resolution
Jongyeon Lee, Face Super-Resolution with Styled Feature Channel Attention, The Hongik University Graduation Project in 2020 
## Overview
![picture_1](https://user-images.githubusercontent.com/36150943/82112414-ad2ce080-9787-11ea-8b7c-b99b84fa21ea.png)

> __Abstract__: _The Style Generative Adversarial Network(StyleGAN) showed very successful results and verified it is the state-of-art model in the area of generating faces. one of the main reasons for the success is to apply noise all layers for the face's detail. we learned through some experiments that the noises act differently on the different facial area. that means the model can extract and segment the face features such as hair, eyes, teeth, etc. so by using this characteristics, we construct a face-super-resolution model to generate photo-realistic 8-scaling face images with enhancement feature extraction for facial details. for the purpose, we propose enhancement channel attention that has same structure with the original channel attention. but the enhancement channel attention is inputted by the inner product of the latent variables for low resolution image and the feature maps. the latent variables are the output of variational encoder with input image using Gaussian prior. the enhancement channel attention guides the directions for high qualitative feature extractions and ehancement features. we identified through experimental results that the model we proposed enhanced the very detailed facial area with less noises._

## Dependencies
* Python 3.7
* Pytorch >= 1.0
* PIL

## Dataset
* [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset)(Aligned and cropped images at 1024×1024)

## Result
* input image size : 32 x 32
* output image size: 256 x 256
* | input | output | true |
![image](https://user-images.githubusercontent.com/36150943/82115529-4b2ba580-979e-11ea-95ef-c20f7a542c53.png)
![image](https://user-images.githubusercontent.com/36150943/82174958-446f7080-990d-11ea-9bdd-d1a79be4f620.png)
![image](https://user-images.githubusercontent.com/36150943/82175217-4a198600-990e-11ea-8ba3-e901e0ae5b0d.png)
![image](https://user-images.githubusercontent.com/36150943/82175297-8f3db800-990e-11ea-9d58-679f55f0e4a9.png)


## Train
```
$ python train.py --data-path [training data path]

Optional arguments :
  --batch-size,     input batch size for training (default: 4)
  --epochs,         number of epochs to train (default: 10)
  --log-interval,   how many batches to wait before logging training status
  --low-image-size, size of input training images (default: 32)
  --scale,          hom much scale (default: 8)
  --train-test,     0 means train phase, and the otherwise test (default: 0)
  --gen-path,       input image path
  --data-path,      training data path
```
The intermediate output is generated in the "results" folder

## Generate
```
$ python train.py --train-test 1 --gen-path [input image file path]
```
The output is generated in the "samples" folder
