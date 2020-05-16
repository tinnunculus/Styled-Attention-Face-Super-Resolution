# Styled-Attention-Face-Super-Resolution
Jongyeon Lee, Face Super-Resolution with Styled Feature Channel Attention, The Hongik University Graduation Project 
## Overview
![picture_1](https://user-images.githubusercontent.com/36150943/82112414-ad2ce080-9787-11ea-8b7c-b99b84fa21ea.png)

> __Abstract__: _The Style Generative Adversarial Network(StyleGAN) showed very successful results and verified it is the state-of-art model in the area of generating faces. one of the main reasons for the success is to apply noise all layers for the face's detail. we learned through some experiments that the noises act differently on the different facial area. that means the model can extract and segment the face features such as hair, eyes, teeth, etc. so we construct a face-super-resolution model to generate photo-realistic 8-scaling face images with enhancement feature extraction for facial details. for the purpose, we propose enhancement channel attention that have same structure with the original channel attention. but the enhancement channel attention is inputted by the inner product of the latent variables of input image and the feature maps. the latent variables are the output of variational encoder with input image using Gaussian prior. the enhancement channel attention guides the directions for high qualitative feature extractions and ehancement features. we identified through experimental results that the model we proposed enhanced the very detailed facial area with less noises._

## Dependencies
* Python 3.7
* Pytorch >= 1.0
* PIL

## Dataset
* [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset)

## Result
* input image size : 32 x 32
* output image size: 256 x 256
![image](https://user-images.githubusercontent.com/36150943/82115529-4b2ba580-979e-11ea-95ef-c20f7a542c53.png)
![image](https://user-images.githubusercontent.com/36150943/82115446-c8a2e600-979d-11ea-8f7a-09a0a71c9d92.png)
![image](https://user-images.githubusercontent.com/36150943/82115447-cb054000-979d-11ea-91d6-d38fd7fe12a0.png)
![image](https://user-images.githubusercontent.com/36150943/82115455-d2c4e480-979d-11ea-94a4-067c13f28187.png)
![image](https://user-images.githubusercontent.com/36150943/82115458-d5273e80-979d-11ea-918a-e418b882e73c.png)

## Train
```
$ python train.py --data-path [training data path]

Optional arguments :
  --batch-size,     input batch size for training (default: 32)
  --epochs,         number of epochs to train (default: 10)
  --no-cuda,        enables CUDA training
  --log-interval,   how many batches to wait before logging training status
  --low-image-size, size of input training images (default: 32)
  --scale,          hom much scale (default: 8)
  --train-test,     0 means train phase, and the otherwise test (default: 0)
  --gen-path,       input image path
  --data-path,      training data path
```

## Generate
```
$ python train.py --gen-path [input image file path]
```
