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
* [FFHQ dataset][https://github.com/NVlabs/ffhq-dataset]

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
$ python train.py --gen-path [input image path]
```
