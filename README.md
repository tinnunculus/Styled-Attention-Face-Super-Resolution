# Styled-Attention-Face-Super-Resolution
Jongyeon Lee, Face Super-Resolution with Styled Feature Channel Attention, The Hongik University Graduation Project in 2020 
## Overview
![image](https://user-images.githubusercontent.com/36150943/91531389-b19c9400-e947-11ea-9db1-c09637877cb6.png)
![image](https://user-images.githubusercontent.com/36150943/91531487-d2fd8000-e947-11ea-9413-e15c1607f518.png)

> __Abstract__: _The Style Generative Adversarial Network(StyleGAN) showed very successful results and verified it is the state-of-art model in the area of generating faces. one of the main reasons for the success is to apply noise all layers for the face's detail. we learned through some experiments that the noises act differently on the different facial area. that means the model can extract and segment the face features such as hair, eyes, teeth, etc. so by using this characteristics, we construct a face-super-resolution model to generate photo-realistic 8-scaling face images with enhancement feature extraction for facial details. for the purpose, we used self channel attention to enhancement extracted feature factors. we identified through experimental results that the model we proposed enhanced the very detailed facial area with less noises._

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
![image](https://user-images.githubusercontent.com/36150943/83629305-d0072380-a5d4-11ea-8119-7dea664afc86.png)
![image](https://user-images.githubusercontent.com/36150943/83629380-f2993c80-a5d4-11ea-9c46-3a1df3abffac.png)
![image](https://user-images.githubusercontent.com/36150943/83629445-0fce0b00-a5d5-11ea-8877-05055480c042.png)




## Train
```
$ python train.py --train-data-path [train data path]

Optional arguments :
  --batch-size,             input batch size for training (default: 4)
  --epochs,                 number of epochs to train (default: 10)
  --log-interval,           how many batches to wait before logging training status (default: 10)
  --low-image-size,         size of input training images (default: 32)
  --scale,                  scale size (default: 8)
  --train-test,             0 means train phase, and the otherwise test (default: 0)
  --pretrain,               1 means using pretrained model (default: 0)
  --train-data-path,        train data path
  --test-data-path,         test data path
```
The intermediate output is generated in the "results" folder

## test
```
$ python train.py --train-test 1 --test-data-path [test data path]
```
The output is generated in the "samples" folder


## reference
