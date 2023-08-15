# Neural Surface (WIP)

## Surface reconstruction and generation with neural fields

<p align="center">
  <img src="car.gif" alt="sdf" />
</p>

Signed distance field (SDF) reconstruction of the 
Frazer Nash Supersport 1929 model provided by [Achmad Sarifudin](https://www.blendswap.com/blend/30188). 


## Description

This repository contains a range of models that can be used for surface reconstruction and generation, based on [neural fields](https://arxiv.org/abs/2111.11426). 

We highly recommend to read this survey [Neural Fields in Visual Computing and Beyond](https://arxiv.org/abs/2111.11426).

The models've been unified and consolidated into one place for convenient comparison and experimentation. Each model has been implemented as a separate class, allowing for easy integration into your own projects. Additionally, for each model, a configuration file including all parameters is provided, as well as a training script.

This repository uses of [PyTorch Lightning](https://www.pytorchlightning.ai/) for training, and Lightning CLI for configuration, which adds to its utility and ease of use.

## Installation

All dependencies are listed in the requirements.txt file. To install them, run the following command:

```pip install -r requirements.txt```

Enjoy!

## Usage

To train a model use the folowing command:

```python train_sdf.py fit -c configs/config.yaml```

### Configs

All configs are located in the configs folder. Each model has its own config file. The config file contains all the parameters of the model, as well as the training process.



### Following papers are implemented (partially):

* [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) [config](configs/config.yaml)
* [DiGS : Divergence guided shape implicit neural representation for unoriented point clouds](https://arxiv.org/abs/2106.10811)
* [Modulated Periodic Activations for Generalizable Local Functional Representations](https://arxiv.org/abs/2104.03960)
* [Implicit geometric regularization for learning shapes](https://arxiv.org/abs/2002.10099)
* [Sphere Tracing: A Geometric Method for the Antialiased Ray Tracing of Implicit Surfaces](https://graphics.stanford.edu/courses/cs348b-20-spring-content/uploads/hart.pdf)
* Subtraction-based attention from [Point Transformer
](https://arxiv.org/abs/2012.09164) [Code](layers/attention.py)

### TODO:
* [ ] Implement PSNR metric 
* [ ] Add Fourier spectrum visualization
* [ ] Improve rendering speed
* [ ] Implement poisson disk sampling. [Fast Poisson Disk Sampling in Arbitrary Dimensions](https://www.cs.ubc.ca/~bridson/docs/bridson-siggraph07-poissondisk.pdf)
* [ ] Implement geodesic training for 3d meshes
* [x] Implement complex valued layer from [Sinusoidal Frequency Estimation by Gradient Descent](https://arxiv.org/abs/2210.14476)
* [x] Make complex valued layer able to learn frequencied greater than $\pi$ 
* [ ] Implement BACON
* [ ] Implement ground truth SDF supervision
* [ ] Implement [Critical Regularizations for Neural Surface Reconstruction in the Wild](https://arxiv.org/abs/2206.03087)
* [ ] Implement [Learning Deep Implicit Functions for 3D Shapes with Dynamic Code Clouds](https://arxiv.org/abs/2203.14048)
* [ ] Investigate [Rectified Adam optimizer](https://arxiv.org/abs/1908.03265) influence on perceptual quality. 
