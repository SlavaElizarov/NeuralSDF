# NeuralSDF

NeuralSDF is a library for reconstructing implicit 3D surface from meshes and point clouds.



To train a model use the folowing command:

```python train_sdf.py fit -c configs/config.yaml```


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
* [x] Make complex valued layer able to learn frequencied bigger that $\pi$ 
* [ ] Implement BACON
* [ ] Implement ground truth SDF supervision
* [ ] Implement [Critical Regularizations for Neural Surface Reconstruction in the Wild](https://arxiv.org/abs/2206.03087)
* [ ] Implement [Learning Deep Implicit Functions for 3D Shapes with Dynamic Code Clouds](https://arxiv.org/abs/2203.14048)
* [x] Investigate bias initialization for PFA modulation method. Result: PFA modulation was removed from the code.


### On optimization algorithms:
* Adam with amsgrad=False and lr=1e-4 works better than amsgrad=True
* RAdam results slightly worse than Adam by metrics and less noisy visually (especially noticable if LogNormal init is used) but a bit unstable during training





