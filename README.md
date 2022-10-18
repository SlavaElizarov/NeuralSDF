# NeuralSDF

NeuralSDF is a library for reconstructing implicit 3D surface from meshes and point clouds.



To train a model use the folowing command:

```python train_sdf.py fit -c config.yaml```


Following papers are implemented (partially):

* [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) [config](configs/config.yaml)
* [DiGS : Divergence guided shape implicit neural representation for unoriented point clouds](https://arxiv.org/abs/2106.10811)
* [Modulated Periodic Activations for Generalizable Local Functional Representations](https://arxiv.org/abs/2104.03960)
* [Implicit geometric regularization for learning shapes](https://arxiv.org/abs/2002.10099)
* [Sphere Tracing: A Geometric Method for the Antialiased Ray Tracing of Implicit Surfaces](https://graphics.stanford.edu/courses/cs348b-20-spring-content/uploads/hart.pdf)

TODO:
* [ ] Implement PSNR metric 
* [ ] Add Fourier spectrum visualization
* [ ] Improve rendering speed
* [ ] Implement poisson disk sampling
* [ ] Implement BACON
* [ ] Implement [Critical Regularizations for Neural Surface Reconstruction in the Wild](https://arxiv.org/abs/2206.03087)
* [ ] Implement [Learning Deep Implicit Functions for 3D Shapes with Dynamic Code Clouds](https://arxiv.org/abs/2203.14048)
* [ ] Investigate bias initialization for PFA modulation method