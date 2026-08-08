# Machine Learning examples

## Basic classifier

- Stochastic Gradient Descent (SGD) method of learning
- Rectified Linear Unit (ReLU) activation function

<p align="center">
  <img src="docs/pics/sgd_l1_n4.webp" />
  <img src="docs/pics/sgd_l1_n8.webp" />
</p>

Pure SGD has a high probability of generating a degenerated border, which can be obeserved
over multiple runs. The Solution is Mini-Batch Gradient Descent method. Skip LeakyReLU.

- Add the second layer

<p align="center">
  <img src="docs/pics/sgd_l1_n4_grad.webp" />
  <img src="docs/pics/sgd_l2_n4x4.webp" />
</p>

The second layer increases "probability contrast".

- Mini-Batch Gradient Descent