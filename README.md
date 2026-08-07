# Machine Learning examples

## Basic classifier

- Stochastic Gradient Descent (SGD) method of learning
- Rectified Linear Unit (ReLU) activation function

<p float="center">
  <img src="docs/pics/sgd_l1_n4.webp" />
  <img src="docs/pics/sgd_l1_n8.webp" />
</p>

Pure SGD has a high probability of generating a degenerated border, which can be obeserved
over multiple runs. The Solution is Mini-Batch Gradient Descent method. Skip LeakyReLU.

- Add the second layer
- Mini-Batch Gradient Descent