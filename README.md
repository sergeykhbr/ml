# Machine Learning examples

## Basic classifier

- Stochastic Gradient Descent (SGD) method of learning
- Rectified Linear Unit (ReLU) activation function

<p align="center">
  <img src="docs/pics/sgd_relu_l1_n4_border.webp" />
  <img src="docs/pics/sgd_relu_l1_n8_border.webp" />
</p>

Pure SGD has a high probability of generating a degenerated border, which can be obeserved
over multiple runs. The Solution is Mini-Batch Gradient Descent method. Skip LeakyReLU.

### Add the second layer

<p align="center">
  <img src="docs/pics/sgd_relu_l1_n4.webp" />
  <img src="docs/pics/sgd_relu_l2_n4x4.webp" />
</p>

The second layer increases "probability contrast".

### Use Sigmoid activation function in hidden layers instead of ReLU

<p align="center">
  <img src="docs/pics/sgd_sigmoid_l1_n4.webp" />
  <img src="docs/pics/sgd_sigmoid_l2_n4x4.webp" />
</p>

  - Sigmoid based system trains much slower
  - Regions border has smooth, organic wavy curves
  - Potential problem: Vanishing Gradaients. Totally makes sense to use mixed Sigmoid/ReLU layers.

### Mini-Batch Gradient Descent
