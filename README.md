# Machine Learning examples

## Simple classifiers

### Single layer SGD

- Stochastic Gradient Descent (SGD) method of learning
- Rectified Linear Unit (ReLU) activation function

<p align="center">
  <img src="docs/pics/sgd_relu_l1_n4_border.webp" />
  <img src="docs/pics/sgd_relu_l1_n8_border.webp" />
</p>

Pure SGD has a high probability of generating a degenerated border, which can be obeserved
over multiple runs. The Solution is Mini-Batch Gradient Descent method or LeakyReLU activation.

### Two layers SGD

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

### Mini-Batch Gradient Descent vs Adam optimizer

  - Batch size for both cases is 32 dots.
  - Two layers system with ReLU: 24 neurons on first layer and 12 on second

<p align="center">
  <img src="docs/pics/minibatch_relu_l2_n24x12.webp" />
  <img src="docs/pics/adam_relu_l2_n24x12.webp" />
</p>

  - Simple mini-batch optimizer uses increased learning rate: 0.05 * sqrt(32), otherwise learning is too slow
  - Adam shows significantly better results in all cases.

## Convolutional Network (CNN)

  - Input layer 14x14 pixels. Input pictures of circles, squares and trianlges in random order
  - Input Convolutional Layer. Three separate 3x3 filters
  - Single Feature map layer 12x12 pixels after passing LeakyReLU activation
  - Classification head / Softmax probabilities

<p align="center">
  <img src="docs/pics/cnn_14x14_f3_o3.webp" />
</p>
