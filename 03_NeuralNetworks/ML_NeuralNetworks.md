# Mechine Learning: Neural Networks

These are my notes on the Coursera course by Andrew Ng ["Machine Learning"](https://www.coursera.org/learn/machine-learning).

For setup and general information, please look at `../README.md`.

This file my notes related to **neural networks**.

Note that Latex formulae are not always rendered in Markdown readers; for instance, they are not rendered on Github, but they are on VS Code with the Markup All In One plugin.
Therefore, I provide a pseudocode of the most important equations.
An alternative would be to use Asciidoc, but rendering of equations is not straightforward either.

Overview of contents:

1. Motivations
2. Neural Networks
3. Applications


## 1. Motivations

### 1.1 Non-Linear Hypotheses

Linear and logistic regression are the basis algorithms in machine learning that make possible to infer categorical or quantitative values.

However, as the complexity of the domain increases (i.e., the number of features used as independent variables), they fail because the cannot deal with **non-linearities and high dimensionalities**.

Example 1: 100 features, logistic regression with curvy decision boundaries. In order to have curvy boundaries, we need higher degree polynomials in the hypothesis model: but the number of terms in the model increases with `O(n^d)`, being `n` the original features and `d` the degree. For instance, if we want a humble quadratic degree `d = 2` (which is not able to model holes and very curvy stuff), we would require approximately `n^2/2 = 5,000` terms.

Example 2: Image classification, 50 x 50 pixels. We consider each pixel an independent variable: we have `50 x 50 = 2,500` pixels. Now, in that space, very complex decision boundaries need to be defined, so we wold require a very high degree polynomial. However, a humble quadratic degree `d = 2` leads to approximately 3 million terms -- the sizes explode even with very simplified models.

### 1.2 Neurons and the Brain

Neural re-wiring experiments with mice have been conducted: visual input was redirected to the auditory cortex and the mice learned to see with that region of the brain. Therefore, the hypothesis that there is one learning algorithm that adapts to the similar hardware pieces has arisen.

Ng provides examples related to how it is possible understand different sensory information with other senses: echolocation, haptic belts, etc.

Neurons in the brain
- They have nucleus, which processes the incoming data and sends the output signal.
- An axon: output data.
- Dendrites: input data.
- The neurons communicate with each other through small pulses of electricity called, spikes.

## 2. Neural Networks: Model Representation

### 2.1 A Single Neuron: A Perceptron

The computer representation of a neuron is a logistic unit:
- Inputs: `x_1, x_2, x_3` and bias `x_0 = 1`
- Output: `h(theta*X) = g(theta*X) = sigmoid(theta*X) = 1 / (1 + exp(-theta*X))`

$x = [x_0 = 1, x_1, x_2, x_3]^T$
$\theta = [\theta_0, \theta_1, \theta_2, \theta_3]^T$
$h(x) = \frac{1}{1 + e^{-\theta^T x}}$

The bias unit is a constant value which is always present, like the intercept variable in linear regression. However, sometimes it is not drawn.

The sigmoid (logistic) function is one of the possible **activation functions** that map the weighted sum of the inputs to a desired output region (e.g., `[0,1]`). The activation function is applied in the neuron.

In neural networks, the `theta` parameters are called **weights** and they are the property of the input links.

A single neuron is called a **perceptron**

![Representation of a single neuron](./pics/single_neuron.png)

### 2.2 A Nerual Network: Multi-Layered Perceptrons (MLP)

We basically connect the output of a neuron to other neurons, creating a network of layers. We distinguish:
- The input layer: the input data or variables
- The output layer: the output signal (it can be one value or several)
- The hidden layer: all layers of neurons between the input and the output layer

Several layers of perceptrons/neurons are called a **Multi-Layer Perceptrons (MLP)**, and they constitute a neural network.

![Multi-layer perceptrons (MLP)](./pics/mlp.png)

