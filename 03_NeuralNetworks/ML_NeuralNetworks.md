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

Andrew Ng provides examples related to how it is possible understand different sensory information with other senses: echolocation, haptic belts, etc.

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

### 2.2 A Neural Network: Multi-Layered Perceptrons (MLP)

We connect the output of a neuron to other neurons, creating a network of layers. We distinguish:

- The input layer: the input data or variables (layer $j=1$)
- The output layer: the output signal (it can be one value or several)
- The hidden layer: all layers of neurons between the input and the output layer

Several layers of perceptrons/neurons are called a **Multi-Layer Perceptrons (MLP)**, and they constitute a neural network.

![Multi-layer perceptrons (MLP)](./pics/mlp.png)

The effect is that we concatenate vector products mapped with sigmoid or activation functions.

Notation:

- $a_i^{(j)}$: activation of unit $i$ in layer $j$
  - $a$ is the output of a hidden layer neuron; $i = 1, 2, ..$ is the number of the neuron or unit (down); $j = 1, 2, ...$ is number of the layer (up)
- $\Theta^{(j)}$: matrix of weights controlling the function mapping from layer $j$ to $j+1$
  - rows: units in layer $j+1$ = $s_{j+1}$
  - columns: units in layer $j$ + 1 (bias) = $s_{j}$ + 1
  - size: $s_{j+1} \times (s_{j} + 1)$ = **new x (old + 1)**
  - $\Theta^{(1)}_3$: row vector for unit 3 in new layer 1+1 = 2.
  - $\Theta^{(1)}_{32}$: weight from old layer unit 2 to new layer unit 3, being the new layer 1+1 = 2
  - $\Theta^{(1)}_{30}$: weight from bias to new layer unit 3, being the new layer 1+1 = 2

 Therefore:

   - Each layer jump is a matrix of weights: **new x (old + 1)**.
   - Wach input is the sigmoid of a dot product between a **weight matrix row and previous layer outputs / activations**.

![Neural network model](./pics/neural_network_model.png)

### 2.3 Feed-Forward Propagation: Vectorized Implementation

Feed-forward is the process of passing an input ($x$) to the neural network and obtaining its output ($h(x)$). In between the input is mapped in stages to the output. Each stage is executed after a layer and consists in multiplying the previous activation/layer-output by the weight matrix between the two layers.

Summary of steps:
- We have our input: $x = [x_1, x_2, x_3, ...]^T$
- We add the bias components to its front: $x = [x_0, x_1, x_2, x_3, ...]^T$, $x_0 = 1$
- For generalization, we consider it to be the activation or output from a previous layer: $a^{(1)} = x = [x_0, x_1, x_2, x_3, ...]^T = [a^{(1)}_0, a^{(1)}_1, a^{(1)}_2, a^{(1)}_3, ...]^T$, being layer 1 the input layer. So, basically, $a^{(j)}$ is the input of layer $j$, also known as the activation of layer $j$.
- We apply the weights to the activations: $z^{(2)} = \Theta^{(1)} a^{(1)} = [z^{(2)}_0, z^{(2)}_1, z^{(2)}_2, ...]^{T}$
- We apply the sigmoid activation function to each element of the vector $z$ to obtain the activations from layer 2: $a^{(2)} = g(z^{(2)}) = [a^{(2)}_0, a^{(2)}_1, a^{(2)}_2, a^{(2)}_3, ...]^T$
- The process is repeated until we reach the output layer $k$: $h(x) = g(z^{(k)})$

![Feed-forward propagation in a vectorized form](./pics/feed_forward.png)

Some remarks:
- $x$ is input vector and we need to insert the bias component to it: $x_0 = 1$
- We need to extend the every activation vector with the bias components in the front, too: $a^{(j)}_0 = 1$
- $a$ is the activation or output after applying the sigmoid; in the case of the input $a^{(1)} = x$
- $z^{(j+1)} = \Theta^{(j)} a^{(j)}$: dot product between weights and activations
- $a^{(j)} = g(z^{(j)})$: the activations are the result of applying the sigmoid to the $z$ vector; applying the sigmoid to a vector is the equivalent to applying the sigmoid to each of its components separately (element-wise) and assembling a new vector of equal size
- $h(x)$: the complete model; if we have $k$ layers (input layer is 1), then $h(x) = a^{(k)}= g(z^{(k)})$
- Note that if the output layer has only one unit, $\Theta$ is a row vector!

#### Interpretation and Architecture

Each layer of a neural network is a logistic regression model.
The features of the model in each layer are the outputs/activations of the previous layer.
Therefore, the neural network learns which intermediate features to generate by adjusting its weights.

By having several hidden layers, we can represent very non-linear models, which gives us a lot of flexibility.

The number of layers, their units, how they are connected and the activation functions are known as the **architecture** of the model.

### 2.4 Examples and Applications

Andrew Ng provides examples of how `AND`, `OR` and `NOT` logical functions can be implemented with a very simple perceptron. Then, these perceptrons can be combined to model the more complex logical function `XNOR` (`1` if both inputs are the same); that model uses a hidden layer. The intuition behind is that hidden layers increase the complexity of decisions.

Consider always how the sigmoid function maps the values. Note that `sigmoid(4.6) = 0.99`.

![`OR` logical function with a perceptron](./pics/or_function_perceptron.png)

![`XNOR` logical function with a MLP](./pics/xnor_function_mlp.png)

### 2.5 Multi-Class Classification

When we want to predict one of multiple classes, we use the same *one-vs-all* approach as in logistic regression. Basically, the output layer will have so many `K` units as classes `K` we want to predict, and:

- $h(x) = [h_1, h_2, ..., h_K]^T$
- If $K = 4$ and the second class is predicted: $h = [0,1,0,0]^T$

The target values or labels represented with the one-hot-encoding notation: $y$ is a column vector of size $K$ with value $1$ in the appropriate class slot, $0$ for the rest.

## 3. Exercise 3

Hand-written digits recognition with (part 1) multi-class logistic regression and (part 2) neural networks.

I completed the official exercises in Octave:

`../exercises/ex1-ex8-octave/ex3`

However, I forked also a python version of the exercises that can be used for submission also!

`~/git_repositories/ml-coursera-python-assignments`

[ml-coursera-python-assignments](https://github.com/mxagar/ml-coursera-python-assignments)

There are some relevant summary notes on `../02_LogisticRegression/ML_LogisticRegression.md` related to the python implementation.

Files provided by Coursera, located under `../exercises/ex1-ex8-octave/ex3`

- `ex3.m` - Octave/MATLAB script that steps you through part 1
- `ex3_nn.m` - Octave/MATLAB script that steps you through part 2
- `ex3data1.mat` - Training set of hand-written digits
- `ex3weights.mat` - Initial weights for the neural network exercise
- `submit.m` - Submission script that sends your solutions to our servers
- `displayData.m` - Function to help visualize the dataset
- `fmincg.m` - Function minimization routine (similar to `fminunc`)
- `sigmoid.m` - Sigmoid function

Files to complete:

- `lrCostFunction.m` - Logistic regression cost function
- `oneVsAll.m` - Train a one-vs-all multi-class classifier
- `predictOneVsAll.m` - Predict using a one-vs-all multi-class classifier
- `predict.m` - Neural network prediction function

Workflow:

- Download latest Octave version of exercise from Coursera
- Complete code in exercise files following `ex3.pdf`
- Whenever an exercise part is finished
  - Check it with `ex3` or `ex3_nn` in Octave terminal
  - Create a submission token on Coursera (exercise submission page, it lasts 30 minutes)
  - Execute `submit` in Octave terminal
  - Introduce email and token
  - Results appear

**Overview of contents:**

0. Setup: `gnuplot`
1. Part 1: Multi-Class Logistic Regression