# Mechine Learning: Introduction and Linear Regression

These are my notes on the Coursera course by Andrew Ng ["Machine Learning"](https://www.coursera.org/learn/machine-learning).

For setup and general information, please look at `../README.md`.

This file my notes related to **linear regression**.

Overview of contents:

1. Introduction
   - What is Machine Learning?
   - Supervised Learning
   - Unsupervised Learning
2. Linear Regression: Model ans Cost Function
   - Linear Regression Model
   - Cost Function

## 1. Introduction

Machine Learning is part Artificial Intelligence.
It is used in application where machines performed actions that were not specifically programmed; they can do that because they learn from past experiences or from the data they have available.

Some applications: Computer Vision, Natural Language Processing, Recommender Systems, Spam Detection, etc.

### 1.1 What is Machine Learning?

Two popular definitions:

- Arthur Samuel (1959): field of study that gives computers the ability to learn without being specifically programmed for that.
- Tom Mitchell (1998): a computer program is said to learn from experience E with respecto som task T and some performance measure P, it its performance P on T improves with experience E.

Types of Machine Learning:

- Supervised Learning: labelled data
- Unsupervised Learning: unlabelled data
- Other: Reinforcement Learning (action-reward), Recommender Systems

### 1.2 Supervised Learning

In supervised learning we have labelled data, i.e., there is a clear relationship between the input and the output data, or in other words, given an input sample we know its response, and we'd like to use that to build predictive models.

Two major supervised learning methods are used:

- Regression: we have continuous output; example: house price prediction based on flat area
- Classification: we have a discrete output, the probability of a class; example: classify if a tumor is benign according to multiple features

### 1.3 Unsupervised Learning

Unsupervised learning algorithms find structure in the data, which is usually unlabelled.

Some notable unsupervised learning methods:

- Clustering: group samples consisting of multiple features in clusters that share similar or related feature distributions. Examples:
  - Genomic groups based on gene expressions of individuals
  - Data center computer clusters
  - Market segmentation
- Dimensionality Reduction
- The Cocktail-party problem: two people speaking simultaneously, two microphones hear the speeches from different locations; the unsupervised learning algorithm is able to separate both speeches after finding structure (i.e., identifying individual voices).

## 2. Linear Regression: Model and Cost Function

We start with the example of predicting house prices based on the square feet of each of them.
We have a dataset for training consisting of:
- `m` number of training samples
- `x`: an input variable with `n` features; e.g.: x = square feet
- `y`: output or target variable; e.g.: price in 1000's of USD

The `i`-th training sample is notes as $(x^{(i)},y^{(i)})$.

### 2.1 Linear Regression Model

We build our hypothesis model $h()$ which, taken a new sample $x$ predicts the target/outcome: $\hat{y} = h(x)$.

For **linear regression, the hypothesis model** is a linear function; for the univariate case, we have:

$\hat{y} = h_{\theta}(x) = \theta_0 + \theta_1 x$.

### 2.2 Cost Function

We would like to detect the value of those $\theta_j$.
For that we build a cost function parametrized in those $\theta_j$ **parameters** and minimize it.

The **Cost Function** is the total error between the predicted target and the actual target, for each sample $i$:

$J(\theta_0,\theta_1) = \frac{1}{2 m} \sum_{i=1}^{m}(h_{\theta}(x^{(i)}) - y^{(i)})^2$

This cost function is the **mean squared error**; we find the $\theta$ parameters that minimize it:

$\theta_0, \theta_1 \leftarrow \min_{\theta_0,\theta_1} J(\theta_0,\theta_1)$

One option for minimization is to compute the derivative and equal it to 0. Note that the factor $1/2$ is for convenience, since it is cancelled when the cost function (squared) is derived.

Important intuition note: Have all the time present that The cost function $J(\theta)$ is function of the parameters! We need to *image how $J$ varies when the $\theta$ parameters are modified*. In other words, $J(\theta)$ is defined in the parameter space ($\theta = (\theta_0,\theta_1)$); in contrast, our data is defined in feature space ($x$).

### 2.3 Contour Plots of the Cost Function

If we have linear regression model with one feature ($x$), it has a $J(\theta)$ cost function with two parameters $\theta = (\theta_0,\theta_1)$, which will be a 2D quadratic surface; if projected on the parameter plane, we have a contour plot:

- Each isoline with a given $J = const.$ value is a contour; note that different lines ($\theta$) in feature space ($x$) can have the same cost $J$: these are the contour points.
- As we move on the contour plot to the minimum, the line on the feature space fits our data better.

![Contour plots by Anfrew Ng](./pics/contour_plots.png)

Our goal is to find an algorithm that is able to wander on the $J$ surface to approach it sminimum. That's **gradient descent**.

Of course, the $J$ function can be visualized in 1D or 2D only if the number of parameters are 1 or 2, respectively.

## 3. Linear Regression: Parameter Learning

### 3.1 Gradient Descent

Gradient descent can be used to minimize any function with many parameters, not only linear regression.

Given $J(\theta=(\theta_0,\theta_1,...,\theta_n))$,
we want $\theta$ which minimizes $J$:
$\min_{\theta} J(\theta)$.

Gradient descent does the following:
- Start with some $\theta=(\theta_0,\theta_1,...,\theta_n)$
- Move slowly in the direction of maximum decrease of $J$, which is the opposite direction of the gradient of $J$! That is called the steepest descent.

And that's implemented as follows:

$\theta_j \leftarrow \theta_j - \alpha*\frac{\partial}{\partial \theta_j} J(\theta_0,\theta_1)$ for $j=0,1$

The **learning rate** $\alpha$ is a *small* coefficient which decreases our step size so that we don't land on undesired spots (overshooting); intuitively, large $\alpha$ values yield too large steps that might make us further away from the minimum.

All parameters $\theta_j$ need to be updated simultaneously:

```
temp0 = theta0 - alpha*derivative(J,theta0)
temp1 = theta1 - alpha*derivative(J,theta1)
theta0 = temp0
theta1 = temp1
```

Note that:
- If we are in a local minimum, the derivative is 0, thus, parameters remain unchanged
- As gradient descent runs, we should have smaller gradient values, thus smaller steps

### 3.2 Gradient Descent for Linear Regression

