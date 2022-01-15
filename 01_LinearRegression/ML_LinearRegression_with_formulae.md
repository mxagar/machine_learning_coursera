# Mechine Learning: Introduction and Linear Regression

These are my notes on the Coursera course by Andrew Ng ["Machine Learning"](https://www.coursera.org/learn/machine-learning).

For setup and general information, please look at `../README.md`.

This file my notes related to **linear regression**.

Overview of contents:

1. Introduction
   - What is Machine Learning?
   - Supervised Learning
   - Unsupervised Learning
2. Linear Regression with One Variable: Model ans Cost Function
   - Linear Regression Model
   - Cost Function
   - Contour Plots of the Cost Function
3. Linear Regression: Parameter Learning
    - Gradient Descent
    - Gradient Descent for Linear Regression
4. Linear Algebra Review
5. Linear Regression with Multiple Variables
   - Multivariate Linear Regression
   - Computing Parameters Analytically
6. Octave/Matlab Tutorial

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

## 2. Linear Regression with One Variable: Model and Cost Function

We start with the example of predicting house prices based on the square feet of each of them.
We have a dataset for training consisting of:
- `m` number of training samples
- `x`: an input variable with `n` features; e.g.: x = square feet
- `y`: output or target variable; e.g.: price in 1000's of USD

The `i`-th training sample is notes as <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/f0105bbd97a5c945533e3f55cea4355e.svg?invert_in_darkmode" align=middle width=69.62915025pt height=29.190975000000005pt/>.

### 2.1 Linear Regression Model

We build our hypothesis model <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/76483451b363f32e6f1162bf89f589ad.svg?invert_in_darkmode" align=middle width=22.25654804999999pt height=24.65753399999998pt/> which, taken a new sample <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> predicts the target/outcome: <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/471fb49ec8bba21bdd0bdec6be2b9e44.svg?invert_in_darkmode" align=middle width=62.218366649999986pt height=24.65753399999998pt/>.

For **linear regression, the hypothesis model** is a linear function; for the univariate case, we have:

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/62e54f8143ebb544b027ece97022c87a.svg?invert_in_darkmode" align=middle width=151.24204755pt height=24.65753399999998pt/>.

### 2.2 Cost Function

We would like to detect the value of those <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/455b7e5df6537b98819492ec6537494c.svg?invert_in_darkmode" align=middle width=13.82140154999999pt height=22.831056599999986pt/>.
For that we build a cost function parametrized in those <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/455b7e5df6537b98819492ec6537494c.svg?invert_in_darkmode" align=middle width=13.82140154999999pt height=22.831056599999986pt/> **parameters** and minimize it.

The **Cost Function** is the total error between the predicted target and the actual target, for each sample <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/>:

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/eb9aa8059c97f6f3bed23dfe7ef9ed07.svg?invert_in_darkmode" align=middle width=265.91932455pt height=29.190975000000005pt/>

This cost function is the **mean squared error**; we find the <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> parameters that minimize it:

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/77c791a1591aacf4da0c8aa371fb4011.svg?invert_in_darkmode" align=middle width=184.24657679999999pt height=24.65753399999998pt/>

One option for minimization is to compute the derivative and equal it to 0. Note that the factor <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/d5d5564ce0bb9999695f32da6ba7af42.svg?invert_in_darkmode" align=middle width=24.657628049999992pt height=24.65753399999998pt/> is for convenience, since it is cancelled when the cost function (squared) is derived.

Important intuition note: Have all the time present that The cost function <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode" align=middle width=31.655311049999987pt height=24.65753399999998pt/> is function of the parameters! We need to *image how <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/> varies when the <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> parameters are modified*. In other words, <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode" align=middle width=31.655311049999987pt height=24.65753399999998pt/> is defined in the parameter space (<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/e6d9f1dff159f22dcd94e75cb2bf8162.svg?invert_in_darkmode" align=middle width=80.36517884999999pt height=24.65753399999998pt/>); in contrast, our data is defined in feature space (<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/>).

### 2.3 Contour Plots of the Cost Function

If we have linear regression model with one feature (<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/>), it has a <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/ca79e4e55e2ba419b202c4c9576a0d0e.svg?invert_in_darkmode" align=middle width=31.655311049999987pt height=24.65753399999998pt/> cost function with two parameters <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/e6d9f1dff159f22dcd94e75cb2bf8162.svg?invert_in_darkmode" align=middle width=80.36517884999999pt height=24.65753399999998pt/>, which will be a 2D quadratic surface; if projected on the parameter plane, we have a contour plot:

- Each isoline with a given <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/6887c4ebff40537e535ff695ea24ff57.svg?invert_in_darkmode" align=middle width=75.77051294999998pt height=22.465723500000017pt/> value is a contour; note that different lines (<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/>) in feature space (<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/>) can have the same cost <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/>: these are the contour points.
- As we move on the contour plot to the minimum, the line on the feature space fits our data better.

![Contour plots by Anfrew Ng](./pics/contour_plots.png)

Our goal is to find an algorithm that is able to wander on the <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/> surface to approach it sminimum. That's **gradient descent**.

Of course, the <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/> function can be visualized in 1D or 2D only if the number of parameters are 1 or 2, respectively.

## 3. Linear Regression: Parameter Learning

### 3.1 Gradient Descent

Gradient descent can be used to minimize any function with many parameters, not only linear regression.

Given <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/86db502860a5acd7f9844cb4ecd24baa.svg?invert_in_darkmode" align=middle width=148.82223015pt height=24.65753399999998pt/>,
we want <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> which minimizes <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/>:
<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/533172db8c1089b656dbbfcf976c4d85.svg?invert_in_darkmode" align=middle width=69.22948064999999pt height=24.65753399999998pt/>.

Gradient descent does the following:
- Start with some <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/eef88651d10cb641ecc4edbce1c05c3c.svg?invert_in_darkmode" align=middle width=125.34044654999998pt height=24.65753399999998pt/>
- Move slowly in the direction of maximum decrease of <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/>, which is the opposite direction of the gradient of <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/>! That is called the steepest descent.

And that's implemented as follows:

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/effb05242aefbdf924b65e0f5ab497ab.svg?invert_in_darkmode" align=middle width=186.09565259999997pt height=28.92634470000001pt/> for <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/a8f306a6f9035370006c5350c9aa4aa8.svg?invert_in_darkmode" align=middle width=53.37235034999999pt height=21.68300969999999pt/>

The **learning rate** <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is a *small* coefficient which decreases our step size so that we don't land on undesired spots (overshooting); intuitively, large <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> values yield too large steps that might make us further away from the minimum.

All parameters <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/455b7e5df6537b98819492ec6537494c.svg?invert_in_darkmode" align=middle width=13.82140154999999pt height=22.831056599999986pt/> need to be updated simultaneously:

```
temp0 = theta0 - alpha*derivative(J,theta0)
temp1 = theta1 - alpha*derivative(J,theta1)
theta0 = temp0
theta1 = temp1
```

Note that:
- If we are in a local minimum, the derivative is 0, thus, parameters remain unchanged, reached that point.
- Indeed, for the general case, the gradient descent algorithm is susceptible to falling into local optima.
- As gradient descent runs, we should have smaller gradient values, thus smaller steps.
- A fixed `alpha` value leads also to optima, we don't need to change it really.

### 3.2 Gradient Descent for Linear Regression

The only key term to remains to be obtained is the partial derivative.

Given:

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/eb9aa8059c97f6f3bed23dfe7ef9ed07.svg?invert_in_darkmode" align=middle width=265.91932455pt height=29.190975000000005pt/>

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/24dbdda02fa945546d4815840ec1d074.svg?invert_in_darkmode" align=middle width=286.49950394999996pt height=29.190975000000005pt/>

We can easily compute:

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/d8f15a5a2133381049ed5ac56c56b82f.svg?invert_in_darkmode" align=middle width=295.77173339999996pt height=29.190975000000005pt/>

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/9c3219fda539a271eae88be2ee24146b.svg?invert_in_darkmode" align=middle width=320.09165265pt height=29.190975000000005pt/>

Now, we could simply plug these terms to our algorithm!

Even though in the general case the cost function might have many local optima, for the linear regression we will have a unique optimum. That is so because the cost function for linear regression is **convex**.

Gradient descent can be:

- **Batch gradient descent**: all <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/> samples are taken into consideration to compute the gradient; in other words, what we have done above. It is considered to be an epoch when a single derivative computation with all samples is done.
- **Stochastic gradient descent**: a unique random sample is taken into consideration to computer the gradient; that is common when processing the gradient of all samples is very expensive, e.g., with convolutional neural networks that work on images. It is considered to be an epoch when all samples have been independently processed.
- **Mini-batch gradient descent**: mini-batches of samples are used instead of all <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/> samples. That is something between the two previous approaches. The term `batch_size` commonly used in deep learning refers to the size of that mini-batch. If `batch_size = 1`, we assume we have stochastic gradient descent.

**Important remark**: It is actually possible to obtain the normal equations of the linear regression without the need of running the gradient descent. However, gradient descent is applied usually for large datasets, because it is more stable. Additionally, gradient descent is necessary when the closed form of <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode" align=middle width=10.69635434999999pt height=22.465723500000017pt/> is unknown.

## 4. Linear Algebra Review

I will not extensively make notes in this section, it is ver basic algebra.

Some Octave/Matlab code is shown. In order to install the Octave kernel for Jupyter:

```bash
brew install octave
conda config --add channels conda-forge
conda install -c conda-forge octave_kernel
# shift-tab doc
conda install texinfo
```

See the notebook `01_0_LinearAlgebra.ipynb`.

Covered topics and notation:
- Matrices (n rows x m columns), vectors (n x 1 rows)
  - A_ij: element in row i and column j from matrix A
- Addition and Scalar Multiplication
- Matrix Vector Multiplication
  - Linear models can be written in matrix notation: <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/57ad218cebfe0abb1c5f29865f6cff59.svg?invert_in_darkmode" align=middle width=54.47093849999998pt height=22.831056599999986pt/>
- Matrix Matrix Multiplication
- Matrix Multiplication Properties
   - Non-commutative: $A \times B \neq B \times A$
   - Associative: $(A \times B) \times C = A \times (B \times C)$
   - Identity matrix: $I \times A = A$
- Inverse and Transpose
  - Only square matrices can be inverted.
  - (Square) Matrices that cannot be inverted are called singular.

## 5. Linear Regression with Multiple Variables

### 5.1 Multivariate Linear Regression

#### Multiple Features

Instead of having a unique feature (e.g., in the house price prediction example: square feet), now we have several features and build a vector <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> of `n` features:

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/dbcb7ef81038c4fdf9da2fa2fa4f94bf.svg?invert_in_darkmode" align=middle width=137.4769275pt height=27.6567522pt/>, size `(n+1)x1`

Notation:

- `n`: number of features
- `m`: number of samples
- <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/ad769e751231d17313953f80471b27a4.svg?invert_in_darkmode" align=middle width=24.319919249999987pt height=29.190975000000005pt/>: sample <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.663225699999989pt height=21.68300969999999pt/> of a total of `m`
- <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/545f0a66cfa03a135f3b47954b0b0247.svg?invert_in_darkmode" align=middle width=24.319919249999987pt height=34.337843099999986pt/>: feature <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.710416999999989pt height=21.68300969999999pt/> of the complete feature vector consisting of `n` unique features

![Multiple features](./pics/multiple_features.png)

The hypothesis/model formula is updated:

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/2e8133420c265d0034f19fa76d86859e.svg?invert_in_darkmode" align=middle width=260.9033163pt height=22.831056599999986pt/>

By convetion: <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/08da0eff87a450c1af2ef3a27bf4243e.svg?invert_in_darkmode" align=middle width=46.90628744999999pt height=21.18721440000001pt/>, and it is associated to the intercept parameters <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/1a3151e36f9f52b61f5bf76c08bdae2b.svg?invert_in_darkmode" align=middle width=14.269439249999989pt height=22.831056599999986pt/>.

Then, in matrix/vector notation:

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/d6d6080404f9984b60a5f11ad02a6917.svg?invert_in_darkmode" align=middle width=66.75002069999998pt height=27.6567522pt/>,

being

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/ab9428c85f6e42a98db826990be2496d.svg?invert_in_darkmode" align=middle width=153.43761675pt height=27.6567522pt/>, size `(n+1)x1`

<img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/78ecc20f17c48f35508bbeaa358aee38.svg?invert_in_darkmode" align=middle width=154.49931915pt height=27.6567522pt/>, size `(n+1)x1`

Note that both <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> and <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/> are column vectors and that <img src="https://rawgit.com/in	git@github.com:mxagar/machine_learning_coursera/main/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/> is transposed for the scalar product between vectors.

#### Gradient Descent for Multiple Features

