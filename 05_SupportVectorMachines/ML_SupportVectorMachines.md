# Machine Learning: Support Vector Machines

These are my notes on the Coursera course by Andrew Ng ["Machine Learning"](https://www.coursera.org/learn/machine-learning).

For setup and general information, please look at `../README.md`.

This file my notes related to **support vector machines**.

Note that Latex formulae are not always rendered in Markdown readers; for instance, they are not rendered on Github, but they are on VS Code with the Markup All In One plugin.
Therefore, I provide a pseudocode of the most important equations.
An alternative would be to use Asciidoc, but rendering of equations is not straightforward either.

Overview of contents:

1. Large Margin Classification


## 1. Large Margin Classification

Support Vector Machines (SVM) are widely used. They provide a clean answer in supervised learning.
SVMs can be used for classification and regression.

### 1.1 Optimization Objective

The cost function of the SVMs is similar to the one in logistic regression, but some changes are applied:

1. Instead of using the continuous `-log(h(x))`, a rectified linear function is used which resembles the decaying cost; that function is called
   - `cost_1`, for the case $y = 1$, i.e., `-log(h(x))` and
   - `cost_0`, for the case $y = 0$, i.e., `-log(1-h(x))`;
   - later on we will see how this function is defined more precisely.
2. The normalizing factor $m$ is removed for later simplifications; i.e., that is equivalent as to multiplying the minimization function by $m$, which does not change the minimum point.
3. The regularization happens by scaling the real cost, not the weight term; thus, we multiply the cost term/part with the factor `C`, and the weight term/part with nothing. In practice, this is the same as multiplying the complete function with `C = 1 / lambda`.
   - For logistic regression, we had: `A + lambda*B`;
   - Now we have: `C*A + B`, with `C = 1 / labmda`.

So:

The **logistic regression cost minimization function**:

$J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} y^{(i)}\log ( h(x^{(i)}) ) + (1 - y^{(i)}) \log (1 - h(x^{(i)})) + \frac{\lambda}{2 m} \sum_{j = 1}^{n}{\theta_{j}^{2}}$

$h(x) = g(\theta^{T}x) = \frac{1}{1 + e^{-\theta^T x}}$


```
J = -(1/m) * sum(y*log(h(x)) + (1 - y)*log(1 - h(x))) + (lambda/(2m)) * sum(j=1:n,theta_j^2)
```
```
h(x) = g(t * x) = 1 / (1 + exp(-t * x))
```

The **support vector machine cost minimization function**:

$J(\theta) = - C \sum_{i=1}^{m} y^{(i)}\textrm{cost}_1(\theta^{T}x^{(i)}) + (1 - y^{(i)}) \textrm{cost}_0(\theta^{T}x^{(i)}) + \frac{1}{2} \sum_{j = 1}^{n}{\theta_{j}^{2}}$

$\textrm{cost}_1(z) = \max (0,-m_1 x -1)$, $m_1$ unimportant (explained later)

$\textrm{cost}_0(z) = \max (0,m_0 x + 1)$, $m_0$ unimportant (explained later)

```
J = -C * sum(y*cost_1(t*x) + (1 - y)*cost_0(t*x)) + (1/2) * sum(j=1:n,theta_j^2)
```

For both logistic regression and SVM, the hypothesis or model function should yield:

```
h(x) = {1 if theta*x >= 0, 0 otherwise}
```

![SVM: Alternative View of Logistic Regression](./pics/svm_logistic_regression.png)

### 1.1 Large Margin Classifier (Intuition)

Sometimes SVMs are called large margin classifiers.

