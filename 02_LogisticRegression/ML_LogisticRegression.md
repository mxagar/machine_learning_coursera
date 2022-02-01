# Mechine Learning: Introduction and Linear Regression

These are my notes on the Coursera course by Andrew Ng ["Machine Learning"](https://www.coursera.org/learn/machine-learning).

For setup and general information, please look at `../README.md`.

This file my notes related to **logistic regression**.

Note that Latex formulae are not always rendered in Markdown readers; for instance, they are not rendered on Github, but they are on VS Code with the Markup All In One plugin.
Therefore, I provide a pseudocode of the most important equations.
An alternative would be to use Asciidoc, but rendering of equations is not straightforward either.

Overview of contents:

1. Classification and Representation
   - 1.1 Classification
   - 1.2 Hypothesis Representation: Which is the Form of $h$ in Classification?
   - 1.3 Decision Boundary
2. Logistic Regression Model
   - 2.1 Cost Function
   - 2.2 Compact Cost Function and Gradient Descent
   - 2.3 Advanced Optimization
3. Multi-class Classification: One vs. All
4. Solving the Problem of Overfitting
   - 4.1 The Problem of Overfitting
   - 4.2 Cost Function with Regularization
   - 4.3 Regularized Linear Regression: Gradient Descent and Normal Equations
   - 4.4 Regularized Logistic Regression
5. Exercise 2 (Week 3)


## 1. Classification and Representation

### 1.1 Classification

Examples of binary classification:
- Is email spam or not?
- Is transaction fraudulent or not?
- Is tumor benign or malignant?

The target or outcome is $y \in {0,1}$; if we have a multi-class classification, $y \in {0,1, 2, 3, ...}$.

One (very bad) solution could consist on applying linear regression to our problem: 
- we plot points in $(x,y)$
- we find the line $h(x)$
- and we define our threshold at the $x_t$ which yields $h(x_t) = 0.5$. Values with $x < x_t$ belong to one class, the rest to the other.

However, that has several issues:
- If a sample far away appears, our line is inclined and the threshold moved, leading to potential wrong classifications
- $h(x)$ might predict values outside from $[0,1]$, which does not make sense

![Classification via regression](./pics/classification.png)

A valid solution is the **logistic regression**, which is the basis for **classification**. It assures that $h(x) \in (0,1)$.

### 1.2 Hypothesis Representation: Which is the Form of $h$ in Classification?

Logistic regression could be called also *sigmoid regression for classification*.
The linear regression equation is passed to the sigmoid function so that $h(x) \in (0,1)$; that is what the sigmoid function does by definition: input values $z \in (-\inf, \inf)$ are mapped to $(0, 1)$, having asymptotes at $y = 0$ and $y = 1$.

$h(x) = g(\theta^T x)$

$g(z) = \frac{1}{1 + e^{-z}}$

```
h(x) = g(t * x)
g(z) = 1 / (1 + exp(-z))
```

We interpret $\hat{y} = h(x)$ to be the probability of $x$ causing $y = 1$.
Formally: $\hat{y} = h(x) = P(y = 1 | x; \theta)$, that is: the probability of $y = 1$ given $x$ and parameter $\theta$.

Intuitively, $P(y = 0 | x; \theta) = 1 - P(y = 1 | x; \theta)$.

![Logistic regression model](./pics/logistic_regression_model.png)

### 1.3 Decision Boundary

The decision boundary is the $z = \theta^T x$ equation before passing it to the sigmoid function.

Depending on the value of $\hat{y} = h(x)$ we choose one class ($y = 0$) or the other ($y = 1$). The threshold is set for $\hat{y} = h(x) = 0.5$, thus:

$\hat{y} = h(x) = \frac{1}{1 + e^{-\theta^T x}}= 0.5 \rightarrow \theta^T x =0 $

In other words: **the decision boundary in feature space is defined by**

$\theta^T x = 0$

From here, we conclude:

$\theta^T x >= 0 \rightarrow y = 1$
$\theta^T x < 0 \rightarrow y = 0$

Our goal is to fit the parameters $\theta$ to best represent that boundary.

In a simple linear case, the decision boundary is a line or a (hyperplane) that divides the feature space in two half-spaces.
Whenever we have a new sample $x$, we evaluate it with the decision boundary equation: if it yields positive, it belongs to the $y = 1$ class.

![Decision boundary](./pics/decision_boundary.png)

But we can also have polynomial regressions with the same formulae.
In that case, the boundaries start taking non-linear shapes: circles or hyper-spheres, ellipsoids, or any blobby shape (even non-convex).
Note that if they are closed, the contain samples associated with $y = 1$.

![Non-linear decision boundary](./pics/nonlinear_decision_boundary.png)

## 2. Logistic Regression Model

### 2.1 Cost Function

We define the cost or objective function to optimally fit the parameters. Let's denote that cost function $J(\theta)$ `= (1/m) * sum(cost(h(x),y))`.

Recap:

- Feature vector of size `n+1`: $x = [x_0, x_1, ..., x_n]^T$ with $x_0 = 1$

  `x = [x_0, x_1, ..., x_n]^T, x_0 = 1`

- Training samples `m`: $(x^(1),y^(1)), (x^(2),y^(2)), ..., (x^(m),y^(m))$, with $y \in {0,1}$

  `(x^(1),y^(1)), (x^(2),y^(2)), ..., (x^(m),y^(m))`, `y in {0,1}`

- Model: $\hat{y} = h(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$

  ```
  h(x) = g(t * x)
  g(z) = 1 / (1 + exp(-z))
  ```

In linear regression we use the cost function $J(\theta)$ defined as the mean of the sum of squares of all errors of predictions with respect to their associated ground truth. Now, we cannot use the same cost function, because non-linear sigmoid function in the prediction model would make that cost function $J$ non-convex. Non-convex functions have local minima or valleys; due to that, gradient descent gets stuck and no optimum is found.

Instead of squared errors, we use the logarithmic function, which does not have this problem. For better understanding, instead of using the `y` truth in the cost function, the cost function is split into two different equations, depending on the value of `y`; however, we see in the following section that we can represent the cost function in a single line with the `y` in it. The **cost of one sample** `cost(h(x),y)` in two equations is:

```
cost(h(x),y) = -log(h(x)), if y = 1
cost(h(x),y) = -log(1 - h(x)), if y = 0
```

And the **total cost** is:

```
J = (1/m) * sum(cost(h(x),y))
```

Note that `cost(h(x),y)`
- goes to `0` if `h -> y`,
- but it goes to `inf` if `h` points in the opposite direction as `y` (i.e., `y = 0, h -> 1` and `y = 1, h -> 0`)
- The `-` sign is just for putting the logarithm function upside down, don't get confused by it...

![Logictic regression: cost when y = 0](./pics/logistic_regression_cost_y0.png)

![Logictic regression: cost when y = 1](./pics/logistic_regression_cost_y1.png)

### 2.2 Compact Cost Function and Gradient Descent

The previous cost function in two parts can be rewritten in a compact way.

Two equations:

```
cost(h(x),y) = -log(h(x)), if y = 1
cost(h(x),y) = -log(1 - h(x)), if y = 0
```

Equivalent in one equation (non matter which `y` value):

```
cost(h(x),y) = -y*log(h(x)) - (1 - y)*log(1 - h(x))
```

Note that
- if `y = 1` the second term cancels and the equation becomes the first of the pair above
- if `y = 0` the first term cancels and the equation becomes the second of the pair above

In summary, the **total cost function** is (minus brought to the front):

$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \textrm{cost}(h(x^{(i)}),y^{(i)})$

$J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} y^{(i)}\log ( h(x^{(i)}) ) + (1 - y^{(i)}) \log (1 - h(x^{(i)}))$

```
J = -(1/m) * sum(y*log(h(x)) + (1 - y)*log(1 - h(x)))
```

This cost function has some nice properties:
- It can be derived from statistics using the principle of the **maximum likelihood estimation**.
- It is **convex**, so we can safely use the gradient descent with it.

The **gradient descent** requires to compute the derivative of the cost function, which has the same form as in linear regression!

$\delta = \frac{\partial J (\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i = 1}^{m} (h(x^{(i)})- y^{(i)})x_j^{(i)}$

```
delta = dJ/dtj = derivative(J,tj) = (1/m) * sum((h(x)-y)*x_j)
```

And the update formula is the same; given the learning rate `alpha`:

```
Repeat:
  compute delta(t)
  t = [t_j]
  t_j (new) <- t_j (previous) - alpha * delta
```

In vectorized form

```
theta = theta - (alpha/m)*X'*(g(X*theta)-y)
```

![Gradient descent in logictic regression](./pics/logistic_regression_gradient_descent.png)

**Important notes**:
- Even though the derivative of the cost function seems to have the same form, note that $h$ is different now! In linear regression $h = \theta^T x$, in logistic regression we apply the sigmoid function $h = g(\theta^T x)$.
- Feature scaling applies here too for accelerating and improving the gradient descent!

### 2.3 Advanced Optimization

Gradient descent is one optimization algorithm, but it is not the only one. In fact, for complex problems, other approaches have been shown to be faster and more robust, such as:
- [The conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
- [BFGS](https://en.wikipedia.org/wiki/Broyden–Fletcher–Goldfarb–Shanno_algorithm)
- [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)

These algorithms are more efficient and often do not require choosinh `alpha`. However, they are also much more complex to understand and implement; therefore, we should use efficient libraries. They often require to have the (1) cost function value `J` and its (2) derivative vector or gradient for any parameter set; therefore, although the gradient descent only requires the gradient, both values are computed in the `costFunction`. In the case of logistic regression, that would be:

```octave
function [jVal, jGrad] = costFunction(theta)
  jVal = ... % J(theta) = J = -(1/m) * sum(y*log(h(x)) + (1 - y)*log(1 - h(x)))
  jGrad = zeros(m)
  jGrad(1) = ... % dJ / dt0 = (1/m) * sum((h(x)-y))
  jGrad(2) = ... % dJ / dt1 = (1/m) * sum((h(x)-y)*x_1)
  ...
```

Then, we pass the pointer to our `costFunction` in the chosen optimization algorithm, such as:

```octave
options = optimset(‘GradObj’, ‘on’, ‘MaxIter’, ‘100’);
initialTheta = zeros(2,1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

**Important note**: the derivative vector is called gradient.

## 3. Multi-class Classification: One vs. All

While in a binary classification problem we work with only two classes $y = {0,1}$, in the multi-class classification problem we have several $q$ classes $y=\{1,2,...,k,...,q\}$ for which we predict their probabilities $h^{(k)}(x)$:

```
y in [1,2,...,k,...,q]
h(1,x) = P(y = 1 | x; theta)
h(2,x) = P(y = 2 | x; theta)
...
h(k,x) = P(y = k | x; theta)
...
h(q,x) = P(y = q | x; theta)
```

The logistic regression is a binary classification method.
One way to use it in a multi-class classification problem is to apply the **One vs. All** or **One vs. Rest** method: We create `q` artificial datasets where in each of them a class `k` is opposing to the rest of the classes, i.e., the rest of the samples belong to one class (rest). The training yields $h^{(k)}(x)$ or `h(k,x)`. Altogether we have `q` prediction functions `h`, one for each of the classes against the rest.

![Multiclass logistic regression](./pics/logistic_multiclass.png)


During inference, a sample is evaluated by all prediction functions and the class identifier associated with the highest prediction `h` is selected as our class prediction:

$\argmax_k{\{h^{(k)}(x)\}}$

```
prediction = argmax(h(1,x), h(2,x), ..., h(k,x), ..., h(q,x))
```

## 4. Solving the Problem of Overfitting

When our model fits the noise of the dataset, we say we have **overfitting**.
With overfitting, predictions are not reliable.
An approach to address overfitting consists in using **regularization**.

### 4.1 The Problem of Overfitting

Overfitting happens when we have too much features/parameters, i.e., the dimension of our model is too large (e.g., we have high degree polynomials with many curves and holes). As a consequence, the model tries to fit every sample in the dataset, the cost function becomes almost `0`, but the model fails to generalize with new samples.

The opposite of overfitting is underfitting: the model is too simplistic to fit our dataset reliably. The cost function is high and the model does not generalize well either.

Nomenclature:
- **Overfitting** is also called **high variance**. Variance relates to change; in this case, we capture all minimal changes.
- **Underfitting** is also called **high bias**. Bias means "preconception"; in this case our preconception is that the model is simplistic.

![Overfitting: problem definition](./pics/overfitting.png)

Approaches to reduce overfitting:

1. Reduce the number of features
   - Manually select which features we want to to keep.
   - Use a model selection algorithm, which select useful features (studied later in the course).
2. Regularization
   - Keep all the features, but reduce the magnitude of parameters $\theta_j$.
   - Regularization works well when we have a lot of slightly useful features.

### 4.2 Cost Function with Regularization

If we have a dataset which can be well modelled with a quadratic function (polynomial of degree 2) but we choose a model a model with degree 4, we might end with a curvy function that overfits our dataset.

```
h = t0 + t1*x + t2*x^2 + t3*x^3 + t4*x^4
```

To prevent that, we want to have small parameters of highest degrees: `t3, t4 -> 0`; that way, our model would resemble a quadratic function with small higher degree components that add minor effects.

That is the intuition behind **regularization**: we add the parameters to the cost function so that their value remains small while still fitting the dataset the best as possible. In other words, **we penalize large values of parameters**. For the linear regression:

$$ J(\theta) = \frac{1}{2 m} (\sum_{i = 1}^{m}{(h(x^{(i)})-y^{(i)})^2 + \lambda \sum_{j = 1}^{n}{\theta_{j}^{2}}})$$

```
J = (1/(2m)) * (sum(i=1:m,(h(x_i)-y_i)^2) + lambda*sum(j=1:n,theta_j^2))
```

Note that by convention the intercept parameter `theta_0` is not considered in the regularization; although using it would not affect so much.

The cost function has now two terms
1. The first one tries to fit the parmeters the best possible way so that the dataset is optimally represented (smallest deviation from samples)
2. The second tries to keep the parameter values small, which prevents overfitting. Large parameter values increase the cost, so they are penalized.

The choice of the **regularization parameter** `lambda` affects the result:
- if we take too large values for `lambda` and our parameters are going to be very small; as a consequence, we will have a biased model reduced to a flat horizontal line, i.e., too simplistic;
- if we take `lambda -> 0`, we might end up with overfitting, i.e., we are cancelling the second term.

### 4.3 Regularized Linear Regression: Gradient Descent and Normal Equations

The paramater update equation for linear regression following gradient descent is:

$\theta_j \leftarrow \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$, for $j = 0, ..., n$

```
tj <- tj - alpha*derivative(J(t),tj); j = 0:n
```

Adding the regularization term and to the cost and deriving it, we obtain:

$\theta_j \leftarrow \theta_j - \alpha[\frac{1}{m} \sum_{i=1}^{m}{(h(x^{(i)}) - y^{(i)})x^{(i)}_j} + \frac{\lambda}{m} \theta_j]$
for $j=1,...,n$

Note that $\theta_0$ is usually not considered for regularization, thus implement in practice the equation as follows:

```
t0 <- t0 - (alpha*/m)*sum(h(x[:,i]) - y[:])

tj <- tj - (alpha*/m)*(sum((h(x[:,i]) - y[:])*x[j,i]) + lambda*theta[j])

t1 <- t1 - (alpha*/m)*(sum((h(x[:,i]) - y[:])*x[1,i]) + lambda*theta[1])

t2 <- t2 - (alpha*/m)*(sum((h(x[:,i]) - y[:])*x[2,i]) + lambda*theta[2])

(with h(x[:,i]) = dot(t[:],x[:,i]))
...
```

Note that `j = 0` is treated differently, bacause `theta_0` is not considered for regularization and `x_0 = 1`.

The step formula can be rewritten as follows:

$\theta_j \leftarrow \theta_j (1 - \frac{\alpha \lambda}{m}) - \frac{\alpha}{m} \sum_{i=1}^{m}{(h(x^{(i)}) - y^{(i)})x^{(i)}_j}$

Interpretation:
- The first term `1 - alpha*lambda/m` should be `< 1` but almost `1`, that is `0.99...`; the intuition is that we shrink the `theta` step by step.
- The second term is the step we need to take to minimize the cost function; that might indeed increase `theta`, but we decrease it in the next iteration again! Notice it is the same term as before; we just added the term to shrink `theta`.

#### Normal Equation with Regularization



$\theta = (X^TX + \lambda H)^{-1}X^Ty$

$H = I$  = `eye(n+1,n+1)`,
but setting `H(0,0,) = 0`, because `theta_0` is not considered for regularization.
That is, $H$ is the identity matrix but with the `H(0,0)` element is set to `0`.

```
t = pinv(transpose(X)*X + lambda*H)*transpose(X)*y
```

If `m < n`, then $X^TX$ is not invertible; however, $X^TX + \lambda H$ is invertible even with `m < n`!

### 4.4 Regularized Logistic Regression

The cost function of logistic regression with regularization is

$$ J(\theta) = -\frac{1}{m} [\sum_{i = 1}^{m}{y^{(i)}\log(h(x^{(i)}) + (1-y^{(i)})\log(1-h(x^{(i)}))] + \frac{\lambda}{2m} \sum_{j = 1}^{n}{\theta_{j}^{2}}}$$

It turns out that end up with the same parameter update formula for gradient descent, but the `h(x)` function is obviously different:

$\theta_j \leftarrow \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$

$\theta_j \leftarrow \theta_j - \alpha[\frac{1}{m} \sum_{i=1}^{m}{(h(x^{(i)}) - y^{(i)})x^{(i)}_j} + \frac{\lambda}{m} \theta_j]$
for $j=1,...,n$

```
t0 <- t0 - (alpha*/m)*sum(h(x[:,i]) - y[:])

tj <- tj - (alpha*/m)*(sum((h(x[:,i]) - y[:])*x[j,i]) + lambda*theta[j])

t1 <- t1 - (alpha*/m)*(sum((h(x[:,i]) - y[:])*x[1,i]) + lambda*theta[1])

t2 <- t2 - (alpha*/m)*(sum((h(x[:,i]) - y[:])*x[2,i]) + lambda*theta[2])

(with h(x[:,i]) = 1 / (1 + exp(-dot(t[:],x[:,i]))))
...
```

We can use that also with advanced optimization methods; we just need to define a `costFunction` that returns the cost and its gradient with the regularization term.