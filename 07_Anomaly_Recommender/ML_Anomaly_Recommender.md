# Machine Learning: Anomaly Detection and Recommender Systems

These are my notes on the Coursera course by Andrew Ng ["Machine Learning"](https://www.coursera.org/learn/machine-learning).

For setup and general information, please look at `../README.md`.

This file my notes related to **anomaly detection** and **recommender systems**.

Note that Latex formulae are not always rendered in Markdown readers; for instance, they are not rendered on Github, but they are on VS Code with the Markup All In One plugin.
Therefore, I provide a pseudocode of the most important equations.
An alternative would be to use Asciidoc, but rendering of equations is not straightforward either.

Overview of contents:

1. Anomaly Detection
   - 1.1 Problem Motivation
   - 1.2 (Univariate) Gaussian Distribution
   - 1.3 Anomaly Detection Algorithm Based on the Gaussian Distribution
   - 1.4 Developing and Evaluating an Anomaly Detection System
   - 1.5 Anomaly Detection vs. Supervised Learning
   - 1.6 Choosing Features in Anomaly Detection
   - 1.7 Multivariate Gaussian Distribution
   - 1.8 Anomaly Detection using the Multivariate Gaussian Distribution
   - 1.9 Univariate vs. Multivariate Gaussians
2. Recommender Systems



## 1. Anomaly Detection

### 1.1 Problem Motivation

In **Anomaly Detection**, given a dataset of unit measurements $X = x^{(1)}, ..., x^{(m)}$, we model the probability of a new examples to belong to the dataset: $p(x \in X)$:

- If `p(x_test) < epsilon`, `x_test` does not belong to the distribution of the dataset: ANOMALY!
- If `p(x_test) > epsilon`, `x_test` belong to the distribution of the dataset.

![Anomaly Detection: Problem Motivation](./pics/anomaly_detection_definition.png)

Typical applications:

- Fraud detection: several features or normal activities are recorded to build the distribution of the dataset (e.g., number of clicks, time with window open, etc.); if a new data point arises which is an outlier, then a red flag is raised.
- Manufacturing: features of machines are recorded (e.g., heat vibrations, etc.) to predict anomalous functioning.
- Data centers: features of computers and network are recorded to detect anomalous situations that might be associated to something not working properly (e.g., CPU load, network traffic, memory use, etc.).

### 1.2 (Univariate) Gaussian Distribution

A variable that follows the Gaussian or Normal distribution with mean $\mu$ and variance $\sigma^2$:

$$x \sim N(\mu, \sigma^2)$$

```
x ~ N(mu, sigma^2)
```

The probability of $x$ of belonging tp the Gaussian distribution:

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi} \sigma} \exp{(- \frac{(x-\mu)^2}{2\sigma^2})}$$

```
p(x; mu, sigma^2) = (1 / (sqrt(2*pi)*sigma)) * exp(-(x-mu)^2/(2*sigma^2))
```

Note:

- Bell-shaped distribution.
- $\mu$ is the center.
- $\sigma$ is the width/spread.
- Area below curve is 1.

Parameter estimation (mean and standard deviation squared):

$$\mu = \frac{1}{m}\sum_{i=1}^{m}x^{(i)}$$ 

$$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(x^{(i)}-\mu)^2$$ 

### 1.3 Anomaly Detection Algorithm Based on the Gaussian Distribution

Our goal is to model the probability `p(x)`. The probability function is the product of the Gaussian probability functions of all the features. Given a feature space of $n$ features:

$$p(x) = p(x_1, \mu_1, \sigma_1^2) p(x_2; \mu_2, \sigma_2^2) ... p(x_n; \mu_n, \sigma_n^2)$$

$$p(x) = \prod_{j=1}^{n} p(x_j, \mu_j, \sigma_j^2)$$

That formula assumes that features are independent from each other; in practice, the algorithm works even when that assumption is not really fulfilled.

Thus, the algorithm is very simple:

- Get all `m` examples, each of dimension `n` (i.e., `n` features).
- Compute the mean and the standard deviation in each dimension: `m_j`, `s_j`. That can be done ina vectorized way.
- The probability model `p(x)` is the product of all `n` Gaussians of each feature.
- Evaluate `p()` given a new `x_test`: if `p(x_test) < epsilon`, ANOMALY.

If we look at an `n = 2` example, `p(x)` is a bell-shaped surface that emerges from the feature space. All possible data-points from the feature space that evaluate a large enough `p` height (`p > epsilon`) belong to region of NORMALITY, the ones outside are ANOMALIES.

![Anomaly Detection: Example](./pics/anomaly_detection_example.png)

### 1.4 Developing and Evaluating an Anomaly Detection System

Anomaly Detection is an unsupervised learning technique because we simply need *normal* data (unlabelled). However, we typically use labelled data, or in other words, a small set of *anomaly examples* to choose the value of `epsilon`; therefore, the algorithm is somewhere between supervised and unsupervised learning.

A typical setting should work like this:

- We get around `m = 10,000` normal examples; additionally, we need `20` *anomalies*. That is the typical proportion we could expect. Note that we have much more *normal* examples than *anomalies*, so the dataset is not balanced.
- We split the dataset in:
  - Training: 6,000 normal; no anomalies are used to train/fit the model.
  - Cross Validation: 2,000 normal; 10 anomalies.
  - Test: 2,000 normal; 10 anomalies.
- We build the model `p(x)` as explained: `m,s` for each of the `n` features.
- Our model looks like this:
  - `y = 1 if p(x) < epsilon`: ANOMALY
  - `y = 0 if p(x) >= epsilon`: NORMAL
- We vary `epsilon` and for each epsilon we compute the `F1` score on the cross validation set. We pick the `epsilon` that yields the best `F1`. We use as metric the `F1` and never the accuracy, because the dataset is skewed! We have also alternative metrics, but `F1` is just one number, which makes the choice of `epsilon` easier; alternative metric:
  - Confusion matrix: TP, TN, FP, FN
  - Precision & Recall
- With all parameters chosen, we evaluate the `F1` of our final model on the test split.

### 1.5 Anomaly Detection vs. Supervised Learning

In theory, we could use a supervised learning algorithm (e.g., logistic regression) instead of the unsupervised learning algorithm for anomaly detection. However, anomaly detection is very well suited for these cases:

- We have few anomaly examples (`y = 1`), in contrast to normal examples (`y = 0`)
- We don't have all possible anomaly cases to model; e.g., we might not know all possible ways a machine can fail, but we know when a machine is not failing.

### 1.6 Choosing Features in Anomaly Detection

In order to improve our anomaly detection algorithm, we can:

1. Normalize non-normal features applying transformations.
2. Look for features that clearly distinguish anomalies.

The **first approach** relies on the fact that we assume each feature is normally distributed. Even though the algorithm works when the assumption is broken, we can improve its performace if we guarantee it. To that end, we should **always plot histograms of the features** and try these families of transformations if their distributions don't look normal:

`x_j -> log(x_j + a)`; `a` is any constant we can try manually: `0, 1, 2.5, ...`

`x_j -> (x_j)^b`; `b` is often a fraction we can try manuall: `1/2, 1/3, 1/10, ...`

We iteratively try different transformations and plot them until we achieve bell-shaped distributions.

![Anomaly Detection: Transforming Non-Gaussian Features](./pics/anomaly_detection_nongaussian.png)

The **second approach** consists is performing **error analysis**, i.e., we pick errors and analyze which are their unique properties that distinguish them from the normal data points. The effect is that we extend the dimensionality of the feature space; while in the previous reduced space the anomaly examples that produced errors where close to the normal ones, with the new dimension, they are far away.

![Anomaly Detection: Error Analysis](./pics/anomaly_detection_error_analysis.png)

Sometimes **new features can be a combination of available ones, so that the combined feature takes unusually large or small values in the event of an anomaly**!

For example: let's image that in a data-center a computer can get unresponsive due to a bad program that blocks the CPU, without using the network (e.g., a "while-true" case); then, if we already have the features `cpu_load` and `network_traffic`, a reasonable new feature to distinguish cases like the aforementioned would be `cpu_load / network_traffic`. The reason is that high values of that new feature indicate the anomaly in question.

### 1.7 Multivariate Gaussian Distribution

The univariate model built until now assumes that the features are independent from one another. However, when correlations between features are not negligible (which is usually the case), the model does not capture correctly the multi-dimensional distribution, and thus, the predicted `p` values are not correct.

What is happening is that instead of having skewed, rotated and stretched bell-shapes, as we should, we have blobby and round curves, which do not capture the underlying distribution.

![Multivariate Gaussian: Motivation](./pics/multivariate_gaussian_motivation.png)

In such cases, instead of using a univariate Gaussian distribution obtained after multiplying supposedly independent Gaussians, it is better to use the **Mulivariate Gaussian** distribution, which is not a product, but a multivariate function:

$x \in \mathbf{R}^n \rightarrow p(x) \in \mathbf{R}$

Parameters: 

- Vectorial mean: $\mu \in \mathbf{R}^n$
- Covariance matrix: $\Sigma \in \mathbf{R}^{n\times n}$

$$p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} \vert\Sigma\vert^{1/2}} \exp{(- \frac{1}{2} (x-\mu)^T \Sigma^{-1}(x-\mu) )}$$

```
p(x; mu, Sigma) = (1 / ((2*pi)^(n/2)*sqrt(det(Sigma)))) * exp(-(x-mu)^T * inv(Sigma)^2 * (x-mu))
```

Note that the **covariance matrix** $\Sigma \in \mathbf{R}^{n \times n}$ is symmetric. As examples, with `n = 2`, it has this form:

```
Sigma = [s_11 s_12; s_21 s_22]
s_11: variance of feature x_1
s_22: variance of feature x_2
s_12 = s_21: covariance of features x_1 & x_2

x = [x_1 x_2]' (column)
Sigma = sum(x * x', i = 1:m)
```

The covariance matrix has these properties:

- It is symmetric, i.e. `s_ij = s_ji if i != j`.
- Only if two features `x_i` and `x_j` are correlated is `s_ij !=0`, otherwise, Sigma is diagonal.
- It must be invertible to use it in our model, i.e., it must have non-zero determinant; a requirement for that is `m > n`.
- The mean vector `mu` is the point around which the Gaussiean is centered
- The diagonal values `s_jj` denote the stretch of the Gaussian in each axis `j`; the biggest value yields proportionally the biggest stretch.
- The non-diagonal values `s_ij` denote the stretch in the direction/axis which is combination of both `i & j`; the sign denotes the sign of the slope.
- The non-diagonal values `s_ij` are related to the correlation between variables `x_i` and `x_j`:

`corr(x_1, x_2) = s_12 / sqrt(s_11*s_22)`


![Covariance Matrix: Examples](./pics/covariance_matrix_examples.png)

### 1.8 Anomaly Detection using the Multivariate Gaussian Distribution

The algorithm is basically the same:

- Get all `m` examples, each of dimension `n` (i.e., `n` features).
- Compute the mean vector `mu` and the covariance matrix `Sigma`.
- The probability model `p(x)` is aforementioned formula.
- Evaluate `p()` given a new `x_test`: if `p(x_test) < epsilon`, ANOMALY.

### 1.9 Univariate vs. Multivariate Gaussians

It can be shown that the Univariate Gaussian is the Multivariate Gaussian when the latter is axis aligned, i.e., the covariance matrix diagonal.

Therefore, if we have correlations between features, the univariate Gaussian will not capture them, unless we introduce them as new features, e.g., `x_3 = x_1 / x2`.

However, the univariate Gaussian is more frequently used, because it is computationally much less expensive and, thus, it scales much better (we don't need to compute the inverse, etc.).

Summary of differences and practical advise:

- `n`: If we have many features (`n > 10,000`), we can use the univariate version, because it is mucch less computationally expensive.
- Correlations: In the univariate version, we should add manually combinations of features is we see there is a string correlation between them, e.g., `x_3 = x_1 / x2`; otherwise, that correlation won't be captured. In the multivariate version, there is no need for that.
- `m`: In the multivariate version, we need to have `m > n` - recommended: `m >= 10*n`; in the univariate version, `m` can be small!
- Redundant features: the inverse of the covariance won't exist (`det(Sigma) = 0`) if a variable is a linear combination of others or itself; thus, we need to detect those redundant variables and erase them. The univariate option does not have this issue.

![Univariate vs. Multivariate Gaussians](./pics/univariate_vs_multivariate_gaussians.png)

## 2. Recommender Systems

Even though recommender systems are not that popular in the academia, they are very popular in the industry, because they bring a lot of money (e.g., Amazon, Netflix, etc.). The key idea is to predict missing values based on similar profiles.

### 2.1 Problem Formulation

The following slide defines very well problem and the goal of a recommender system:

![Recommender Systems: Problem Formulation](./pics/recommender_systems_problem_furmulation.png)

We have a matrix formed by `items of interest vs. user profiles` (e.g., movies vs. users), and in each cell we have a rating given by each user to the corresponding item. Now, some notions:

- Each user has not rated all the items.
- An item doesn't need to be been rated by all users.
- We can expect that similarities between users and between items: for instance, people who like romantic movies, or movies which have romantic content.
- Thus, we should be able to predict the missing values in the matrix!

Notation:

- `n_u`: number of users (profiles)
- `n_m`: number of movies (items)
- `r(i,j)`: `1`, if user `j` has rated movie `i`, `0` otherwise
- `y^(i,j)`: rating given by user `j` to movie `i`, only if `r(i,j) = 1`

The **goal: estimate missing `y^(i,j)` values**.

In the slide above, we see that we have two clusters of users and movies:

- Users who like romance vs. action.
- Movies with romance vs. action content.

### 2.2 Content-Based Recommendations

One approach to estimate missing values are the *content-based* recommendations. This assumes we have feature vectors of the items/movies, related to their content, e.g., degree of romance, action, etc. That is the major disadvantage of the approach: we need to have feature vectors!

Given those feature vectors, linear regression is performed across all movies and all users so that each user gets a parameter vector `theta`. Thus, given a movie with a feature vector `x`, the estimated rating of a user of that movie is `theta*x`.

![Content-Based Recommender Systems](./pics/content_based_parameters.png)

The key concepts are:
- Each movie `i` has its own feature vector `x^(i)`, with `n` features and `x_0 = 1`, so size `n + 1`. Each element in `x` is how much the movie is aligned with a feature (e.g., it has romance, action, etc.).
- Each user `j` gets a parameter vector `theta^(j)`. Each element of `theta` is the importance given by the user to each feature (e.g., it has romance, action, etc.).
- `r(i,j) = 1` when a user `j` has rated movie `i` with `y^(i,j)`; the total movies rated by a user is `m^(j)`.

The estimation of a movie `i` rating by user `j` will be:

```
y_hat = (theta^(j))' *

theta^(j): (n + 1) x 1; one for each user j = 1:n_u
x^(i): (n + 1) x 1; one for ach movie i = 1:n_m
```

While the real rating is:

```
y^(i,j), if r(i,j) = 1
```

With that, usual linear regression can be applied, even regularized, and an optimization algorithm of any kind (e.g., gradient descent) can be used, because the cost function `J` and its derivative are known. 

The cost function is the sum of squared differences `y_hat - y` over all movies `i` and users `j` if `r(i,j) = 1`. In a regular linear regression, we would have normalized the sum with `m^(j)`, but here we drop it.

Then, the cost function and the optimization with gradient descent is defined in the following slides:

![Content-Based Recommender System: Cost Function](./pics/recommender_cont_based_cost.png)

![Content-Based Recommender System: Cost Minimization](./pics/recommender_cont_based_cost_minimization.png)

### 2.3 Collaborative Filtering




