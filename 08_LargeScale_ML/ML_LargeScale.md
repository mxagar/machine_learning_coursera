# Machine Learning: Large Scale Machine Learning

These are my notes on the Coursera course by Andrew Ng ["Machine Learning"](https://www.coursera.org/learn/machine-learning).

For setup and general information, please look at `../README.md`.

This file my notes related to **large scale machine learning**.

Note that Latex formulae are not always rendered in Markdown readers; for instance, they are not rendered on Github, but they are on VS Code with the Markup All In One plugin.
Therefore, I provide a pseudocode of the most important equations.
An alternative would be to use Asciidoc, but rendering of equations is not straightforward either.

Overview of contents:

1. Gradient Descent with Large Datasets
2. Advanced Topics

## 1. Gradient Descent with Large Datasets

Nowadays, it is becoming more and more usual to have large datasets, consisting of several millions of data points. However, before investing in acquiring more data points, we should however check whether it makes sense or not: that is done by testing if we have a high bias (simplistic) or higher variance model:

- If the model is simplistic, adding more examples won't be helpful; we need to add new features.
- If the model has higher variance, adding more example is helpful: the error is decreased.

In order to check in which case we are, we plot the `J_train` and `J_cv` as a function of used `m` examples:

- If theiir curves converge with few examples `m = 1,000`, we have a high bias problem; we need to add more features and test again before adding more examples.
- If their curves are quite separate but approach slowly, we have a high variance model, we can add more examlples to improve it.

Note that this check is essential: we do it with few examples `m = 1,000` to test whether it makes sense to increase to `m = 100,000,000` examples, which is presumably very expensive (computationally and also in terms of effort to be done).

![Check bias and variance for deciding whether to increase `m`](./pics/check_bias_variance_for_increasing_m.png)