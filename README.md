# Machine Learning

These are my notes on the Coursera course by Andrew Ng ["Machine Learning"](https://www.coursera.org/learn/machine-learning).

The course is divided in 11 learning weeks which cover several topics of Machine Learning. I divided the course in topic folders, which do not necessarily match the learning weeks. **Each folder contains my notes on the topic as well as the course material: slides, scripts, etc.**:

- [`01_LinearRegression/`](./01_LinearRegression/)
- [`02_LogisticRegression/`](./02_LogisticRegression)
- [`03_NeuralNetworks/`](./03_NeuralNetworks)
- [`04_MLSystemDesign/`](./04_MLSystemDesign)
- [`05_SupportVectorMachines/`](./05_SupportVectorMachines)
- [`06_UnsupervisedLearning/`](./06_UnsupervisedLearning)
- [`07_Anomaly_Recommender/`](./07_Anomaly_Recommender): Anomaly Detection and Recommender Systems
- [`08_OCR_ApplicationExample/`](./08_OCR_ApplicationExample/) 

Additionally, **the exercises are located in the folder [`exercises/ex1-ex8-octave`](./exercises/ex1-ex8-octave/)**.

I followed the original course by Andrew Ng, which has exercises in Octave/Matlab, but I completed the exercises in Jupyter notebooks running on a [conda](https://docs.conda.io/en/latest/) environment. Alternatively, if you'd like to implement the exercises in Python with Numpy, you can check this repository: [ml-coursera-python-assignments](https://github.com/mxagar/ml-coursera-python-assignments).

## Setup

In order to run my implementations from the Jupyter notebooks, you need to install the [octave kernel](https://pypi.org/project/octave-kernel/) in a [conda](https://docs.conda.io/en/latest/) environment. The following is the list of steps I followed to set things up on my Mac:

```bash
# 1) Install brew: https://brew.sh/
# 2) Install octave with brew
brew install octave
# 3) Create and activate a conda environment
conda create -n ml-octave python=3.6
conda activate ml-octave
# 4) Add the necessary packages to your environment
conda config --add channels conda-forge
conda install octave_kernel
conda install texinfo # For the inline documentation (shift-tab) to appear.
```

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

If you find this repository helpful and use it, please refer back to the original source.