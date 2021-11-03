# Hyperparameter Tuning

## Contents

1. Introduction to Hyperparameters

2. Setting Up

4. Grid Search

5. Random Search

6. Which Should I Use?

7. Extra: Bayesian Optimization

...

![meme1](data:image/jpeg;base64,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)

A hyperparameter is a parameter that is set by the user before the machine learning process begins. They are different
from model parameters, which are tuned by a machine learning algorithm automatically.
Hyperparameters come in many different forms including:

- K in KNN
- Alpha in LASSO regression
- The learning rate for a neural network 
- And the [wide variety of hyperparameters for XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html)

Essentially, the hyperparameters represent the "knobs," "dials," and "switches" that
are moved around—as we feed an algorithm data—to produce the optimal model for the dataset. 


(insert ML meme of guy standing on garbage pile)


With that in mind, it can be overwhelming to memorize all the different hyperparameters of each model and 
which values would help them function best on a dataset. Often, it is not clear how the hyperparameters
will affect performance on a dataset and many hyperparameters interact with each other in a non-linear fashion. So beware:
beecause trying to tune a model completely by hand can be an easy way to make yourself go crazy. 

Fortunately, data scientists have found a few algorithmic approaches to automating the process of 
hyperparameter tuning so that you don't have to do it yourself. In this blog, we will discuss three different
methods for tuning hyperparameters: grid search, random search, and breifly on bayesian optimization. 

## Hyperparameter Tuning with Scikit-Learn 

#### Setting up the Model and Search Space

We will use the Scikit-Learn API to set up our model and run our hyperparameter tuning. 

```python
# importing Logistic Regressiona and dataset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs

# Get blob data
X, y = make_blobs(n_samples=25000, centers=2, n_features=100, cluster_std=20)

# Create model
model = LogisticRegression()
```

Next, we want to set up the search space we will be using for our Logistic Regression hyperparameters. 

```python
# Our chosen hyperparameters
params = {
          # sample between different solvers
          'solver': ['newton-cg', 'lbfgs', 'liblinear'],
          # Set our penalty as l2
          'penalty': ['l2'],
          # Chose between a range of our regularization values
          'C': [100, 10, 1.0, 0.1, 0.01, 0.001]
        }
```

There are more hyperparameters to choose from and they can be found and added from the [logistic regression documentation.](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

With all of our hyperparameters set, we can run it through our grid search method. 

#### Grid Search

The grid search method is very similar to random search when it comes to hyperparameter tuning. 
In a nutshell, grid search will build on every hyperparameter combination possible in the given search space. 

**Pros**

- Exhaustive, it will find the best combination out of the given hyperparameter space

**Cons**

- Computationally expensive, the exhaustive search can become too computationally expensive if 
there aree too many hyperparameter combinations to try out
- Overfitting, there is the possibility of overfitting to a specific space the search

Let's finish our code: 

```python
from sklearn.model_selection import GridSearchCV

# Set up our Grid search and ad our model to it
search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy')

# Find results
result = search.fit(X, y)

print(f'Best Score: {result.best_score_}')
print(f'Best Hyperparameters: {result.best_params_}')
```

It may take some time to run the model, but after it finished it will report the best score and 
hyperparameter combinations. We can see it here:

```python
Best Score: 0.9872799999999999
Best Hyperparameters: {'C': 100, 'penalty': 'l2', 'solver': 'liblinear'}
```

Now you can go ahead and use those hyperparameters on the test set!

Next, we will do the same with a random forest model, but this time we will
try out the random search method. 

#### Random Search

The random search method is an alternative approach to hyperparameter tuning. It is pretty straightforward
in that it chooses a random set of hyperparameters within the chosen search space.

**Pros**

- Exploratory, you can give randomized search a wide distribution of hyperparameters to choose from 
to test out different search spaces.
- Less likely to overfit
- Not as computationally expensive as grid search

**Cons**

- Lots more potential for variance due to its random nature

Let's dive into it:

```python

# Import our packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Set up the model
rf = RandomForestClassifier()
``` 

Instead of setting discrete distributions of hyperparameters like we did with the logistic regression model, 
we can give continuous distributions for the randomized search method to explore:

```python
rf_params = {
    # Range of integers from 4 to 204
    'n_estimators': randint(4,200),
    'max_features' : ["auto", "sqrt", "log2"],
    # Uniform distribution of values between 0.01 and 0.02
    'min_samples_split': uniform(0.01, 0.199)
}
```

Again check out the [docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) on
random forest hyperparameters to get an understanding of which ones you may want to use. 

Laslty, we can implement them in our randomized search:

```python
random_search = RandomizedSearchCV(rf, rf_params, n_iter=100, scoring='accuracy')

rf_result = random_search.fit(X, y)

print(f'Best Score: {rf_result.best_score_}')
print(f'Best Hyperparameters: {rf_result.best_params_}')
```

Note that the n_iter argument in RandomizedSearchCV can be set to run as many iterations of hyperparameter
tuning as you want. 

And our results as shown:

```python
Best Score: 0.776114081996435
Best Hyperparameters: {'max_features': 'sqrt', 'min_samples_split': 0.02046703678839075, 'n_estimators': 118}
```



