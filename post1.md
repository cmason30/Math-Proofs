# Hyperparameter Tuning

## Contents

1. Introduction to Hyperparameters

2. Setting Up

4. Grid Search

5. Random Search

6. Which Should I Use?

7. Extra: Bayesian Optimization

...

![meme1](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.reddit.com%2Fr%2Flearnmachinelearning%2Fcomments%2Fpk2xcn%2Fml_meme_war_this_is_one_of_my_favourite_machine%2F&psig=AOvVaw14lChuR7UIYrDIgI4C8TjQ&ust=1635997930196000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCNjS7a2l-_MCFQAAAAAdAAAAABAD)


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



