# %%
"""
## Selmon Real Estate - Price Predictor
"""

# %%
import pandas as pd

# %%
housing = pd.read_csv("data.csv")

# %%
housing.head()

# %%
housing.info()

# %%
housing['CHAS'].value_counts()

# %%
housing.describe()

# %%
# %matplotlib inline

# %%
# # For plotting histogram
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20, 15))

# %%
"""
## Train-Test Splitting
"""

# %%
# For learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:] 
    return data.iloc[train_indices], data.iloc[test_indices]

# %%
# train_set, test_set = split_train_test(housing, 0.2)

# %%
# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

# %%
from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

# %%
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# %%
strat_test_set['CHAS'].value_counts()

# %%
strat_train_set['CHAS'].value_counts()

# %%
# 95/7

# %%
# 376/28

# %%
housing = strat_train_set.copy()

# %%
"""
## Looking for Correlations
"""

# %%
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

# %%
# from pandas.plotting import scatter_matrix
# attributes = ["MEDV", "RM", "ZN", "LSTAT"]
# scatter_matrix(housing[attributes], figsize = (12,8))

# %%
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

# %%
"""
## Trying out Attribute combinations

"""

# %%
housing["TAXRM"] = housing['TAX']/housing['RM']

# %%
housing.head()

# %%
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

# %%
housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)

# %%
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# %%
"""
## Missing Attributes
"""

# %%
# To take care of missing attributes, you have three options:
#     1. Get rid of the missing data points
#     2. Get rid of the whole attribute
#     3. Set the value to some value(0, mean or median)

# %%
a = housing.dropna(subset=["RM"]) #Option 1
a.shape
# Note that the original housing dataframe will remain unchanged

# %%
housing.drop("RM", axis=1).shape # Option 2
# Note that there is no RM column and also note that the original housing dataframe will remain unchanged

# %%
median = housing["RM"].median() # Compute median for Option 3

# %%
housing["RM"].fillna(median) # Option 3
# Note that the original housing dataframe will remain unchanged

# %%
housing.shape

# %%
housing.describe() # before we started filling missing attributes

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)

# %%
imputer.statistics_

# %%
X = imputer.transform(housing)

# %%
housing_tr = pd.DataFrame(X, columns=housing.columns)

# %%
housing_tr.describe()

# %%
"""
## Scikit-learn Design
"""

# %%
"""
Primarily, three types of objects
1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method. Fit method - Fits the dataset and calculates internal parameters

2. Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a convenience function called fit_transform() which fits and then transforms.

3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions. It also gives score() function which will evaluate the predictions.
"""

# %%
"""
## Feature Scaling
"""

# %%
"""
Primarily, two types of feature scaling methods:
1. Min-max scaling (Normalization)
    (value - min)/(max - min)
    Sklearn provides a class called MinMaxScaler for this
    
2. Standardization
    (value - mean)/std
    Sklearn provides a class called StandardScaler for this

"""

# %%
"""
## Creating a Pipeline
"""

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])


# %%
housing_num_tr = my_pipeline.fit_transform(housing)

# %%
housing_num_tr.shape

# %%
"""
## Selecting a desired model for Dragon Real Estates
"""

# %%
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

# %%
some_data = housing.iloc[:5]

# %%
some_labels = housing_labels.iloc[:5]

# %%
prepared_data = my_pipeline.transform(some_data)

# %%
model.predict(prepared_data)

# %%
list(some_labels)

# %%
"""
## Evaluating the model
"""

# %%
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

# %%
rmse

# %%
"""
## Using better evaluation technique - Cross Validation
"""

# %%
# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# %%
rmse_scores

# %%
def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

# %%
print_scores(rmse_scores)

# %%
"""
Quiz: Convert this notebook into a python file and run the pipeline using Visual Studio Code
"""

# %%
"""
## Saving the model
"""

# %%
from joblib import dump, load
dump(model, 'Dragon.joblib') 

# %%
"""
## Testing the model on test data
"""

# %%
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))

# %%
final_rmse

# %%
prepared_data[0]

# %%
"""
## Using the model
"""

# %%
from joblib import dump, load
import numpy as np
model = load('Dragon.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])
print(model.predict(features))

# %%
