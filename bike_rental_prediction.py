import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import pylab


bike_rentals = pd.read_csv("hour.csv")
# print(bike_rentals.head(5))


# Visualize column "cnt" histogram
plt.hist(bike_rentals["cnt"])
# plt.show()


# Correlation between each column and column "cnt"
# print(bike_rentals.corr(method="pearson")["cnt"])


def assign_label(hour):
    if hour >= 6 and hour <=12:
        return 1
    elif hour >= 13 and hour <=18:
        return 2
    elif hour >= 19 and hour <=24:
        return 3
    elif hour >= 0 and hour <=5:
        return 4
bike_rentals["time_label"] = bike_rentals["hr"].apply(assign_label)
# print(bike_rentals["time_label"])


# Setting the training set and testing set here
train = bike_rentals.sample(frac=0.8)
test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]

# Decide which column features to be trained
predictors = list(train.columns)
predictors.remove("cnt")
predictors.remove("casual")
predictors.remove("registered")
predictors.remove("dteday")

# Training linear regression here
alg = LinearRegression()
alg.fit(train[predictors], train["cnt"])
# Predict using test set
prediction = alg.predict(test[predictors])
# Calculate error using mean of all distance squared
error = np.mean((prediction - test["cnt"])**2)
print("Linear Regression Error: ")
print(error)


# Training/predicting decision tree here
tree = DecisionTreeRegressor(random_state=0, 
                             min_samples_split=4,
                            min_samples_leaf=2)
tree.fit(train[predictors], train["cnt"])

tree_pred = tree.predict(test[predictors])
tree_error = np.mean((tree_pred - test["cnt"])**2)
print("Tree Error: ")
print(tree_error)


# Training/predicting random forest here
rf = RandomForestRegressor(n_estimators=50,
                           min_samples_split=6,
                           min_samples_leaf=4)
rf.fit(train[predictors], train["cnt"])
rf_pred = rf.predict(test[predictors])
rf_error = np.mean((rf_pred - test["cnt"]) ** 2)
print("Random Forest Error: ")
print(rf_error)
