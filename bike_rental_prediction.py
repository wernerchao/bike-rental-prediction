import pandas as pd
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


# setting the training set and testing set here
train = bike_rentals.sample(frac=0.8)
test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]
