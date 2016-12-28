import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pylab


bike_rentals = pd.read_csv("hour.csv")
print bike_rentals.describe()
print bike_rentals.info()


def ecdf(data):
    """Compute ECDF x & y for a one-dimensional array of measurements."""
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n
    print "ecdf: ", n, x, y
    return x, y

x_temp, y_temp = ecdf(bike_rentals['temp'])


# Visualize column "cnt" histogram
bins = np.int(np.sqrt(len(bike_rentals["cnt"])))
plt.hist(bike_rentals["cnt"], bins=bins)
plt.show()
plt.plot(x_temp, y_temp, marker='.', linestyle='none')
plt.show()


# Correlation between each column and column "cnt"
print(bike_rentals.corr(method="pearson")["cnt"])


