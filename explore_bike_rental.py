import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pylab


bike_rentals = pd.read_csv("hour.csv")
print bike_rentals.describe()
print bike_rentals.info()

# Visualize column "cnt" histogram
bins = np.int(np.sqrt(len(bike_rentals["cnt"])))
plt.hist(bike_rentals["cnt"], bins=bins)
plt.show()


# Correlation between each column and column "cnt"
print(bike_rentals.corr(method="pearson")["cnt"])


