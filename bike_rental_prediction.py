import pandas as pd
import matplotlib.pyplot as plt
import pylab

bike_rentals = pd.read_csv("hour.csv")
print(bike_rentals.head(5))

plt.hist(bike_rentals["cnt"])
plt.show()

bike_rentals.corr(method="pearson")["cnt"]