# Bike-Rental-Prediction
Predict shared rental bikes usage. Original dataset is available [here](http://capitalbikeshare.com/system-data).
Overall, we were able to achieve 94% R^2 score using random forest model.

1. Background

Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return 
back has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return 
back at another position. Currently, there are about over 500 bike-sharing programs around the world which is composed of 
over 500 thousands bicycles. Today, there exists great interest in these systems due to their important role in traffic, 
environmental and health issues.

2. Objective

Regression: 
Predication of bike rental count hourly or daily based on the environmental and seasonal settings.

3. Files:

- hour.csv : bike sharing counts aggregated on hourly basis. Records: 17379 hours
- explore_bike_rental.py: data exploratory
- bike_rental_prediction.py: predict using linear regression, decision trees, and random forest models.
