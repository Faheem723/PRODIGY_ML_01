'''
TASK-1:
    Implement a linear regression model to predict the prices of houses based on their square footage
    and the number of bedrooms and bathrooms.
'''

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# DATA COLLECTION
# Load the dataset
df = pd.read_csv("D:\Internship\Prodigy InfoTech\house_prices_dataset.csv")

# Displaying the first few rows of the dataset
print("Sample of raw dataset:")
print(df.head())
print()



# DATA PREPROCESSING
# Checking for missing values
print("Checking for missing values:")
print(df.isnull().sum())
print()

# Handling missing values
median_bathrooms=df.bathrooms.median()
print("Median of Bathrooms: ",median_bathrooms)
df.bathrooms=df.bathrooms.fillna(median_bathrooms)  #Fills rows with median values
df = df.dropna()                                    #Removes the rows with missing values
print()

print("Handling the missing values of bedrooms and bathrooms:")
print(df)
print()



# DATA TRANSFORMING
# Split the data into features (X) and target variable (y)
X = df[['Area', 'bedrooms', 'bathrooms']]
y = df['price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# DATA MODELLING
# Create a linear regression model
model = LinearRegression()

# Fit the model on the training set
model.fit(X_train, y_train)

print("Coefficients of the equation: ", model.coef_)
print()

print("Intercept of the equation: ",model.intercept_)
print()

# Making predictions on the test set
y_predicted = model.predict(X_test)
print("Predcited Prices of the testing data based on the model:")
print(y_predicted)
print()
print("Actual Prices of testing data:")
print(y_test.to_numpy())    #Converting the data in array form
print()



# Evaluating the model
mse = mean_squared_error(y_test, y_predicted)
print(f'Mean Squared Error: {mse}')



# DATA VISUALIZATION
# Visualizing the results
plt.scatter(X_test['Area'], y_test, color='black', label='Actual Prices')
plt.scatter(X_test['Area'], y_predicted, color='blue', label='Predicted Prices')

# Plotting the Regression trendline
sb.regplot(x=X_test['Area'], y=y_test, scatter=False, color='red', label='Regression Trendline(Line of Best Fit)')

plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.show()



print("Price Prediction of own value:")
print(model.predict([[1050,2,2]]))

print("OK")



