# Import libraries
import pandas as pd               
import seaborn as sns             
import numpy as np                
import matplotlib.pyplot as plt   
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("D://FuelConsumption.csv")

# View the dataset and understand it
print(df.head())                 
print("Data Info:", df.info())  
print("Statistics:", df.describe())  
print("Null values:", df.isnull().sum())  

# Plot a scatter plot to visualize relation between Engine Size and CO2 Emissions
plt.scatter(df["ENGINESIZE"], df["CO2EMISSIONS"], color="red")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSION")
plt.show()

# Select input (X) and output (y) features
# Drop unwanted columns and keep only 'ENGINESIZE' as feature
X = df[["ENGINESIZE"]]
y = df[["CO2EMISSIONS"]]

print("X Features:\n", X)
print("y Target:\n", y)

# Split the data into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
regr = LinearRegression()
regr.fit(x_train, y_train)   

# Predict using the test data
ypred = regr.predict(x_test)

# Print model parameters
print("Predicted CO2 Emissions:", ypred)
print("Model Coefficient (slope):", regr.coef_)     
print("Model Intercept:", regr.intercept_)          

# Visualize the regression line
plt.scatter(x_train, y_train, color="blue")         
plt.plot(x_test, ypred, color="black")               
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSION")
plt.show()

# Predict custom value from user input
my_engine_size = eval(input("Enter your engine size: "))

# Custom prediction function
def predict_polution(enginesize, coef, intercept):
    return enginesize * coef + intercept

estimate_pollution = predict_polution(my_engine_size, regr.coef_[0][0], regr.intercept_[0])
print("Estimated CO2 Emission:", estimate_pollution)