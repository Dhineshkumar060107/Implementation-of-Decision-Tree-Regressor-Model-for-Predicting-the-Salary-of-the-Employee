# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and load the dataset; separate input feature (Level) and output (Salary).

2. Create and train the Decision Tree Regressor model using the training data.

3. Use the trained model to predict salary for a given Level value.

4. Visualize or display the prediction results to analyze model performance. 

## Program:
```
NAME:Dhineshkumar.L
REGISTER NUMBER: 212224230066
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('Salary.csv')

print(dataset.head())

X = dataset[['Level']]  
y = dataset['Salary']            

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor(random_state=42)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

level = 6.5
predicted_salary = regressor.predict(pd.DataFrame([[Level]], columns=['Level']))
print(f"Predicted salary for {level} years of experience is: {predicted_salary[0]}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

X_grid = np.arange(min(X.values), max(X.values), 0.01) 
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red')                
plt.plot(X_grid, regressor.predict(X_grid), color='blue') 
plt.title('Decision Tree Regression')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

```

## Output:


<img width="1083" height="378" alt="image" src="https://github.com/user-attachments/assets/1e31a108-9dba-4b47-8be0-86a213c9b2d2" />


<img width="793" height="568" alt="Screenshot 2025-10-16 134156" src="https://github.com/user-attachments/assets/68319b88-0410-4763-ac33-18adfeaa7e32" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
