import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
df = pd.read_csv("Dataset.csv")

df.shape
df.describe()
df.isnull().any()


X = df[['Problem Solved', 'Followers','Contribution','Registered Years']].values
y = df['Ratings'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)


#Linear Regression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print("Linear Regression Score: ",regressor.score(X_train, y_train))
print("Co-Efficient ",regressor.coef_)

plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred,s=15)
plt.xlabel('Actual',fontsize=14)
plt.ylabel('Predict',fontsize=14)
plt.title('Linear Regression Actual vs Predict')
plt.show()

#df=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
#print(df)

df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df2.plot(kind='bar',figsize=(10,8),xlabel='Test Data',ylabel='Ratings',title='Linear Regression')
plt.show()


print('Mean Absolute Error:', 	metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Absolute Percentage Error:', 	metrics.mean_absolute_percentage_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))


#DecisionTreeRegressor

regressor2 = DecisionTreeRegressor()
regressor2.fit(X_train, y_train)
y_pred2 = regressor2.predict(X_test)

print("Decision Tree Regression Score: ",regressor2.score(X_train, y_train))


plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred2,s=15,c='r')
plt.xlabel('Actual',fontsize=14)
plt.ylabel('Predict',fontsize=14)
plt.title('Decision Tree Regression Actual vs Predict')
plt.show()

#df=pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
#print(df)

df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
df2.plot(kind='bar',figsize=(10,8),xlabel='Test Data',ylabel='Ratings',title='Decision Tree Regression')
plt.show()


print('Mean Absolute Error:', 	metrics.mean_absolute_error(y_test, y_pred2)) 
print('Mean Absolute Percentage Error:', 	metrics.mean_absolute_percentage_error(y_test, y_pred2))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred2))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
print('R2 Score:', metrics.r2_score(y_test, y_pred2))


plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred2,s=15)
plt.scatter(y_test,y_pred,s=15,c='r')
plt.xlabel('Actual',fontsize=14)
plt.ylabel('Predict',fontsize=14)
plt.title('Comparision between Linear Regression and Decision Tree Regression')
plt.show()


