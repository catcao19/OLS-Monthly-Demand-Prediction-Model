import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

monthly_demand = [100, 112, 107, 103, 91, 85, 84, 85, 79, 81, 134, 86, 99, 89, 111, 114, 118, 163, 193, 143, 144, 202, 158, 160, 144]
advance_demand = [71, 30, 75, 64, 41, 51, 42, 51, 57, 49, 134, 52, 99, 56, 81, 79, 73, 163, 193, 99, 91, 202, 105, 101, 96]

def removena(df):
    return df.dropna(how='all').dropna(axis='columns', how='all')

def solve(df):
    return pd.DataFrame(np.linalg.pinv(df.values), df.columns, df.index)

Y = monthly_demand[1:]
ones_m = [1]*len(Y)
# L = advance_demand[:-1]
L = advance_demand[1:]   ###################<<<<<<<<<<<<<<<<Second month X*Beta
X1 = monthly_demand[:-1]
# X2 = advance_demand[1:]
X2 = L

X = pd.DataFrame([ones_m, X1, X2]).T
y = pd.DataFrame(Y)
print('X')
print(X)

################## Labmda Array????
Lambda = (2*solve((X.dot(solve(X.T.dot(X)))).dot(X.T))).dot(L)- 2*y.T

# print(y.T)


# Lambda = [0 if x<0 else x for x in Lambda.values[0]]
print('Lambda')
print(Lambda)


Lambda = 1
Lambda=max(0,Lambda)

Beta = removena((solve(X.T.dot(X))).dot(X.T.dot(y) + 1/2*Lambda*X.T))
# Beta = solve(X.T.dot(X)).dot((X.T.dot(y) + 1/2*Lambda.T.dot(X.T)))




print('Beta')
print(Beta)

Yhat = X *Beta.T                #############????
print('Yhat')
print(Yhat)

fig=plt.figure()
ax=plt.axes()
plt.title('X and y')
plt.ylabel('y')
plt.xlabel('X')
plt.plot(X,y,color='blue')
plt.show()