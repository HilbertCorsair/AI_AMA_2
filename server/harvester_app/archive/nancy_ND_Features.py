import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

feature_table = pd.DataFrame({
    "AGIX" : [1, 0, 1, 1, 0, 0, 0,1],
    "green_tail_ride": [1, 0, 1, 0, 1, 1, 0, 0],
    "optimistic": [1, 0, 0, 1, 1, 0, 1, 0],
    "results": [635, 115, 320, 98, 123, 107, 93, 50]
})

print(feature_table)
for c in feature_table.columns[0:3]:
    score = sum(feature_table[c] * feature_table["results"])/len(feature_table["results"])
    print(c, score)


#Most powerfull feature : green tail ride
# other features : noise canceling, MINA and AGIX priority 


# create a numpy array with your data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([[2, 3], [4, 6], [6, 9], [8, 12], [10, 15]])

# create a linear regression object
reg = LinearRegression()

# fit the model
reg.fit(X, y)

# display results
print('Coefficients: ', reg.coef_)
print('Intercept: ', reg.intercept_)