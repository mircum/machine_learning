import numpy as np
import matplotlib.pyplot as plt

# load data
filename = "data_1d.csv"
raw_data = open(filename, 'rt')
data = np.loadtxt(raw_data, delimiter=",")

# get the X and Y array
X, Y = np.split(data, [-1], axis=1)


X = np.asarray(X).reshape(-1)
Y = np.asarray(Y).reshape(-1)

# calculate linear regressions
denom = X.dot(X) - X.mean()*X.sum()

a = (X.dot(Y) - Y.mean()*X.sum()) / denom
b = (Y.mean()*X.dot(X) - X.mean()*X.dot(Y)) / denom

yLin = a*X + b

# plot X and Y
plt.scatter(X, Y)
plt.plot(X, yLin)
plt.show()
