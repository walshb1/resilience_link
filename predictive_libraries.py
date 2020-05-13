from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np

import numpy.polynomial.polynomial as poly
np.random.seed(123)

def decay_func(x, a, k, b):
    return a * np.exp(-k*x) + b

def df_to_decay_fit(df,colX,colY,x2=None):

    # sample data
    X = df[colX].squeeze().T
    Y = df[colY].squeeze().T
    # x = np.array([399.75, 989.25, 1578.75, 2168.25, 2757.75, 3347.25, 3936.75, 4526.25, 5115.75, 5705.25])
    # y = np.array([109,62,39,13,10,4,2,0,1,2])

    # curve fit
    p0 = (1.,1.e-5,1.) # starting search koefs
    opt, pcov = curve_fit(decay_func,X,Y, p0)
    a, k, b = opt
    # test result
    if x2 is None: 
        x2 = np.linspace(0, 50, 100)
    y2 = decay_func(x2, a, k, b)

    return x2,y2

def df_to_linear_fit(df,colX,colY,wgt=None,x2=None):

    X = df[colX].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df[colY].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y,sample_weight=wgt)  # perform linear regression

    if x2 is not None: Y_pred = linear_regressor.predict(x2)
    else: Y_pred = linear_regressor.predict(X)  # make predictions
    
    coef = float(linear_regressor.coef_)

    return Y_pred,coef

def df_to_exponential_fit(df,colX,colY,wgt=None):

    X = df[colX].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df[colY].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

    # Y = np.log(df[colY].values.reshape(-1, 1)) # -1 means that calculate the dimension of rows, but have 1 column
    transformer = FunctionTransformer(np.log, validate=True)
    y_trans = transformer.fit_transform(Y)     

    linear_regressor = LinearRegression()  # create object for the class
    results = linear_regressor.fit(X, y_trans,sample_weight=wgt)

    linear_regressor.fit(X, y_trans,sample_weight=wgt)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    coef = float(linear_regressor.coef_)

    return Y_pred,coef

def df_to_polynomial_fit(df,colX,colY,power,wgt=None,x_new=None):

	# X = df[colX].values.reshape(-1, 1)  # values converts it into a numpy array
	# Y = df[colY].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
	X = df[colX].squeeze().T
	Y = df[colY].squeeze().T

	coefs = poly.polyfit(X,Y,power)

	if x_new is None: x_new = np.linspace(0, 40, num=100)
	ffit = poly.polyval(x_new, coefs)

	return x_new,ffit

