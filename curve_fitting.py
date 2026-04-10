import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# x_data = np.array([0.        , 0.15789474, 0.31578947, 0.47368421, 0.63157895,
#        0.78947368, 0.94736842, 1.10526316, 1.26315789, 1.42105263,
#        1.57894737, 1.73684211, 1.89473684, 2.05263158, 2.21052632,
#        2.36842105, 2.52631579, 2.68421053, 2.84210526, 3.        ])
# y_data = np.array([  2.95258285,   2.49719803,  -2.1984975 ,  -4.88744346,
#         -7.41326345,  -8.44574157, -10.01878504, -13.83743553,
#        -12.91548145, -15.41149046, -14.93516299, -13.42514157,
#        -14.12110495, -17.6412464 , -16.1275509 , -16.11533771,
#        -15.66076021, -13.48938865, -11.33918701, -11.70467566])

df = pd.read_csv('data.csv')

x_data = df['x'].values
y_data = df['y'].values
plt.scatter(x_data, y_data)

# BETA = a, b, c = model parameters to be estimated
# f(x, BETA) = f(x, a, b, c)
def string_to_func(func):
    return lambda x, a, b, c: eval(func)

user_func = input("Enter the function (use 'a', 'b', 'c' as parameters and 'x' as variable): ")
model_f = string_to_func(user_func)
#popt = optimal parameters, pcov = covariance of the parameters
popt, pcov = curve_fit(model_f, x_data, y_data, p0=[3, 2, -16])
a_opt, b_opt, c_opt = popt

# print(popt)
print(popt)

#data for model training
x_model = np.linspace(min(x_data), max(x_data), 100)
y_model = model_f(x_model, a_opt, b_opt, c_opt)
plt.plot(x_model, y_model, color='red', label='Fitted Curve')
plt.legend()
plt.show()

#More the covariance, less the effect it has on the model