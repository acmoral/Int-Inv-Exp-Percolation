# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:26:16 2021

@author: Carolina
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv(
    r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\Results\30x5FC\p_inf\pinf_30x5FC.csv",
    na_filter=False,
)
display(data)
p_ideal_array = np.array(data["p"])
p_ideal_array = p_ideal_array[-8:]
p_inf_array = np.array(data["p_inf"])
p_inf_array = p_inf_array[-8:]
p_real_array = np.array(data["p_real"])
p_real_array = p_real_array[-8:]
p_inf_err = data["p_inf_dev"]
p_inf_err = p_inf_err[-8:]
p_real_err = data["preal_dev"]
p_real_err = p_real_err[-8:]
pc_real = 0.39
pc_ideal = 0.85
pc_real_dev = 0.03
pc_ideal_dev = 0.15
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(16, 8)
)  # aca puedes cambiar el tamaño de la figura 12 es el ancho y 8 la altura
ax1.errorbar(
    p_ideal_array,
    p_inf_array,
    xerr=0,
    yerr=p_inf_err,
    linestyle="dashed",
    marker="o",
    markersize=9,
    color="black",
    mfc="red",
)
ax2.errorbar(
    p_real_array,
    p_inf_array,
    xerr=p_real_err,
    yerr=p_inf_err,
    linestyle="dashed",
    marker="o",
    markersize=9,
    color="black",
    mfc="red",
)
ax1.set_xlabel("p", fontsize=30)
ax2.set_xlabel("$p_{real}$", fontsize=30)
ax1.set_ylabel(r"$p_{\infty}(p)$", fontsize=30)
ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
plt.show()
plt.close()
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(16, 8)
)  # aca puedes cambiar el tamaño de la figura 12 es el ancho y 8 la altura
logs_real = np.log((p_real_array - pc_real))
logs_ideal = np.log(np.abs(p_ideal_array - pc_ideal))
pinf_log = np.log(p_inf_array)
ax2.plot(
    logs_real,
    pinf_log,
    marker="D",
    ls="",
    ms=7,
    markerfacecolor="None",
    markeredgewidth=2,
    color="black",
)
ax1.plot(
    logs_ideal,
    pinf_log,
    marker="D",
    ls="",
    ms=7,
    markerfacecolor="None",
    markeredgewidth=2,
    color="black",
)
linear_regressor = LinearRegression()  # create object for the class
X = logs_real.reshape(-1, 1)
Y = pinf_log.reshape(-1, 1)
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)
ax2.plot(X, Y_pred, color="black")
ax1.set_xlabel("log(p-pc)", fontsize=30)
ax2.set_xlabel("$log(p_{real}-p_c)$", fontsize=30)
ax1.set_ylabel(r"$p_{\infty}(p)$", fontsize=30)
ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)

print(linear_regressor.coef_)
print(linear_regressor.intercept_)
print(linear_regressor.score(X, Y))
