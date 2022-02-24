# -*- coding: utf-8 -*-
"""
Here will graph things
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------Import end, begin script-----------------------------------------
index = ["0,05", "0,15"]
columns = ["pendiente", "intercepto", "R"]
data = pd.DataFrame(index=index, columns=columns)
# ---------------------p=0,A=3mm----------------------------------------
df = pd.read_csv(
    r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel\p=0,05A=4mmH=2mm.csv"
)
y = df["y"]
height = -y[::3] * 100
si = len(height)
weight = np.linspace(0, si, si) * 15.64 + 66.4
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
ax1.errorbar(weight, height, marker=".", ls=" ", ms=6, label="P=0.05")
# --------------------p=0,15,A=4mm,H=2mm LR--------------------------------------
X = np.array(weight).reshape(-1, 1)
# values converts it into a numpy array
Y = np.array(height).reshape(
    -1, 1
)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
ax1.plot(X, Y_pred)  # esta linea roja si es una regresion
pendiente = linear_regressor.coef_
inter = linear_regressor.intercept_
R = linear_regressor.score(X, Y)
data.at["0,05", "pendiente"] = float("{:.3f}".format(float(pendiente)))
data.at["0,05", "intercepto"] = float("{:.3f}".format(float(inter)))
data.at["0,05", "R"] = float("{:.3f}".format(float(R)))

# ---------------------p=0,05,A=3mm,H=1mm----------------------------------------
df = pd.read_csv(
    r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel\p=0,15A=4mmH=2mm.csv"
)
y = df["y"]
height = -y[::4] * 100
si = len(height)
weight = np.linspace(0, si, si) * 15.64 + 66.4
ax1.errorbar(weight, height, marker=".", ls=" ", ms=6, label="P=0.15")
# --------------------p=0,05A=3mmH=1mm LR--------------------------------------
X = np.array(weight).reshape(-1, 1)
# values converts it into a numpy array
Y = np.array(height).reshape(
    -1, 1
)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..)
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
ax1.plot(X, Y_pred, color="red")  # esta linea roja si es una regresion
pendiente = linear_regressor.coef_
inter = linear_regressor.intercept_
R = linear_regressor.score(X, Y)
data.at["0,15", "pendiente"] = float("{:.3f}".format(float(pendiente)))
data.at["0,15", "intercepto"] = float("{:.3f}".format(float(inter)))
data.at["0,15", "R"] = float("{:.3f}".format(float(R)))


# -----------------------Titles and axes and stuff------------------------------
ax1.set_xlabel(r"peso $[g] \pm 0.1$", fontsize=15)
ax1.set_ylabel(r"Desplazamiento en Y $[cm] $", fontsize=15)
ax1.legend()
display(data)
