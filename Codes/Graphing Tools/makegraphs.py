# -*- coding: utf-8 -*-
"""
Here will graph things
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from uncertainties import ufloat
from uncertainties.umath import *

# -------------Import end, begin script-----------------------------------------
index = ["0,0", "0,05", "0,1", "0,15", "0,2"]
columns = ["pendiente", "intercepto", "R", "BSE"]
data = pd.DataFrame(index=index, columns=columns)
# ---------------------p=0,A=3mm----------------------------------------
df = pd.read_csv(
    r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel\p=0A=3mm.csv"
)
y = df["y"]
height = -y[::3] * 100
si = len(height)
weight = np.linspace(0, si, si) * 15.64 + 76.7 - 15.64
weight[0] = 0
weight = weight * 980  # gravity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 9))
ax1.errorbar(weight, height, marker="s", ls=" ", ms=6, label="P=0.0", color="black")
ax2.errorbar(weight, height, marker="s", ls=" ", ms=6, label="P=0.0", color="black")
# --------------------p=0,A=4mm,H=2mm LR--------------------------------------
X = np.array(weight).reshape(-1, 1)
X = X[:21]  # values converts it into a numpy array
Y = np.array(height).reshape(
    -1, 1
)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..)
Y = Y[:21]
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
ax1.plot(X, Y_pred, color="black", ls="dashed")  # esta linea roja si es una regresion
pendiente = linear_regressor.coef_
inter = linear_regressor.intercept_
R = linear_regressor.score(X, Y)
data.at["0,0", "pendiente"] = float("{:.6f}".format(float(pendiente)))
data.at["0,0", "intercepto"] = float("{:.6f}".format(float(inter)))
data.at["0,0", "R"] = float("{:.3f}".format(float(R)))
ols = sm.OLS(Y, X)
ols_result = ols.fit()
data.at["0,0", "BSE"] = float("{:.8f}".format(float(ols_result.bse)))
# ---------------------p=0,05A=3mmH=1mm----------------------------------------
df = pd.read_csv(
    r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel\p=0,05A=3mmH=1mm.csv"
)
y = df["y"]
height = -y[::4] * 100
height = height[:39]
si = len(height)
weight = np.linspace(0, si, si) * 15.64 + 76.7 - 15.64
weight[0] = 0
weight = weight * 980  # gravity
weight = weight[:39]
ax1.errorbar(weight, height, marker=".", ls=" ", ms=9, label="P=0.05", color="seagreen")
ax2.errorbar(weight, height, marker=".", ls=" ", ms=9, label="P=0.05", color="seagreen")
# --------------------p=0,05A=4mm,H=2mm LR--------------------------------------
X = np.array(weight).reshape(-1, 1)
X = X[:10]  # values converts it into a numpy array
Y = np.array(height).reshape(
    -1, 1
)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..)
Y = Y[:10]
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
ax1.plot(X, Y_pred, color="black", ls="dashed")  # esta linea roja si es una regresion
pendiente = linear_regressor.coef_
inter = linear_regressor.intercept_
R = linear_regressor.score(X, Y)
data.at["0,05", "pendiente"] = float("{:.6f}".format(float(pendiente)))
data.at["0,05", "intercepto"] = float("{:.6f}".format(float(inter)))
data.at["0,05", "R"] = float("{:.3f}".format(float(R)))
ols = sm.OLS(Y, X)
ols_result = ols.fit()
data.at["0,05", "BSE"] = float("{:.8f}".format(float(ols_result.bse)))
# ---------------------p=0,1,A=3mm,H=1mm----------------------------------------
df = pd.read_csv(
    r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel\p=0,1A=3mmH=1mm.csv"
)
y = df["y"]
height = -y[::4] * 100
si = len(height)
weight = np.linspace(0, si, si) * 15.64 + 76.7 - 15.64
weight[0] = 0
weight[1] = 0
weight = weight * 980  # gravity
ax1.errorbar(
    weight, height, marker="^", ls=" ", ms=9, label="P=0.1", color="darkmagenta"
)
ax2.errorbar(
    weight, height, marker="^", ls=" ", ms=9, label="P=0.1", color="darkmagenta"
)
# --------------------p=0,1A=3mmH=1mm LR--------------------------------------
X = np.array(weight).reshape(-1, 1)
X = X[:12]  # values converts it into a numpy array
Y = np.array(height).reshape(
    -1, 1
)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..)
Y = Y[:12]
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
ax1.plot(X, Y_pred, color="black", ls="dashed")  # esta linea roja si es una regresion
pendiente = linear_regressor.coef_
inter = linear_regressor.intercept_
R = linear_regressor.score(X, Y)
data.at["0,1", "pendiente"] = float("{:.6f}".format(float(pendiente)))
data.at["0,1", "intercepto"] = float("{:.6f}".format(float(inter)))
data.at["0,1", "R"] = float("{:.3f}".format(float(R)))
ols = sm.OLS(Y, X)
ols_result = ols.fit()
data.at["0,1", "BSE"] = float("{:.8f}".format(float(ols_result.bse)))
# ---------------------p=0,15,A=3mm,H=1mm----------------------------------------
df = pd.read_csv(
    r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel\p=0,15A=3mmH=1mm.csv"
)
y = df["y"]
height = -y[::4] * 100
print(height[0])
si = len(height)
weight = np.linspace(0, si, si) * 15.64 + 76.7 - 15.64
weight[0] = 0
weight = weight * 980  # gravity
ax1.errorbar(
    weight, height, marker="D", ls=" ", ms=9, label="P=0.15", color="royalblue"
)
ax2.errorbar(
    weight, height, marker="D", ls=" ", ms=9, label="P=0.15", color="royalblue"
)
# --------------------p=0,15A=3mmH=1mm LR--------------------------------------
X = np.array(weight).reshape(-1, 1)
X = X[:8]  # values converts it into a numpy array
Y = np.array(height).reshape(
    -1, 1
)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..)
Y = Y[:8]
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
ax1.plot(X, Y_pred, color="black", ls="dashed")  # esta linea roja si es una regresion
pendiente = linear_regressor.coef_
inter = linear_regressor.intercept_
R = linear_regressor.score(X, Y)
data.at["0,15", "pendiente"] = float("{:.6f}".format(float(pendiente)))
data.at["0,15", "intercepto"] = float("{:.6f}".format(float(inter)))
data.at["0,15", "R"] = float("{:.3f}".format(float(R)))
ols = sm.OLS(Y, X)
ols_result = ols.fit()
data.at["0,15", "BSE"] = float("{:.8f}".format(float(ols_result.bse)))
# ---------------------p=0,2,A=3mm,H=1mm----------------------------------------
df = pd.read_csv(
    r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel\p=0,2A=3mmH=1mm.csv"
)
y = df["y"]
height = -y[::4] * 100
si = len(height)
weight = np.linspace(0, si, si) * 15.64 + 76.7 - 15.64
weight[0] = 0
weight = weight * 980  # gravity
ax1.errorbar(weight, height, marker="P", ls=" ", ms=9, label="P=0.2", color="orange")
ax2.errorbar(weight, height, marker="P", ls=" ", ms=9, label="P=0.2", color="orange")
# --------------------p=0,2A=3mmH=1mm LR--------------------------------------
X = np.array(weight).reshape(-1, 1)
X = X[:6]  # values converts it into a numpy array
Y = np.array(height).reshape(
    -1, 1
)  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..)
Y = Y[:6]
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
ax1.plot(X, Y_pred, color="black", ls="dashed")  # esta linea roja si es una regresion
pendiente = linear_regressor.coef_
inter = linear_regressor.intercept_
R = linear_regressor.score(X, Y)
data.at["0,2", "pendiente"] = float("{:.6f}".format(float(pendiente)))
data.at["0,2", "intercepto"] = float("{:.6f}".format(float(inter)))
data.at["0,2", "R"] = float("{:.3f}".format(float(R)))
ols = sm.OLS(Y, X)
ols_result = ols.fit()
data.at["0,2", "BSE"] = float("{:.8f}".format(float(ols_result.bse)))
# -----------------------Titles and axes and stuff------------------------------
ax1.set_xlabel(r"peso $[dyn] \pm 0.006$", fontsize=20)
ax1.set_ylabel(r"Desplazamiento en Y $[cm] $", fontsize=20)
ax2.set_xlabel(r"peso $[dyn] \pm 0.006$", fontsize=20)
ax2.set_ylabel(r"Desplazamiento en Y $[cm] $", fontsize=20)
ax1.tick_params(axis="both", labelsize=20)
ax2.tick_params(axis="both", labelsize=20)
ax1.legend()
# --------------------Calculating Young's Moduli---------------------------------
I = (
    5 * 0.3 * 0.3 * 0.3 / 12
)  # The second moment of inertia of the material, respect to theZ axis
so = 30  # The initial lenght of the material
soc = so * so * so  # cube it
E = (
    np.ones(5) * soc / (3 * I * data["pendiente"] * 10)
)  # el 10 es para que las unidades pasen de g/cm s^2 a kg/m s^2
data["E"] = E  # append it to the calculated E
# ------------------Plotting probability versus E---------------------------
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
ax1.plot(index, data["E"], marker="P", ls="dashed", linewidth=0.5)
ax1.set_xlabel(r"Probability", fontsize=15)
ax1.set_ylabel(r"$E\quad[N/m^2] $", fontsize=15)
# -------------Here i want to also calculate the error propagation--------------------
af = ufloat(
    5, 0
)  # The measures were cut to precision with the machine, some of the material is consumed, but its sistematic in all the plates
bf = ufloat(
    0.3, 0
)  # this measure is given by the manufacturer, given that they are all cut form the same material, no error is given
If = af * bf * bf * bf / 12
sof = ufloat(30, 0.1)
sof = sof * sof * sof
s = data["pendiente"]
l = data["BSE"]
Ef0 = ufloat(s["0,0"], l["0,0"])
Ef0 = sof / (3 * If * Ef0 * 10)
data["E"]["0,0"] = Ef0
Ef0 = ufloat(s["0,05"], l["0,05"])
Ef0 = sof / (3 * If * Ef0 * 10)
data["E"]["0,05"] = Ef0
Ef0 = ufloat(s["0,1"], l["0,1"])
Ef0 = sof / (3 * If * Ef0 * 10)
data["E"]["0,1"] = Ef0
Ef0 = ufloat(s["0,15"], l["0,15"])
Ef0 = sof / (3 * If * Ef0 * 10)
data["E"]["0,15"] = Ef0
Ef0 = ufloat(s["0,2"], l["0,2"])
Ef0 = sof / (3 * If * Ef0 * 10)
data["E"]["0,2"] = Ef0
display(data)
