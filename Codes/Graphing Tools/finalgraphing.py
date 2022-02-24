import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from uncertainties import ufloat
from uncertainties.umath import *
from itertools import cycle

cycol = cycle("bgrcmk")
marker = cycle((",", "+", ".", "o", "*"))
delta = 15.64
I = (
    5 * 0.3 * 0.3 * 0.3 / 12
)  # The second moment of inertia of the material, respect to theZ axis
so = 30  # The initial lenght of the material
soc = so * so * so  # cube it


class showcase:
    def __init__(self):
        self.weights = []
        self.heights = []
        self.Ypred = []
        self.X = []
        self.pendiente = 0
        self.R = 0
        self.int = 0
        self.BSE = 0

    def read(self, path, step):
        df = pd.read_csv(path)
        y = df["y"]
        height = -y[::step] * 100
        si = len(height)
        weight = np.linspace(0, si, si) * delta + 61.06
        weight[0] = 0
        weight = weight * 980  # gravity
        self.weights = weight
        self.heights = height

    def init_tab(probs, cols):
        return pd.DataFrame(index=probs, columns=cols)

    def returns(self, path, step, stop, prob):
        self.read(path, step)
        self.linear_regression(stop)
        return self.X, self.Ypred, self.weights, self.heights

    def tab(self, stop, prob, data):
        self.linear_regression(stop)
        data.at[prob, "pendiente"] = float("{:.6f}".format(float(self.pendiente)))
        data.at[prob, "intercepto"] = float("{:.6f}".format(float(self.int)))
        data.at[prob, "R"] = float("{:.3f}".format(float(self.R)))
        data.at[prob, "BSE"] = float("{:.8f}".format(float(self.BSE)))

    def Youngs(self, prob, data):
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
        Ef0 = ufloat(s[prob], l[prob])
        Ef0 = sof / (3 * If * Ef0 * 10)
        data["E"][prob] = Ef0

    def linear_regression(self, stop):
        X = np.array(self.weights).reshape(-1, 1)
        X = X[:stop]  # values converts it into a numpy array
        Y = np.array(self.heights).reshape(
            -1, 1
        )  # -1 means that calculate the dimension of rows, but have 1 column basicamente la transpuesta (1,2,3..)
        Y = Y[:stop]
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(X, Y)  # perform linear regression
        Y_pred = linear_regressor.predict(X)  # make predictions
        ols = sm.OLS(Y, X)
        ols_result = ols.fit()
        self.X = X
        self.BSE = ols_result.bse
        self.Ypred = Y_pred
        self.pendiente = linear_regressor.coef_
        self.int = linear_regressor.intercept_
        self.R = linear_regressor.score(X, Y)

    def plot_youngs(data, probs):
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
        ax1.plot(probs[:5], data["E"][:5], marker="P", ls="dashed", ms=4, label="H=1mm")
        ax1.plot(
            probs[5:],
            data["E"][5:],
            marker="P",
            ls="dashed",
            ms=4,
            color="black",
            label="H=2mm",
        )
        ax1.set_xlabel(r"Probability", fontsize=15)
        ax1.set_ylabel(r"$E\quad[N/m^2] $", fontsize=15)
        ax1.legend()
