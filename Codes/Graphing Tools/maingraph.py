# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:34:10 2021

@author: Carolina
"""

from finalgraphing import *
import os
from itertools import cycle

I = (
    5 * 0.3 * 0.3 * 0.3 / 12
)  # The second moment of inertia of the material, respect to theZ axis
so = 30  # The initial lenght of the material
soc = so * so * so  # cube it
cycol = cycle(("darkgoldenrod", "navy", "purple", "grey", "teal", "black", "darkred"))
marker = cycle(("s", "+", "d", "o", "*"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 9))
path = r"C:\Users\Carolina\OneDrive\Escritorio\Int inv Exp\videos y fotos medidas\excel"
probs = [
    "0,0",
    "0,05",
    "0,1",
    "0,15",
    "0,2",
    "0,05_2mm",
    "0,1_2mm",
    "0,15_2mm",
    "0,2_2mm",
]
columns = ["pendiente", "intercepto", "R", "BSE"]
df = showcase.init_tab(probs, columns)

# ---------------------P=0.0--------------------------------------------------------------------
p0 = showcase()
stop0 = 30
X, Y, weight, height = p0.returns(os.path.join(path, "P=0_A=3mm.csv"), 2, stop0, "0,0")
ax1.errorbar(
    weight,
    height,
    marker=next(marker),
    ls=" ",
    ms=6,
    label="P=0.0",
    c=next(cycol),
    markerfacecolor="None",
    markeredgewidth=2,
)
ax1.plot(X, Y, color="black", ls="dashed")
ax2.errorbar(
    weight, height, marker=next(marker), ls=" ", ms=6, label="P=0.0", c=next(cycol)
)
ax2.plot(X, Y, color="black", ls="dashed")

# --------------------P=0.05--------------------------------------------------------------------
p1 = showcase()
stop1 = 20
X, Y, weight, height = p1.returns(
    os.path.join(path, "P=0,05_A=3mm_H=1mm_a.csv"), 4, stop1, "0,05"
)
ax1.errorbar(
    weight,
    height,
    marker=next(marker),
    ls=" ",
    ms=6,
    label="P=0.05",
    c=next(cycol),
    markerfacecolor="None",
    markeredgewidth=2,
)
ax1.plot(X, Y, color="black", ls="dashed")

# --------------------P=0.1-----------------------------------------------------------------------
p2 = showcase()
stop2 = 18
X, Y, weight, height = p2.returns(
    os.path.join(path, "P=0,1_A=3mm_H=1mm.csv"), 4, stop2, "0,1"
)
ax1.errorbar(
    weight,
    height,
    marker=next(marker),
    ls=" ",
    ms=6,
    label="P=0.1",
    c=next(cycol),
    markerfacecolor="None",
    markeredgewidth=2,
)
ax1.plot(X, Y, color="black", ls="dashed")

# --------------------P=0.15-----------------------------------------------------------------------
p3 = showcase()
stop3 = 15
X, Y, weight, height = p3.returns(
    os.path.join(path, "P=0,15_A=3mm_H=1mm.csv"), 4, stop3, "0,15"
)
ax1.errorbar(
    weight, height, marker=next(marker), ls=" ", ms=6, label="P=0.15", c=next(cycol)
)
ax1.plot(X, Y, color="black", ls="dashed")

# --------------------P=0.2-----------------------------------------------------------------------
p4 = showcase()
stop4 = 10
X, Y, weight, height = p4.returns(
    os.path.join(path, "P=0,2_A=3mm_H=1mm.csv"), 4, stop4, "0,2"
)
ax1.errorbar(
    weight, height, marker=next(marker), ls=" ", ms=6, label="P=0.2", c=next(cycol)
)
ax1.plot(X, Y, color="black", ls="dashed")

# --------------------P=0.05_2mm--------------------------------------------------------------------
p1_2 = showcase()
stop1_2 = 20
X, Y, weight, height = p1_2.returns(
    os.path.join(path, "P=0,05_A=3mm_H=2mm.csv"), 4, stop1_2, "0,05"
)
ax2.errorbar(
    weight, height, marker=next(marker), ls=" ", ms=6, label="P=0.05", c=next(cycol)
)
ax2.plot(X, Y, color="black", ls="dashed")
# --------------------P=0.1_2mm-----------------------------------------------------------------------
p2_2 = showcase()
stop2_2 = 20
X, Y, weight, height = p2_2.returns(
    os.path.join(path, "P=0,1_A=3mm_H=2mm.csv"), 4, stop2_2, "0,1"
)
ax2.errorbar(
    weight,
    height,
    marker=next(marker),
    ls=" ",
    ms=6,
    label="P=0.1",
    c=next(cycol),
    markerfacecolor="None",
    markeredgewidth=2,
)
ax2.plot(X, Y, color="black", ls="dashed")

# --------------------P=0.15_2mm-----------------------------------------------------------------------
p3_2 = showcase()
stop3_2 = 10
X, Y, weight, height = p3_2.returns(
    os.path.join(path, "P=0,15_A=3mm_H=2mm_b.csv"), 4, stop3_2, "0,15"
)
ax2.errorbar(
    weight,
    height,
    marker=next(marker),
    ls=" ",
    ms=6,
    label="P=0.15",
    c=next(cycol),
    markerfacecolor="None",
    markeredgewidth=2,
)
ax2.plot(X, Y, color="black", ls="dashed")
# --------------------P=0.2_2mm-----------------------------------------------------------------------
p4_2 = showcase()
stop4_2 = 5
X, Y, weight, height = p4_2.returns(
    os.path.join(path, "P=0,2_A=3mm_H=2mm.csv"), 4, stop4_2, "0,2"
)
ax2.errorbar(
    weight, height, marker=next(marker), ls=" ", ms=6, label="P=0.2", c=next(cycol)
)
ax2.plot(X, Y, color="black", ls="dashed")
# ---------------------------legends,titles
# -----------------------------------------------------------
ax1.set_xlabel(r"peso $[dyn] \pm 0.006$", fontsize=20)
ax1.set_ylabel(r"Desplazamientso en Y $[cm] $", fontsize=20)
ax2.set_xlabel(r"peso $[dyn] \pm 0.006$", fontsize=20)
ax2.set_ylabel(r"Desplazamiento en Y $[cm] $", fontsize=20)
ax1.tick_params(axis="both", labelsize=20)
ax2.tick_params(axis="both", labelsize=20)
ax1.set_title(r"Ancho= 3mm, Hueco= 1mm", size=20)
ax2.set_title(r"Ancho= 3mm, Hueco= 2mm", size=20)
ax1.legend()
ax2.legend()

p0.tab(stop0, "0,0", df)
p1.tab(stop1, "0,05", df)
p2.tab(stop2, "0,1", df)
p3.tab(stop3, "0,15", df)
p4.tab(stop4, "0,2", df)
p1_2.tab(stop1, "0,05_2mm", df)
p2_2.tab(stop2, "0,1_2mm", df)
p3_2.tab(stop3, "0,15_2mm", df)
p4_2.tab(stop4, "0,2_2mm", df)
E = (
    np.ones(9) * soc / (3 * I * df["pendiente"] * 10)
)  # el 10 es para que las unidades pasen de g/cm s^2 a kg/m s^2
df["E"] = E  # append it to the calculated E
showcase.plot_youngs(df, [0, 0.05, 0.1, 0.15, 0.2, 0.05, 0.1, 0.15, 0.2])
display(df)
p0.Youngs("0,0", df)
p1.Youngs("0,05", df)
p2.Youngs("0,1", df)
p3.Youngs("0,15", df)
p4.Youngs("0,2", df)
p1_2.Youngs("0,05_2mm", df)
p2_2.Youngs("0,1_2mm", df)
p3_2.Youngs("0,15_2mm", df)
p4_2.Youngs("0,2_2mm", df)

display(df)
