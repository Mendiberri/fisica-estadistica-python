# -*- coding: utf-8 -*-
"""
Created on Fri May 26 19:39:00 2023

@author: imend
"""

import numpy as np
import matplotlib.pyplot as plt

#Primero busco la correlación entre las siguientes tiras de datos

X = np.linspace(2.00,3.00,11)
Y = np.array([2.78,3.29,3.29,3.33,3.23,3.69,3.46,3.87,3.62,3.40,3.99])

#calculo "a mano" de la correlación

#saco los promedios y el promedio del producto, con esto puedo obtener la covarianza
avg_x = np.mean(X)
avg_y = np.mean(Y)
x_y = []
for i in range(len(X)):
    x_y.append(X[i]*Y[i])
avg_xy = np.mean(x_y)

#calculo las varianzas
x2 = []
for i in range(len(X)):
    x2.append(X[i]**2)
y2 = []
for i in range(len(X)):
    y2.append(Y[i]**2)

var_x = np.mean(x2)-(avg_x**2)
var_y = np.mean(y2)-(avg_y**2)

#Saco la correlación como la covarianza divido el producto de la raiz de las varianzas
cor = (avg_xy - avg_x*avg_y)/np.sqrt(var_x*var_y)
#Me coincide con el cálculo hecho con la matriz de numpy en 12 cifras significativas