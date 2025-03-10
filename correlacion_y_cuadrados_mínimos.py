# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:54:27 2023

@author: Mendiberri
"""

import numpy as np

##Primero busco la correlación entre las siguientes tiras de datos

X = np.linspace(2.00,3.00,11)
Y = np.array([2.78,3.29,3.29,3.33,3.23,3.69,3.46,3.87,3.62,3.40,3.99])

#Numpy tiene una función asignada para esto
matriz_corr = np.corrcoef(X,Y) #la correlación X,Y está en la antidiagonal de esta matriz
correlacion = matriz_corr[1,0]

print("correlación :", correlacion)
#Lo saqué por su formula en cálculos auxiliares y me dio lo mismo (con 12 cifras significativas)

#%%
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

##Quiero ajustar los datos con una recta, tomando sigma = 0.3

#Defino la función con la que voy a ajustar
def lineal(x,a,b):
    y = a*x + b
    return(y)

##Uso la función curve_fit para obtener los parámetros del ajuste y la matriz de covarianza.

#La función misma me da la opción de asignar el sigma que quiero (sigma = 0.3)
opt, cov = curve_fit(lineal, X, Y, sigma=0.3*np.ones(11))
pendiente = opt[0]
ordenada = opt[1]

#Recta obtenida del ajuste
predic = lineal(X,pendiente,ordenada)

#Error
res = Y - predic
error = np.sqrt(np.mean(res**2))

print("Pendiente ajustada:",pendiente)
print("Ordenada al origen ajustada:",ordenada)
print("Error:",error)

##Luego grafico los puntos con su error y la recta obtenida entre X=0 y X=5, ambos en simultaneo

plt.figure()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Ajuste de recta")

#Grafico los datos con su error
sigma = 0.3*np.ones(11)
plt.errorbar(X, Y, yerr=sigma, fmt="o", label="Datos")

#Grafico la recta ajustada
rango_x = np.linspace(0,5,100)
rango_y = lineal(rango_x,pendiente,ordenada)
plt.plot(rango_x, rango_y, label="Recta ajustada")

##Saco el R cuadrado

ss_total = np.sum((Y - np.mean(Y))**2)
ss_res = np.sum(res**2)
R_sqr = 1 - (ss_res / ss_total)

print("R cuadrado:",R_sqr) #El R cuadrado resulta igual a la correlación al cuadrado

##Con los parámetros y la matriz de covarianza obtenidos del ajuste trato de poner banda de error

Xa = np.linspace(0,5,100)
Ya = lineal(Xa,pendiente,ordenada)

#Cálculo del error de la predicción utilizando la fórmula de propagación de errores
error_pred = np.sqrt(cov[0, 0] * Xa**2 + cov[1, 1] + 2 * Xa * cov[0, 1])

#Gráfico con la predicción y la banda de error
plt.plot(Xa, Ya, "ro")
plt.fill_between(rango_x, rango_y - error_pred, rango_y + error_pred, alpha=0.3, label="Banda de error")
plt.legend()

##Busco el Xa que minimiza el error

#Analíticamente hallé que Xmin = - V12/V11, para Vij elementos de la matriz de covarianza
Xa_min_error = -cov[1,0]/cov[0,0] #Me da 2.500000009
#Numericamente va a errar un poco, porque tengo solo 100 posiciones discretas
Xa_min_error2 = rango_x[np.where(error_pred == min(error_pred))[0][0]] #Dio 2.525252525252525

#%%
##Quiero graficar similar a lo anterior, pero la banda de error no tendrá en cuenta la correlacion
##O sea, al propagar solo voy a considerar las varianzas, como si la matriz cov fuera diagonal

#Este error errado sale como el anterior, sacando los términos con elementos no diagonales de cov
error_pred_sin_corr = np.sqrt(cov[0, 0] * Xa**2 + cov[1, 1] + 0)

plt.fill_between(rango_x, rango_y - error_pred_sin_corr, rango_y + error_pred_sin_corr, alpha=0.3, label="Banda error sin correlación")
plt.legend()

#%%
##Por último quiero sacar los valores Yi para los Xi a partir de una gaussiana
##Luego ajustar (X,Y) 1000 veces y poner la predicción para X=0.5 en un histograma
from scipy.stats import norm

repet = 1000
Ya_rep = []

for j in range(repet):
    #Obtengo los yi de con la distribución y parámetros que saqué
    np.random.seed()
    Yi = np.random.normal(loc=X*pendiente + ordenada, scale=0.3)

    #Ajusto una recta
    popt, pcov = curve_fit(lineal, X, Yi, sigma=0.3*np.ones(11))
    a = popt[0]
    b = popt[1]

    #Obtengo la predicción del ajuste para 0.5 y lo guardo
    Xa = 0.5
    Ya = lineal(Xa,a,b)
    Ya_rep.append(Ya)

#Armo el histograma de los valores de Ya
hist, bins = np.histogram(Ya_rep, bins="sturges", density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

#Armo la gaussiana que debería encajar con el histograma
paso = np.linspace(0,3.5,100)
mu = Xa*pendiente + ordenada
s = np.sqrt(cov[0, 0] * Xa**2 + cov[1, 1] + 2 * Xa * cov[0, 1])
gauss = norm.pdf(paso, mu, s)

#grafico el histograma y la gaussiana
plt.figure()
plt.bar(bin_centers, hist, width=(bins[1] - bins[0]), align="center")
#plt.errorbar(bin_centers, hist, yerr=np.sqrt(hist), fmt='o', label='Histograma')
plt.plot(paso,gauss,label="gaussiana esperada")
plt.xlabel("Valor de ya")
plt.ylabel("Densidad")
plt.title("Histograma de ya")
plt.legend()





