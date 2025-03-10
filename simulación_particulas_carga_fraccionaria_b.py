# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:34:45 2023

@author: imend
"""

import numpy as np
import matplotlib.pyplot as plt

## Simulo N=1000 mediciones de fondo como 1000 vectores de 100 elementos (detectores) cada uno.
N = 1000  # Número de vectores
n = 100   # Número de elementos en cada vector
mu = 0.1 # Parámetro mu para la distribución de Poisson

# Genero los vectores aleatorios
vectores = np.random.poisson(mu, size=(N, n))
vectores = np.where(vectores > 1, 1, vectores) # Cambia los elementos mayores a 1 por 1


## Armo un histograma con las distancias entre eventos consecutivos
distancias = []
for vector in vectores:
    posiciones = np.where(vector == 1)[0] #posiciones de los 1 en la fila
    diferencias = np.diff(posiciones) # Calcula las distancias entre eventos consecutivos
    distancias.extend(diferencias)
distancias = distancias - np.ones(len(distancias)) #cuento la distancia como la cantidad de no detecciones entre 2 detecciones

#obtener el histograma común y normalizado
hist, bins = np.histogram(distancias, bins=np.arange(0,max(distancias))) # Obtener el histograma
h_norm = hist/np.sum(hist)

# Calcular las alturas de las barras de error
error_heights = np.sqrt(hist)/np.sum(hist)
# Calcular el centro de cada bin
bin_centers = 0.5 * (bins[:-1] + bins[1:])

#graficar la exponencial
from scipy.stats import expon
lambda_estimado = 1 / np.mean(distancias) #estimo el lambda de la exponencial
exp = expon.pdf(bins[:-1], scale=1/lambda_estimado) #valores esperados para la exponencial

# Graficar el histograma con barras de error
plt.figure()
plt.xlabel('Distancia', fontsize=20)
plt.ylabel('Densidad (%)', fontsize=20)
plt.bar(bin_centers, h_norm, width=np.diff(bins), align='center', alpha=0.5,label='Eventos de fondo')
plt.errorbar(bin_centers, h_norm, yerr=error_heights, fmt='none', color='r', capsize=3)
plt.plot(bins[:-1], exp, 'g', label='Distribución Exponencial')
plt.legend(fontsize=20)

#%%
## Armo el histograma del estadístico LLR cuando H0 es cierta

def LLR(lambda0,lambda1,rep,p=0):
    LL = []
    r = 0
    while r < rep:
        V = np.random.poisson(mu, size=(N, n))
        S = np.random.choice([0, 1], size=(1000, 100), p=[1-p, p])
        M = S + V
        M = np.where(M > 1, 1, M)
        dist = []
        for s in M:
            posiciones = np.where(s == 1)[0]  #posiciones de los 1 en la fila
            diferencias = np.diff(posiciones) #calcula las distancias entre eventos consecutivos
            dist.extend(diferencias)
        dist = dist - np.ones(len(dist))
        L1 = expon.pdf(dist,scale=1/lambda1)
        L0 = expon.pdf(dist,scale=1/lambda0)
        L = np.prod(L1/L0)
        LL.append(L)
        r = r+1
    return((-2)*np.log(LL))

# Obtenidos del otro código
lambda0 = 0.11681282
lambda1 = 0.14014115

#obtener el histograma común y normalizado para H0
hist_L, bins_L = np.histogram(LLR(lambda0,lambda1,1000,0), bins='auto') # Obtener el histograma
h_norm_L = hist_L/np.sum(hist_L)

# Calcular las alturas de las barras de error
error = np.sqrt(hist_L)/np.sum(hist_L)
# Calcular el centro de cada bin
bin_cent = 0.5 * (bins_L[:-1] + bins_L[1:])

#graficar histograma
plt.figure()
#plt.xlabel('Distancia', fontsize=20)
plt.ylabel('Densidad (%)', fontsize=20)
plt.bar(bin_cent, h_norm_L, width=np.diff(bins_L), align='center', alpha=0.5,label='Estadístico LLR')
plt.errorbar(bin_cent, h_norm_L, yerr=error, fmt='none', color='r', capsize=3)
plt.legend(fontsize=20)
