# -*- coding: utf-8 -*-
"""
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

hist, bins = np.histogram(distancias, bins=np.arange(0,max(distancias)), density=True) # Obtener el histograma normalizado

# Grafico
#plt.figure()
#plt.hist(distancias, bins=np.arange(0,max(distancias)), density=True)
#plt.xlabel('Distancia', fontsize=20)
#plt.ylabel('Densidad (%)', fontsize=20)
#plt.show()


## Aplico test chi-2 para ver la compatibilidad con el histograma de variable exponencial
from scipy.stats import expon, chi2

lambda_estimado = 1 / np.mean(distancias) #estimo el lambda de la exponencial
exp = expon.pdf(bins[:-1], scale=1/lambda_estimado) #valores esperados para la exponencial
estadistico_chi2 = np.sum((hist - exp)**2 / exp)

# Obtengo el p-valor para el test chi-sqr
grados_libertad = len(bins) - 2
p_valor_chi2 = 1 - chi2.cdf(estadistico_chi2, grados_libertad)


## Aplico el test de runs 
from statsmodels.sandbox.stats import runs

# Calcular el p-valor utilizando el test de runs
p_valor_runs = runs.runstest_2samp(hist, exp)[1]


## Obtengo p-valor conjunto usando el estadístico con distrib. chi2 de 4 grados de libertad p_valor_total
p_valor_total = -2*np.log(p_valor_chi2*p_valor_runs) #estadístico con distrib. chi2 4 G.L.

p_val_conjunto = 1 - chi2.cdf(p_valor_total, 4) #si es mayor a 0.05 no rechazo H0

#%%
## Simulo 1000 veces el paso de una partícula mCP. Tiene 2% prob. de interactuar con cada detector (sin ser detenida)
S = np.random.choice([0, 1], size=(1000, 100), p=[0.98, 0.02])
M = S + vectores
M = np.where(M > 1, 1, M) #cambio los elementos mayores a 1 por 1


## Con M puedo sacar un nuevo lambda, correspondiente a que haya mCP además de fondo
distanciasH1 = []
for s in M:
    posiciones = np.where(s == 1)[0] #posiciones de los 1 en la fila
    diferencias = np.diff(posiciones) #calcula las distancias entre eventos consecutivos
    distanciasH1.extend(diferencias)
distanciasH1 = distanciasH1 - np.ones(len(distanciasH1))

# Estimo el lambda cuando hay mCP, y su correspondiente exponencial
lambda1 = 1 / np.mean(distanciasH1) 

# El lambda para H0 y su exponencial ya están calculado de antes (los de solo eventos de fondo)
lambda0 = lambda_estimado

#%%
## defino el estadístico LLR
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

# Esta función da tantas mediciones como quiera del estadístico LLR.
# Si considero H0 cierta dejo p=0, si considero H1 cierta pongo p=0.02

#%%
#Quiero calcular el poder del LLR test para p=0.02

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

#obtener el histograma común y normalizado para H1
hist_L, bins_L = np.histogram(LLR(lambda0,lambda1,1000,0.02), bins='auto') # Obtener el histograma
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

#%%
# Graficar el histograma normalizado y la distribución exponencial
plt.hist(distancias, bins=np.arange(0,max(distancias)), density=True, alpha=0.5, label='Eventos de fondo')
plt.plot(bins[:-1], exp, 'r', label='Distribución Exponencial')

plt.xlabel('Distancia', fontsize=20)
plt.ylabel('Densidad (%)', fontsize=20)
plt.legend(fontsize=20)
plt.show()
#%%


