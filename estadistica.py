#Programa: parametros estadisticos previos al metodo montecarlo
#Tras hallar los valores de forma aleatoria para 1 ohmio de valor nominal y errores
# del 1% y 10%, obtendra las desviaciones estándar correspondientes de cada
#segmento para obtener valores de ese error relativo 
import numpy as np

#Inicializacion de variables
media = 1 #R0
desviacion_estandar = 0.10 #Sigma general
R = np.zeros(7) #Resistencia de cada segmento
sigma = np.zeros(7) #Sigma de cada segmento

#Generacion valores aleatorios de las resistencias de los segmentos
#Calculo de la desviacion estandar de cada segmento
for n in range(0,7):
    R[n] = np.random.normal(loc=media,scale=desviacion_estandar)
    sigma[n] = desviacion_estandar*R[n]
    print(n+1,R[n],sigma[n])
