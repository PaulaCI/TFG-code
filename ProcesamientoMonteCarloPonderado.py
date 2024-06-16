import os
import numpy as np
import math
import matplotlib.pyplot as plt

# Lectura de los archivos
dir_path = r'C:\Users\anjer\OneDrive\Documentos\Universidad de Granada\Año 7\Cuatrimestre 2\Python\Datos'
data = []

for o in range(1, 101):
    file_name = f"montecarlo_g_1_{o}.txt"
    file_path = os.path.join(dir_path, file_name)
    
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            third_column = [float(line.split()[2]) for line in lines]
            data.append(third_column)

# Crear una matriz con ceros
matrix = np.zeros((2500, 102))

# Rellenar la matriz con los datos de las terceras columnas de los archivos
for i in range(2500):
    for j in range(100):
        matrix[i, j + 2] = data[j][i]

# Generar pesos con un perfil gaussiano
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

mu = 50
sigma = 0.07 * 100 / 6

# Crear los pesos usando la función gaussiana
x = np.arange(1, 101)
pesos = gaussian(x, mu, sigma)

# Normalizar los pesos para que sumen 1
pesos /= np.sum(pesos)

# Cálculos estadísticos
estadistica = np.zeros((2500, 4))

c = 299792458
As = 10**-3
At = As * 0.9 / (c * math.sqrt(3))

for i in range(2500):
    matrix[i, 0] = i + 1 
    matrix[i, 1] = (i + 1) * At
    # Seleccionar las columnas deseadas para calcular la media y la varianza
    valores_fila = matrix[i, 2:]  # Seleccionar valores desde la tercera hasta la centésima primera columna 
    # Calcular la media ponderada
    media_ponderada = np.average(valores_fila, weights=pesos)
    # Calcular la desviación estándar ponderada
    varianza_ponderada = np.average((valores_fila - media_ponderada) ** 2,weights=pesos)
    desviacion_estandar_ponderada = np.sqrt(varianza_ponderada)  

    estadistica[i, 0] = matrix[i, 1]
    estadistica[i, 1] = media_ponderada
    estadistica[i, 2] = desviacion_estandar_ponderada

# Extraer los datos para la visualización
x = estadistica[:, 0]  # Segunda columna de estadistica para el eje x
media = estadistica[:, 1]  # Tercera columna de estadistica para la media
desviacion_estandar = estadistica[:, 2]  # Cuarta columna de estadistica para la varianza


# Graficar la media
#plt.plot(x, media, '-', label='Media MC-FDTD 10%', color = 'blue', markersize=3)

# Graficar la desviación estándar
#ax.plot(x, desviacion_estandar, label='Desviacion estandar')
plt.plot(x, desviacion_estandar, '--', label='Desviacion estandar MC-FDTD 10%', color = 'green', markersize=3)
plt.legend()
# Mostrar la gráfica
plt.show()