import os
import matplotlib.pyplot as plt

# Directorio y nombre del archivo
directorio = r'C:\Users\anjer\OneDrive\Documentos\Universidad de Granada\Año 7\Cuatrimestre 2\Python\Datos'
archivo = 'montecarlo_g_1_1.txt'

# Ruta completa del archivo
ruta_archivo = os.path.join(directorio, archivo)

# Listas para almacenar los datos de las columnas
x = []
y = []

# Leer el archivo y extraer los datos de las columnas
with open(ruta_archivo, 'r') as file:
    for line in file:
        data = line.split()
        x.append(float(data[1]))  # Segunda columna como x
        y.append(float(data[2]))  # Tercera columna como y

# Crear la gráfica
plt.plot(x, y, marker='o', linestyle='-')
plt.title('Gráfico de los datos')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()