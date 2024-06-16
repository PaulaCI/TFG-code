#Programa FDTD_2d condiciones de contorno periódicas, sin for, todos los puntos
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time

#INICIALIZACION DE VARIABLES
#Constantes del medio (convertir en vectores)
sigma_z = 0             
epsilon_z = 8.8541878176*10**-12
mu_x = 4*math.pi*10**-7
mu_y = 4*math.pi*10**-7
sigma_m_x = 0
sigma_m_y = 0
esc = math.sqrt((8.8541878176*10**-12)/(4*math.pi*10**-7))
c = 299792458
#Constantes de paso
PPW = 20 #Habría que probar con 15
As = 10 * 10**-3 #Probar también con 5mm
ho = 0.5 #Este valor se ha escrito siguiendo el único criterio de que debía 
         #ser inferior a 1.0
At = As*ho/(c*math.sqrt(2))
N_max = 61
Ni = N_max
Nj = N_max
Nk = N_max
lim_step = 600
fmax = c/(PPW*As)
p = 1/(math.pi*fmax)
to = 5*p
t = 0
#Constantes algebraicas
C_Ez = (epsilon_z - 0.5*sigma_z*At)/(epsilon_z + 0.5*sigma_z*At)
D_Ez = At/(esc*(epsilon_z + 0.5*sigma_z*At))
C_Hx = (mu_x - 0.5*sigma_m_x*At)/(mu_x + 0.5*sigma_m_x*At)
C_Hy = (mu_y - 0.5*sigma_m_y*At)/(mu_y + 0.5*sigma_m_y*At)
D_Hx = esc*At/(mu_x + 0.5*sigma_m_x*At)
D_Hy = esc*At/(mu_y + 0.5*sigma_m_y*At)
#Campos
E_z = np.zeros((Ni,Nj))
H_x = np.zeros((Ni,Nj))
H_y = np.zeros((Ni,Nj))

#CUERPO PRINCIPAL
#Ciclo for principal
for n in range(1,lim_step+1):
    t=n*At
    #Fuente gaussiana
    gauss = math.exp(-(t-to)**2/p**2)
    #Actualización del campo H con condiciones de contorno periódicas
    H_x[0:Ni,0:Nj-1] = C_Hx * H_x[0:Ni,0:Nj-1] - D_Hx * (E_z[0:Ni,1:Nj] - E_z[0:Ni,0:Nj-1]) / As
    H_x[0:Ni,Nj-1] = C_Hx * H_x[0:Ni,Nj-1] - D_Hx * (E_z[0:Ni,0] - E_z[0:Ni,Nj-1]) / As
    H_y[0:Ni-1,0:Nj] = C_Hy * H_y[0:Ni-1,0:Nj] - D_Hy * (E_z[0:Ni-1,0:Nj] - E_z[1:Ni,0:Nj]) / As
    H_y[Ni-1,0:Nj] = C_Hy * H_y[Ni-1,0:Nj] - D_Hy * (-E_z[0,0:Nj] + E_z[Ni-1,0:Nj]) / As
    #Actualización del campo E con condiciones de contorno periódicas
    E_z[0,0] = C_Ez * E_z[0,0] + D_Ez * (H_y[0,0] - H_y[Ni-1,0] - H_x[0,0] + H_x[0,Nj-1]) / As
    E_z[1:Ni,0] = C_Ez * E_z[1:Ni,0] + D_Ez * (H_y[1:Ni,0] - H_y[0:Ni-1,0] - H_x[1:Ni,0] + H_x[1:Ni,Nj-1]) / As
    E_z[0,1:Nj] = C_Ez * E_z[0,1:Nj] + D_Ez * (H_y[0,1:Nj] - H_y[Ni-1,1:Nj] - H_x[0,1:Nj] + H_x[0,0:Nj-1]) / As
    E_z[1:Ni,1:Nj] = C_Ez * E_z[1:Ni,1:Nj] + D_Ez * (H_y[1:Ni,1:Nj] - H_y[0:Ni-1,1:Nj] - H_x[1:Ni,1:Nj] + H_x[1:Ni,0:Nj-1]) / As
    E_z[10,10] = E_z[10,10] + gauss
# Gráfico del campo eléctrico en cada iteración
    if n % 4 == 0:
        plt.imshow(E_z, cmap='viridis', vmin=-0.25, vmax=0.2)
        plt.xlabel('Posición en el eje x')
        plt.ylabel('Posición en el eje y')
        plt.title(f'Tiempo: {n}')

        # Actualiza y muestra el gráfico en cada iteración
        display(plt.gcf())
        clear_output(wait=True)
        time.sleep(0.005)  # Pausa entre iteración

# Fin de la figura
plt.show()