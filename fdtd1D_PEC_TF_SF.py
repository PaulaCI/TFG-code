#Programa FDTD_1d PEC (fuente suave TF/SF)
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time

#INICIALIZACION DE VARIABLES
#Constantes del medio (convertir en vectores)
sigma_y = 0             
epsilon_y = 8.8541878176*10**-12
mu_x = 4*math.pi*10**-7
sigma_m_x = 0
esc = math.sqrt((8.8541878176*10**-12)/(4*math.pi*10**-7))
c = 299792458
#Constantes de paso
PPW = 20 #Habría que probar con 15
As = 10 * 10**-3 #Probar también con 5mm
ho = 0.5 #Este valor se ha escrito siguiendo el único criterio de que debía 
         #ser inferior a 1.0
At = As*ho/c
Nk = 81
lim_step = 300
fmax = c/(PPW*As)
p = 1/(math.pi*fmax)
to = 5*p
t = 0
#Constantes algebraicas
C_Ey = (epsilon_y - 0.5*sigma_y*At)/(epsilon_y + 0.5*sigma_y*At)
D_Ey = At*esc/(epsilon_y + 0.5*sigma_y*At)
C_Hx = (mu_x - 0.5*sigma_m_x*At)/(mu_x + 0.5*sigma_m_x*At)
D_Hx = At/((mu_x + 0.5*sigma_m_x*At)*esc)
#Campos
H_x = np.zeros(Nk)
E_y = np.zeros(Nk)

#Almacena el campo eléctrico en cada iteración
E_y_rep = np.zeros((lim_step, Nk))

#CUERPO PRINCIPAL
#Ciclo for principal
for n in range(1,lim_step+1):
    t=n*At
    #Fuente gaussiana
    gauss = math.exp(-(t-to)**2/(p**2))
    gauss_t = math.exp(-(t+As/(2*c)+At/2-to)**2/(p**2))
    #Actualización del campo H con condiciones de contorno PEC
    H_x[:Nk-1] = C_Hx * H_x[:Nk-1] - D_Hx * (E_y[:Nk-1] - E_y[1:Nk]) / As
    H_x[15] = H_x[15] - D_Hx * (gauss) / As
    H_x[Nk-1] = C_Hx * H_x[Nk-1] - D_Hx * (E_y[Nk-1]) / As
    #Actualización del campo E con condiciones de contorno PEC
    E_y[0] = C_Ey * E_y[0] + D_Ey * (H_x[0]) / As 
    E_y[1:Nk] = C_Ey * E_y[1:Nk] + D_Ey * (H_x[1:Nk] - H_x[0:Nk-1]) / As
    E_y[16] = E_y[16] + D_Ey * gauss_t / As
    #Fuente dura
    #E_y[16] = gauss
    #Fuente suave
    #E_y[16] = E_y[16] + D_Ey * gauss #Pendiente revisión de si incluir el coef algebraico
    #E_y[16] = E_y[16] + gauss
    #Almacenamiento del campo eléctrico en cada posición a lo largo de z
    E_y_rep[n-1,:] = E_y
    
    if n % 2 == 0:
        # Gráfico del campo eléctrico en cada iteración
        plt.plot(range(Nk), E_y, color = 'black')
        # Establecer límites del eje x e y
        plt.xlim(0, 80)
        plt.ylim(-1.4, 1.4)
        plt.xlabel('Posición en el eje z',fontsize=16)
        plt.ylabel('Campo Eléctrico [C]',fontsize=16)
        plt.title(f'Tiempo: {n}')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.show()
        time.sleep(0.005)  # Pausa entre iteraciÃ³n    
    # Actualiza y muestra el grÃ¡fico en cada iteraciÃ³n
        display(plt.gcf())
        if n == 184:
            break
        clear_output(wait=True)


#Fin de la figura
plt.show()