#Programa FDTD_1d PEC (fuente suave TF/SF) FFT
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

    # Calcula la transformada de Fourier de tus campos en el dominio del tiempo
    E_y_fft = np.fft.fft(E_y)

    # Obtén las frecuencias correspondientes
    freq = np.fft.fftfreq(Nk, d=At)

    # Gráfica los resultados en el dominio de las frecuencias
    plt.plot(freq, np.abs(E_y_fft),color='black')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.title(f'Tiempo: {n}')
    # Establecer límites del eje x e y
    plt.xlim(0, 1e10)
    plt.ylim(0, 12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    
    if n == 250:
        break
        
    #Actualiza y muestra el gráfico en cada iteración
    display(plt.gcf())
    clear_output(wait=True)
    time.sleep(0.01) #Pausa entre iteración
    plt.clf()
    
#Fin de la figura
plt.show()