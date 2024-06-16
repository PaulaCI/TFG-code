#Programa FDTD_3D fuente suave, sin ciclo for, PEC
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time

#INICIALIZACION DE VARIABLES
#Constantes del medio
N_max = 41
Ni = N_max
Nj = N_max
Nk = N_max
sigma_x = np.zeros((Ni,Nj,Nk))  
sigma_y = np.zeros((Ni,Nj,Nk)) 
sigma_z = np.zeros((Ni,Nj,Nk))
epsilon_x = np.zeros((Ni,Nj,Nk))  
epsilon_y = np.zeros((Ni,Nj,Nk)) 
epsilon_z = np.zeros((Ni,Nj,Nk))  
epsilon_x[:,:,:] = 8.8541878176*10**-12 
epsilon_y[:,:,:] = 8.8541878176*10**-12 
epsilon_z[:,:,:] = 8.8541878176*10**-12 
mu_x = np.zeros((Ni,Nj,Nk))  
mu_y = np.zeros((Ni,Nj,Nk)) 
mu_z = np.zeros((Ni,Nj,Nk))  
mu_x[:,:,:] = 4*math.pi*10**-7
mu_y[:,:,:] = 4*math.pi*10**-7 
mu_z[:,:,:] = 4*math.pi*10**-7 
sigma_m_x = np.zeros((Ni,Nj,Nk))  
sigma_m_y = np.zeros((Ni,Nj,Nk)) 
sigma_m_z = np.zeros((Ni,Nj,Nk))
esc = math.sqrt((8.8541878176*10**-12)/(4*math.pi*10**-7))
c = 299792458
#Constantes de paso
PPW = 20 #HabrÃ­a que probar con 15
As = 10 * 10**-3 #Probar tambiÃ©n con 5mm
ho = 0.5 #Este valor se ha escrito siguiendo el Ãºnico criterio de que debÃ­a 
         #ser inferior a 1.0
At = As*0.99/(c*math.sqrt(3))
lim_step = 8000
fmax = c/(PPW*As)
p = 1/(math.pi*fmax)
to = 5*p
t = 0
#Constantes algebraicas
C_Ex = (epsilon_x - 0.5*sigma_x*At)/(epsilon_x + 0.5*sigma_x*At)
D_Ex = At/((epsilon_x + 0.5*sigma_x*At))
C_Ey = (epsilon_y - 0.5*sigma_y*At)/(epsilon_y + 0.5*sigma_y*At)
D_Ey = At/((epsilon_y + 0.5*sigma_y*At))
C_Ez = (epsilon_z - 0.5*sigma_z*At)/(epsilon_z + 0.5*sigma_z*At)
D_Ez = At/((epsilon_z + 0.5*sigma_z*At))
C_Hx = (mu_x - 0.5*sigma_m_x*At)/(mu_x + 0.5*sigma_m_x*At)
D_Hx = At/(mu_x + 0.5*sigma_m_x*At)
C_Hy = (mu_y - 0.5*sigma_m_y*At)/(mu_y + 0.5*sigma_m_y*At)
D_Hy = At/(mu_y + 0.5*sigma_m_y*At)
C_Hz = (mu_z - 0.5*sigma_m_z*At)/(mu_z + 0.5*sigma_m_z*At)
D_Hz = At/(mu_z + 0.5*sigma_m_z*At)
#InicializaciÃ³n a 0 de los campos
H_x = np.zeros((Ni,Nj,Nk))          
H_y = np.zeros((Ni,Nj,Nk))
H_z = np.zeros((Ni,Nj,Nk))
E_x = np.zeros((Ni,Nj,Nk))
E_y = np.zeros((Ni,Nj,Nk))
E_z = np.zeros((Ni,Nj,Nk))

#CALCULOS
times=[]
probe1=[]
probe2=[]
probe3=[]
probe4=[]
probe5=[]
probe6=[]

for n in range(1,lim_step+1):
    t = n*At
    gauss = math.exp(-(t-to)**2/p**2)
    gauss_t = math.exp(-(t+As/(2*c)+At/2-to)**2/p**2)
    #ActualizaciÃ³n campo magnÃ©tico H con condiciones (PEC)
    H_x[0:Ni,0:Nj-1,0:Nk-1] = C_Hx[0:Ni,0:Nj-1,0:Nk-1] * H_x[0:Ni,0:Nj-1,0:Nk-1] - D_Hx[0:Ni,0:Nj-1,0:Nk-1] * (E_y[0:Ni,0:Nj-1,0:Nk-1] - E_y[0:Ni,0:Nj-1,1:Nk] + E_z[0:Ni,1:Nj,0:Nk-1] - E_z[0:Ni,0:Nj-1,0:Nk-1]) / As
    H_x[0:Ni,0:Nj-1,Nk-1] = C_Hx[0:Ni,0:Nj-1,Nk-1] * H_x[0:Ni,0:Nj-1,Nk-1] - D_Hx[0:Ni,0:Nj-1,Nk-1] * (E_y[0:Ni,0:Nj-1,Nk-1] + E_z[0:Ni,1:Nj,Nk-1] - E_z[0:Ni,0:Nj-1,Nk-1]) / As
    H_x[0:Ni,Nj-1,0:Nk-1] = C_Hx[0:Ni,Nj-1,0:Nk-1] * H_x[0:Ni,Nj-1,0:Nk-1] - D_Hx[0:Ni,Nj-1,0:Nk-1] * (E_y[0:Ni,Nj-1,0:Nk-1] - E_y[0:Ni,Nj-1,1:Nk] - E_z[0:Ni,Nj-1,0:Nk-1]) / As   
    H_x[0:Ni,Nj-1,Nk-1] = C_Hx[0:Ni,Nj-1,Nk-1] * H_x[0:Ni,Nj-1,Nk-1] - D_Hx[0:Ni,Nj-1,Nk-1] * (E_y[0:Ni,Nj-1,Nk-1] - E_z[0:Ni,Nj-1,Nk-1]) / As
    
    H_y[0:Ni-1,0:Nj,0:Nk-1] = C_Hy[0:Ni-1,0:Nj,0:Nk-1] * H_y[0:Ni-1,0:Nj,0:Nk-1] - D_Hy[0:Ni-1,0:Nj,0:Nk-1] * (E_x[0:Ni-1,0:Nj,1:Nk] - E_x[0:Ni-1,0:Nj,0:Nk-1] + E_z[0:Ni-1,0:Nj,0:Nk-1] - E_z[1:Ni,0:Nj,0:Nk-1]) / As
    H_y[0:Ni-1,0:Nj,Nk-1] = C_Hy[0:Ni-1,0:Nj,Nk-1] * H_y[0:Ni-1,0:Nj,Nk-1] - D_Hy[0:Ni-1,0:Nj,Nk-1] * (- E_x[0:Ni-1,0:Nj,Nk-1] + E_z[0:Ni-1,0:Nj,Nk-1] - E_z[1:Ni,0:Nj,Nk-1]) / As
    H_y[Ni-1,0:Nj,0:Nk-1] = C_Hy[Ni-1,0:Nj,0:Nk-1] * H_y[Ni-1,0:Nj,0:Nk-1] - D_Hy[Ni-1,0:Nj,0:Nk-1] * (E_x[Ni-1,0:Nj,1:Nk] - E_x[Ni-1,0:Nj,0:Nk-1] + E_z[Ni-1,0:Nj,0:Nk-1]) / As
    H_y[Ni-1,0:Nj,Nk-1] = C_Hy[Ni-1,0:Nj,Nk-1] * H_y[Ni-1,0:Nj,Nk-1] - D_Hy[Ni-1,0:Nj,Nk-1] * (- E_x[Ni-1,0:Nj,Nk-1] + E_z[Ni-1,0:Nj,Nk-1]) / As
    
    H_z[0:Ni-1,0:Nj-1,0:Nk] = C_Hz[0:Ni-1,0:Nj-1,0:Nk] * H_z[0:Ni-1,0:Nj-1,0:Nk] - D_Hz[0:Ni-1,0:Nj-1,0:Nk] * (E_x[0:Ni-1,0:Nj-1,0:Nk] - E_x[0:Ni-1,1:Nj,0:Nk] + E_y[1:Ni,0:Nj-1,0:Nk] - E_y[0:Ni-1,0:Nj-1,0:Nk])  / As  
    H_z[0:Ni-1,Nj-1,0:Nk] = C_Hz[0:Ni-1,Nj-1,0:Nk] * H_z[0:Ni-1,Nj-1,0:Nk] - D_Hz[0:Ni-1,Nj-1,0:Nk] * (E_x[0:Ni-1,Nj-1,0:Nk] + E_y[1:Ni,Nj-1,0:Nk] - E_y[0:Ni-1,Nj-1,0:Nk])  / As
    H_z[Ni-1,0:Nj-1,0:Nk] = C_Hz[Ni-1,0:Nj-1,0:Nk] * H_z[Ni-1,0:Nj-1,0:Nk] - D_Hz[Ni-1,0:Nj-1,0:Nk] * (E_x[Ni-1,0:Nj-1,0:Nk] - E_x[Ni-1,1:Nj,0:Nk] - E_y[Ni-1,0:Nj-1,0:Nk])  / As  
    H_z[Ni-1,Nj-1,0:Nk] = C_Hz[Ni-1,Nj-1,0:Nk] * H_z[Ni-1,Nj-1,0:Nk] - D_Hz[Ni-1,Nj-1,0:Nk] * (E_x[Ni-1,Nj-1,0:Nk] - E_y[Ni-1,Nj-1,0:Nk])  / As
   
    #ActualizaciÃ³n campo elÃ©ctrico E con condiciones (PEC)
    E_x[0:Ni,0,0] = C_Ex[0:Ni,0,0] * E_x[0:Ni,0,0] + D_Ex[0:Ni,0,0] * (H_z[0:Ni,0,0] - H_y[0:Ni,0,0]) / As 
    E_x[0:Ni,1:Nj,0] = C_Ex[0:Ni,1:Nj,0] * E_x[0:Ni,1:Nj,0] + D_Ex[0:Ni,1:Nj,0] * (H_z[0:Ni,1:Nj,0] - H_z[0:Ni,0:Nj-1,0] - H_y[0:Ni,1:Nj,0]) /As           
    E_x[0:Ni,0,1:Nk] = C_Ex[0:Ni,0,1:Nk] * E_x[0:Ni,0,1:Nk] + D_Ex[0:Ni,0,1:Nk] * (H_z[0:Ni,0,1:Nk] + H_y[0:Ni,0,0:Nk-1] - H_y[0:Ni,0,1:Nk]) /As
    E_x[0:Ni,1:Nj,1:Nk] = C_Ex[0:Ni,1:Nj,1:Nk] * E_x[0:Ni,1:Nj,1:Nk] + D_Ex[0:Ni,1:Nj,1:Nk] * (H_z[0:Ni,1:Nj,1:Nk] - H_z[0:Ni,0:Nj-1,1:Nk] + H_y[0:Ni,1:Nj,0:Nk-1] - H_y[0:Ni,1:Nj,1:Nk]) /As
    
    E_y[0,0:Nj,0] = C_Ey[0,0:Nj,0] * E_y[0,0:Nj,0] + D_Ey[0,0:Nj,0] * (- H_z[0,0:Nj,0] + H_x[0,0:Nj,0]) / As
    E_y[1:Ni,0:Nj,0] = C_Ey[1:Ni,0:Nj,0] * E_y[1:Ni,0:Nj,0] + D_Ey[1:Ni,0:Nj,0] * (H_z[0:Ni-1,0:Nj,0] - H_z[1:Ni,0:Nj,0] + H_x[1:Ni,0:Nj,0]) / As         
    E_y[0,0:Nj,1:Nk] = C_Ey[0,0:Nj,1:Nk] * E_y[0,0:Nj,1:Nk] + D_Ey[0,0:Nj,1:Nk] * (- H_z[0,0:Nj,1:Nk] + H_x[0,0:Nj,1:Nk] - H_x[0,0:Nj,0:Nk-1]) / As
    E_y[1:Ni,0:Nj,1:Nk] = C_Ey[1:Ni,0:Nj,1:Nk] * E_y[1:Ni,0:Nj,1:Nk] + D_Ey[1:Ni,0:Nj,1:Nk] * (H_z[0:Ni-1,0:Nj,1:Nk] - H_z[1:Ni,0:Nj,1:Nk] + H_x[1:Ni,0:Nj,1:Nk] - H_x[1:Ni,0:Nj,0:Nk-1]) / As
    
    E_z[0,0,0:Nk] = C_Ez[0,0,0:Nk] * E_z[0,0,0:Nk] + D_Ez[0,0,0:Nk] * (H_y[0,0,0:Nk] - H_x[0,0,0:Nk]) / As 
    E_z[1:Ni,0,0:Nk] = C_Ez[1:Ni,0,0:Nk] * E_z[1:Ni,0,0:Nk] + D_Ez[1:Ni,0,0:Nk] * (H_y[1:Ni,0,0:Nk] - H_y[0:Ni-1,0,0:Nk] - H_x[1:Ni,0,0:Nk]) / As            
    E_z[0,1:Nj,0:Nk] = C_Ez[0,1:Nj,0:Nk] * E_z[0,1:Nj,0:Nk] + D_Ez[0,1:Nj,0:Nk] * (H_y[0,1:Nj,0:Nk] + H_x[0,0:Nj-1,0:Nk] - H_x[0,1:Nj,0:Nk]) / As
    E_z[1:Ni,1:Nj,0:Nk] = C_Ez[1:Ni,1:Nj,0:Nk] * E_z[1:Ni,1:Nj,0:Nk] + D_Ez[1:Ni,1:Nj,0:Nk] * (H_y[1:Ni,1:Nj,0:Nk] - H_y[0:Ni-1,1:Nj,0:Nk] + H_x[1:Ni,0:Nj-1,0:Nk] - H_x[1:Ni,1:Nj,0:Nk]) / As
   
    #Pulso gaussiano
    E_z[10,10,10] = E_z[10,10,10] + gauss * D_Ez[10,10,10]          
    times.append(t)
    probe1.append(E_x[35,35,35]) 
    probe3.append(E_y[35,35,35]) 
    probe4.append(E_z[35,35,35]) 
       
    probe2.append(H_x[35,35,35])
    probe5.append(H_y[35,35,35])
    probe6.append(H_z[35,35,35])
    print(n,t,E_x[15,15,15],H_x[15,15,15])

# Gráfica de E_x[1,1,1] y H_x[1,1,1] frente al tiempo
#plt.plot(times, probe1, label='E_x[15,15,15]')
#plt.plot(times, probe3, label='E_y[15,15,15]')
#plt.plot(times, probe4, label='E_z[15,15,15]')
plt.plot(times, probe2, label='H_x[15,15,15]')
plt.plot(times, probe5, label='H_y[15,15,15]')
plt.plot(times, probe6, label='H_z[15,15,15]')
plt.xlabel('Tiempo')
plt.ylabel('Campo eléctrico y magnético')
plt.title('Evolución de E_x[15,15,15] y H_x[15,15,15] en el tiempo')
plt.legend()
plt.grid(True)
plt.show()

print('FIN')

