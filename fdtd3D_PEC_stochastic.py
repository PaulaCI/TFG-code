#Programa FDTD_3D fuente suave, sin ciclo for, PEC, S-FDTD
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
PPW = 20 
As = 10 * 10**-3 
ho = 0.5 
At = As*0.99/(c*math.sqrt(3))
lim_step = 1000
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
#S-FDTD
sto_Ex = np.zeros((Ni,Nj,Nk))
sto_Ey = np.zeros((Ni,Nj,Nk))
sto_Ez = np.zeros((Ni,Nj,Nk))
sto_Hx = np.zeros((Ni,Nj,Nk))
sto_Hy = np.zeros((Ni,Nj,Nk))
sto_Hz = np.zeros((Ni,Nj,Nk))
sto_sigma_x = np.zeros((Ni,Nj,Nk))  
sto_sigma_y = np.zeros((Ni,Nj,Nk)) 
sto_sigma_z = np.zeros((Ni,Nj,Nk))
sto_epsilon_x = np.zeros((Ni,Nj,Nk))  
sto_epsilon_y = np.zeros((Ni,Nj,Nk)) 
sto_epsilon_z = np.zeros((Ni,Nj,Nk))  
sto_mu_x = np.zeros((Ni,Nj,Nk))  
sto_mu_y = np.zeros((Ni,Nj,Nk)) 
sto_mu_z = np.zeros((Ni,Nj,Nk))  
sto_sigma_m_x = np.zeros((Ni,Nj,Nk))  
sto_sigma_m_y = np.zeros((Ni,Nj,Nk)) 
sto_sigma_m_z = np.zeros((Ni,Nj,Nk))

#Funciones
def update_Hx(H_x,E_x,E_y,E_z,C_Hx,D_Hx,As,At,Ni,Nj,Nk,sto_Hx,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_x,sto_mu_x,sigma_m_x,mu_x):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Hx * C_Hx * As * As / At
    B = D_Hx / At
    #Calculo campo magnético Hx y su varianza
    m[0:Ni,0:Nj-1,0:Nk-1] = E_y[0:Ni,0:Nj-1,0:Nk-1] - E_y[0:Ni,0:Nj-1,1:Nk] + E_z[0:Ni,1:Nj,0:Nk-1] - E_z[0:Ni,0:Nj-1,0:Nk-1]  
    H_x[0:Ni,0:Nj-1,0:Nk-1] = C_Hx[0:Ni,0:Nj-1,0:Nk-1] * H_x[0:Ni,0:Nj-1,0:Nk-1] - D_Hx[0:Ni,0:Nj-1,0:Nk-1] * m[0:Ni,0:Nj-1,0:Nk-1] / As
    sto_Hx[0:Ni,0:Nj-1,0:Nk-1] = C_Hx[0:Ni,0:Nj-1,0:Nk-1] * sto_Hx[0:Ni,0:Nj-1,0:Nk-1] + A[0:Ni,0:Nj-1,0:Nk-1] * (sigma_m_x[0:Ni,0:Nj-1,0:Nk-1]*sto_mu_x[0:Ni,0:Nj-1,0:Nk-1]-mu_x[0:Ni,0:Nj-1,0:Nk-1]*sto_sigma_m_x[0:Ni,0:Nj-1,0:Nk-1]) * H_x[0:Ni,0:Nj-1,0:Nk-1] + D_Hx[0:Ni,0:Nj-1,0:Nk-1] * (sto_Ey[0:Ni,0:Nj-1,0:Nk-1] - sto_Ey[0:Ni,0:Nj-1,1:Nk] + sto_Ez[0:Ni,1:Nj,0:Nk-1] - sto_Ez[0:Ni,0:Nj-1,0:Nk-1] - B[0:Ni,0:Nj-1,0:Nk-1] * (sto_mu_x[0:Ni,0:Nj-1,0:Nk-1]+1/2*At*sto_sigma_m_x[0:Ni,0:Nj-1,0:Nk-1]) * m[0:Ni,0:Nj-1,0:Nk-1])
    m[0:Ni,0:Nj-1,Nk-1] = E_y[0:Ni,0:Nj-1,Nk-1] + E_z[0:Ni,1:Nj,Nk-1] - E_z[0:Ni,0:Nj-1,Nk-1]
    H_x[0:Ni,0:Nj-1,Nk-1] = C_Hx[0:Ni,0:Nj-1,Nk-1] * H_x[0:Ni,0:Nj-1,Nk-1] - D_Hx[0:Ni,0:Nj-1,Nk-1] * m[0:Ni,0:Nj-1,Nk-1] / As
    sto_Hx[0:Ni,0:Nj-1,Nk-1] = C_Hx[0:Ni,0:Nj-1,Nk-1] * sto_Hx[0:Ni,0:Nj-1,Nk-1] + A[0:Ni,0:Nj-1,Nk-1] * (sigma_m_x[0:Ni,0:Nj-1,Nk-1]*sto_mu_x[0:Ni,0:Nj-1,Nk-1]-mu_x[0:Ni,0:Nj-1,Nk-1]*sto_sigma_m_x[0:Ni,0:Nj-1,Nk-1]) * H_x[0:Ni,0:Nj-1,Nk-1] + D_Hx[0:Ni,0:Nj-1,Nk-1] * (sto_Ey[0:Ni,0:Nj-1,Nk-1] + sto_Ez[0:Ni,1:Nj,Nk-1] - sto_Ez[0:Ni,0:Nj-1,Nk-1] - B[0:Ni,0:Nj-1,Nk-1] * (sto_mu_x[0:Ni,0:Nj-1,Nk-1]+1/2*At*sto_sigma_m_x[0:Ni,0:Nj-1,Nk-1]) * m[0:Ni,0:Nj-1,Nk-1])
    m[0:Ni,Nj-1,0:Nk-1] = E_y[0:Ni,Nj-1,0:Nk-1] - E_y[0:Ni,Nj-1,1:Nk] - E_z[0:Ni,Nj-1,0:Nk-1]  
    H_x[0:Ni,Nj-1,0:Nk-1] = C_Hx[0:Ni,Nj-1,0:Nk-1] * H_x[0:Ni,Nj-1,0:Nk-1] - D_Hx[0:Ni,Nj-1,0:Nk-1] * m[0:Ni,Nj-1,0:Nk-1] / As   
    sto_Hx[0:Ni,Nj-1,0:Nk-1] = C_Hx[0:Ni,Nj-1,0:Nk-1] * sto_Hx[0:Ni,Nj-1,0:Nk-1] + A[0:Ni,Nj-1,0:Nk-1] * (sigma_m_x[0:Ni,Nj-1,0:Nk-1]*sto_mu_x[0:Ni,Nj-1,0:Nk-1]-mu_x[0:Ni,Nj-1,0:Nk-1]*sto_sigma_m_x[0:Ni,Nj-1,0:Nk-1]) * H_x[0:Ni,Nj-1,0:Nk-1] + D_Hx[0:Ni,Nj-1,0:Nk-1] * (sto_Ey[0:Ni,Nj-1,0:Nk-1] - sto_Ey[0:Ni,Nj-1,1:Nk] - sto_Ez[0:Ni,Nj-1,0:Nk-1]  - B[0:Ni,Nj-1,0:Nk-1] * (sto_mu_x[0:Ni,Nj-1,0:Nk-1]+1/2*At*sto_sigma_m_x[0:Ni,Nj-1,0:Nk-1]) * m[0:Ni,Nj-1,0:Nk-1])
    m[0:Ni,Nj-1,Nk-1] = E_y[0:Ni,Nj-1,Nk-1] - E_z[0:Ni,Nj-1,Nk-1]
    H_x[0:Ni,Nj-1,Nk-1] = C_Hx[0:Ni,Nj-1,Nk-1] * H_x[0:Ni,Nj-1,Nk-1] - D_Hx[0:Ni,Nj-1,Nk-1] * m[0:Ni,Nj-1,Nk-1] / As
    sto_Hx[0:Ni,Nj-1,Nk-1] = C_Hx[0:Ni,Nj-1,Nk-1] * sto_Hx[0:Ni,Nj-1,Nk-1] + A[0:Ni,Nj-1,Nk-1] * (sigma_m_x[0:Ni,Nj-1,Nk-1]*sto_mu_x[0:Ni,Nj-1,Nk-1]-mu_x[0:Ni,Nj-1,Nk-1]*sto_sigma_m_x[0:Ni,Nj-1,Nk-1]) * H_x[0:Ni,Nj-1,Nk-1] + D_Hx[0:Ni,Nj-1,Nk-1] * (sto_Ey[0:Ni,Nj-1,Nk-1] - sto_Ez[0:Ni,Nj-1,Nk-1] - B[0:Ni,Nj-1,Nk-1] * (sto_mu_x[0:Ni,Nj-1,Nk-1]+1/2*At*sto_sigma_m_x[0:Ni,Nj-1,Nk-1]) * m[0:Ni,Nj-1,Nk-1])

def update_Hy(H_y,E_x,E_y,E_z,C_Hy,D_Hy,As,At,Ni,Nj,Nk,sto_Hy,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_y,sto_mu_y,sigma_m_y,mu_y):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Hy * C_Hy * As * As / At
    B = D_Hy / At
    #Calculo campo magnético Hy y su varianza
    m[0:Ni-1,0:Nj,0:Nk-1] = E_x[0:Ni-1,0:Nj,1:Nk] - E_x[0:Ni-1,0:Nj,0:Nk-1] + E_z[0:Ni-1,0:Nj,0:Nk-1] - E_z[1:Ni,0:Nj,0:Nk-1]
    H_y[0:Ni-1,0:Nj,0:Nk-1] = C_Hy[0:Ni-1,0:Nj,0:Nk-1] * H_y[0:Ni-1,0:Nj,0:Nk-1] - D_Hy[0:Ni-1,0:Nj,0:Nk-1] * m[0:Ni-1,0:Nj,0:Nk-1] / As
    sto_Hy[0:Ni-1,0:Nj,0:Nk-1] = C_Hy[0:Ni-1,0:Nj,0:Nk-1] * sto_Hy[0:Ni-1,0:Nj,0:Nk-1] + A[0:Ni-1,0:Nj,0:Nk-1] * (sigma_m_y[0:Ni-1,0:Nj,0:Nk-1]*sto_mu_y[0:Ni-1,0:Nj,0:Nk-1] - mu_y[0:Ni-1,0:Nj,0:Nk-1]*sto_sigma_m_y[0:Ni-1,0:Nj,0:Nk-1]) * H_y[0:Ni-1,0:Nj,0:Nk-1] + D_Hy[0:Ni-1,0:Nj,0:Nk-1] * (sto_Ex[0:Ni-1,0:Nj,1:Nk] - sto_Ex[0:Ni-1,0:Nj,0:Nk-1] + sto_Ez[0:Ni-1,0:Nj,0:Nk-1] - sto_Ez[1:Ni,0:Nj,0:Nk-1] - B[0:Ni-1,0:Nj,0:Nk-1]*(sto_mu_y[0:Ni-1,0:Nj,0:Nk-1]+1/2*At*sto_sigma_m_y[0:Ni-1,0:Nj,0:Nk-1]) * m[0:Ni-1,0:Nj,0:Nk-1])
    m[0:Ni-1,0:Nj,Nk-1] = - E_x[0:Ni-1,0:Nj,Nk-1] + E_z[0:Ni-1,0:Nj,Nk-1] - E_z[1:Ni,0:Nj,Nk-1]
    H_y[0:Ni-1,0:Nj,Nk-1] = C_Hy[0:Ni-1,0:Nj,Nk-1] * H_y[0:Ni-1,0:Nj,Nk-1] - D_Hy[0:Ni-1,0:Nj,Nk-1] * m[0:Ni-1,0:Nj,Nk-1] / As
    sto_Hy[0:Ni-1,0:Nj,Nk-1] = C_Hy[0:Ni-1,0:Nj,Nk-1] * sto_Hy[0:Ni-1,0:Nj,Nk-1] + A[0:Ni-1,0:Nj,Nk-1] * (sigma_m_y[0:Ni-1,0:Nj,Nk-1]*sto_mu_y[0:Ni-1,0:Nj,Nk-1] - mu_y[0:Ni-1,0:Nj,Nk-1]*sto_sigma_m_y[0:Ni-1,0:Nj,Nk-1]) * H_y[0:Ni-1,0:Nj,Nk-1] + D_Hy[0:Ni-1,0:Nj,Nk-1] * ( - sto_Ex[0:Ni-1,0:Nj,Nk-1] + sto_Ez[0:Ni-1,0:Nj,Nk-1] - sto_Ez[1:Ni,0:Nj,Nk-1] - B[0:Ni-1,0:Nj,Nk-1]*(sto_mu_y[0:Ni-1,0:Nj,Nk-1]+1/2*At*sto_sigma_m_y[0:Ni-1,0:Nj,Nk-1]) * m[0:Ni-1,0:Nj,Nk-1])
    m[Ni-1,0:Nj,0:Nk-1] = E_x[Ni-1,0:Nj,1:Nk] - E_x[Ni-1,0:Nj,0:Nk-1] + E_z[Ni-1,0:Nj,0:Nk-1]
    H_y[Ni-1,0:Nj,0:Nk-1] = C_Hy[Ni-1,0:Nj,0:Nk-1] * H_y[Ni-1,0:Nj,0:Nk-1] - D_Hy[Ni-1,0:Nj,0:Nk-1] * m[Ni-1,0:Nj,0:Nk-1] / As
    sto_Hy[Ni-1,0:Nj,0:Nk-1] = C_Hy[Ni-1,0:Nj,0:Nk-1] * sto_Hy[Ni-1,0:Nj,0:Nk-1] + A[Ni-1,0:Nj,0:Nk-1] * (sigma_m_y[Ni-1,0:Nj,0:Nk-1]*sto_mu_y[Ni-1,0:Nj,0:Nk-1] - mu_y[Ni-1,0:Nj,0:Nk-1]*sto_sigma_m_y[Ni-1,0:Nj,0:Nk-1]) * H_y[Ni-1,0:Nj,0:Nk-1] + D_Hy[Ni-1,0:Nj,0:Nk-1] * (sto_Ex[Ni-1,0:Nj,1:Nk] - sto_Ex[Ni-1,0:Nj,0:Nk-1] + sto_Ez[Ni-1,0:Nj,0:Nk-1] - B[Ni-1,0:Nj,0:Nk-1]*(sto_mu_y[Ni-1,0:Nj,0:Nk-1]+1/2*At*sto_sigma_m_y[Ni-1,0:Nj,0:Nk-1]) * m[Ni-1,0:Nj,0:Nk-1])
    m[Ni-1,0:Nj,Nk-1] = - E_x[Ni-1,0:Nj,Nk-1] + E_z[Ni-1,0:Nj,Nk-1]
    H_y[Ni-1,0:Nj,Nk-1] = C_Hy[Ni-1,0:Nj,Nk-1] * H_y[Ni-1,0:Nj,Nk-1] - D_Hy[Ni-1,0:Nj,Nk-1] * m[Ni-1,0:Nj,Nk-1] / As
    sto_Hy[Ni-1,0:Nj,Nk-1] = C_Hy[Ni-1,0:Nj,Nk-1] * sto_Hy[Ni-1,0:Nj,Nk-1] + A[Ni-1,0:Nj,Nk-1] * (sigma_m_y[Ni-1,0:Nj,Nk-1]*sto_mu_y[Ni-1,0:Nj,Nk-1] - mu_y[Ni-1,0:Nj,Nk-1]*sto_sigma_m_y[Ni-1,0:Nj,Nk-1]) * H_y[Ni-1,0:Nj,Nk-1] + D_Hy[Ni-1,0:Nj,Nk-1] * (-sto_Ex[Ni-1,0:Nj,Nk-1] + sto_Ez[Ni-1,0:Nj,Nk-1] - B[Ni-1,0:Nj,Nk-1]*(sto_mu_y[Ni-1,0:Nj,Nk-1]+1/2*At*sto_sigma_m_y[Ni-1,0:Nj,Nk-1]) * m[Ni-1,0:Nj,Nk-1])

def update_Hz(H_z,E_x,E_y,E_z,C_Hz,D_Hz,As,At,Ni,Nj,Nk,sto_Hz,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_z,sto_mu_z,sigma_m_z,mu_z):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Hz * C_Hz * As * As / At
    B = D_Hz / At
    #Calculo campo magnético Hz y su varianza
    m[0:Ni-1,0:Nj-1,0:Nk] = E_x[0:Ni-1,0:Nj-1,0:Nk] - E_x[0:Ni-1,1:Nj,0:Nk] + E_y[1:Ni,0:Nj-1,0:Nk] - E_y[0:Ni-1,0:Nj-1,0:Nk]
    H_z[0:Ni-1,0:Nj-1,0:Nk] = C_Hz[0:Ni-1,0:Nj-1,0:Nk] * H_z[0:Ni-1,0:Nj-1,0:Nk] - D_Hz[0:Ni-1,0:Nj-1,0:Nk] * m[0:Ni-1,0:Nj-1,0:Nk]  / As  
    sto_Hz[0:Ni-1,0:Nj-1,0:Nk] = C_Hz[0:Ni-1,0:Nj-1,0:Nk] * sto_Hz[0:Ni-1,0:Nj-1,0:Nk] + A[0:Ni-1,0:Nj-1,0:Nk] * (sigma_m_z[0:Ni-1,0:Nj-1,0:Nk]*sto_mu_z[0:Ni-1,0:Nj-1,0:Nk]-mu_z[0:Ni-1,0:Nj-1,0:Nk]*sto_sigma_m_z[0:Ni-1,0:Nj-1,0:Nk]) * H_z[0:Ni-1,0:Nj-1,0:Nk] + D_Hz[0:Ni-1,0:Nj-1,0:Nk] * (sto_Ex[0:Ni-1,0:Nj-1,0:Nk] - sto_Ex[0:Ni-1,1:Nj,0:Nk] + sto_Ey[1:Ni,0:Nj-1,0:Nk] - sto_Ey[0:Ni-1,0:Nj-1,0:Nk]- B[0:Ni-1,0:Nj-1,0:Nk] * (sto_mu_z[0:Ni-1,0:Nj-1,0:Nk]+1/2*At*sto_sigma_m_z[0:Ni-1,0:Nj-1,0:Nk]) * m[0:Ni-1,0:Nj-1,0:Nk])
    m[0:Ni-1,Nj-1,0:Nk] = E_x[0:Ni-1,Nj-1,0:Nk] + E_y[1:Ni,Nj-1,0:Nk] - E_y[0:Ni-1,Nj-1,0:Nk]
    H_z[0:Ni-1,Nj-1,0:Nk] = C_Hz[0:Ni-1,Nj-1,0:Nk] * H_z[0:Ni-1,Nj-1,0:Nk] - D_Hz[0:Ni-1,Nj-1,0:Nk] * m[0:Ni-1,Nj-1,0:Nk]  / As
    sto_Hz[0:Ni-1,Nj-1,0:Nk] = C_Hz[0:Ni-1,Nj-1,0:Nk] * sto_Hz[0:Ni-1,Nj-1,0:Nk] + A[0:Ni-1,Nj-1,0:Nk] * (sigma_m_z[0:Ni-1,Nj-1,0:Nk]*sto_mu_z[0:Ni-1,Nj-1,0:Nk]-mu_z[0:Ni-1,Nj-1,0:Nk]*sto_sigma_m_z[0:Ni-1,Nj-1,0:Nk]) * H_z[0:Ni-1,Nj-1,0:Nk] + D_Hz[0:Ni-1,Nj-1,0:Nk] * (sto_Ex[0:Ni-1,Nj-1,0:Nk] + sto_Ey[1:Ni,Nj-1,0:Nk] - sto_Ey[0:Ni-1,Nj-1,0:Nk] - B[0:Ni-1,Nj-1,0:Nk] * (sto_mu_z[0:Ni-1,Nj-1,0:Nk]+1/2*At*sto_sigma_m_z[0:Ni-1,Nj-1,0:Nk]) * m[0:Ni-1,Nj-1,0:Nk])
    m[Ni-1,0:Nj-1,0:Nk] = E_x[Ni-1,0:Nj-1,0:Nk] - E_x[Ni-1,1:Nj,0:Nk] - E_y[Ni-1,0:Nj-1,0:Nk]
    H_z[Ni-1,0:Nj-1,0:Nk] = C_Hz[Ni-1,0:Nj-1,0:Nk] * H_z[Ni-1,0:Nj-1,0:Nk] - D_Hz[Ni-1,0:Nj-1,0:Nk] * m[Ni-1,0:Nj-1,0:Nk]  / As  
    sto_Hz[Ni-1,0:Nj-1,0:Nk] = C_Hz[Ni-1,0:Nj-1,0:Nk] * sto_Hz[Ni-1,0:Nj-1,0:Nk] + A[Ni-1,0:Nj-1,0:Nk] * (sigma_m_z[Ni-1,0:Nj-1,0:Nk]*sto_mu_z[Ni-1,0:Nj-1,0:Nk]-mu_z[Ni-1,0:Nj-1,0:Nk]*sto_sigma_m_z[Ni-1,0:Nj-1,0:Nk]) * H_z[Ni-1,0:Nj-1,0:Nk] + D_Hz[Ni-1,0:Nj-1,0:Nk] * (sto_Ex[Ni-1,0:Nj-1,0:Nk] - sto_Ex[Ni-1,1:Nj,0:Nk] - sto_Ey[Ni-1,0:Nj-1,0:Nk] - B[Ni-1,0:Nj-1,0:Nk] * (sto_mu_z[Ni-1,0:Nj-1,0:Nk]+1/2*At*sto_sigma_m_z[Ni-1,0:Nj-1,0:Nk]) * m[Ni-1,0:Nj-1,0:Nk])
    m[Ni-1,Nj-1,0:Nk] = E_x[Ni-1,Nj-1,0:Nk] - E_y[Ni-1,Nj-1,0:Nk]
    H_z[Ni-1,Nj-1,0:Nk] = C_Hz[Ni-1,Nj-1,0:Nk] * H_z[Ni-1,Nj-1,0:Nk] - D_Hz[Ni-1,Nj-1,0:Nk] * m[Ni-1,Nj-1,0:Nk]  / As
    sto_Hz[Ni-1,Nj-1,0:Nk] = C_Hz[Ni-1,Nj-1,0:Nk] * sto_Hz[Ni-1,Nj-1,0:Nk] + A[Ni-1,Nj-1,0:Nk] * (sigma_m_z[Ni-1,Nj-1,0:Nk]*sto_mu_z[Ni-1,Nj-1,0:Nk]-mu_z[Ni-1,Nj-1,0:Nk]*sto_sigma_m_z[Ni-1,Nj-1,0:Nk]) * H_z[Ni-1,Nj-1,0:Nk] + D_Hz[Ni-1,Nj-1,0:Nk] * (sto_Ex[Ni-1,Nj-1,0:Nk] - sto_Ey[Ni-1,Nj-1,0:Nk] - B[Ni-1,Nj-1,0:Nk] * (sto_mu_z[Ni-1,Nj-1,0:Nk]+1/2*At*sto_sigma_m_z[Ni-1,Nj-1,0:Nk]) * m[Ni-1,Nj-1,0:Nk])
   
def update_Ex(E_x,H_x,H_y,H_z,C_Ex,D_Ex,As,At,Ni,Nj,Nk,sto_Ex,sto_Hx,sto_Hy,sto_Hz,sto_sigma_x,sto_epsilon_x,sigma_x,epsilon_x):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Ex * C_Ex * As * As / At
    B = D_Ex / At
    #Calculo campo eléctrico Ex y su varianza
    m[0:Ni,0,0] = H_z[0:Ni,0,0] - H_y[0:Ni,0,0]
    E_x[0:Ni,0,0] = C_Ex[0:Ni,0,0] * E_x[0:Ni,0,0] + D_Ex[0:Ni,0,0] * m[0:Ni,0,0] / As 
    sto_Ex[0:Ni,0,0] = C_Ex[0:Ni,0,0] * sto_Ex[0:Ni,0,0] + A[0:Ni,0,0] * (sigma_x[0:Ni,0,0]*sto_epsilon_x[0:Ni,0,0]-epsilon_x[0:Ni,0,0]*sto_sigma_x[0:Ni,0,0]) * E_x[0:Ni,0,0] + D_Ex[0:Ni,0,0] * (sto_Hz[0:Ni,0,0] - sto_Hy[0:Ni,0,0] - B[0:Ni,0,0] * (sto_epsilon_x[0:Ni,0,0]+1/2*At*sto_sigma_x[0:Ni,0,0]) * m[0:Ni,0,0])
    m[0:Ni,1:Nj,0] = H_z[0:Ni,1:Nj,0] - H_z[0:Ni,0:Nj-1,0] - H_y[0:Ni,1:Nj,0]
    E_x[0:Ni,1:Nj,0] = C_Ex[0:Ni,1:Nj,0] * E_x[0:Ni,1:Nj,0] + D_Ex[0:Ni,1:Nj,0] * m[0:Ni,1:Nj,0] /As           
    sto_Ex[0:Ni,1:Nj,0] = C_Ex[0:Ni,1:Nj,0] * sto_Ex[0:Ni,1:Nj,0] + A[0:Ni,1:Nj,0] * (sigma_x[0:Ni,1:Nj,0]*sto_epsilon_x[0:Ni,1:Nj,0]-epsilon_x[0:Ni,1:Nj,0]*sto_sigma_x[0:Ni,1:Nj,0]) * E_x[0:Ni,1:Nj,0] + D_Ex[0:Ni,1:Nj,0] * (sto_Hz[0:Ni,1:Nj,0] - sto_Hz[0:Ni,0:Nj-1,0] - sto_Hy[0:Ni,1:Nj,0] - B[0:Ni,1:Nj,0] * (sto_epsilon_x[0:Ni,1:Nj,0]+1/2*At*sto_sigma_x[0:Ni,1:Nj,0]) * m[0:Ni,1:Nj,0])
    m[0:Ni,0,1:Nk] = H_z[0:Ni,0,1:Nk] + H_y[0:Ni,0,0:Nk-1] - H_y[0:Ni,0,1:Nk]
    E_x[0:Ni,0,1:Nk] = C_Ex[0:Ni,0,1:Nk] * E_x[0:Ni,0,1:Nk] + D_Ex[0:Ni,0,1:Nk] * m[0:Ni,0,1:Nk] /As
    sto_Ex[0:Ni,0,1:Nk] = C_Ex[0:Ni,0,1:Nk] * sto_Ex[0:Ni,0,1:Nk] + A[0:Ni,0,1:Nk] * (sigma_x[0:Ni,0,1:Nk]*sto_epsilon_x[0:Ni,0,1:Nk]-epsilon_x[0:Ni,0,1:Nk]*sto_sigma_x[0:Ni,0,1:Nk]) * E_x[0:Ni,0,1:Nk] + D_Ex[0:Ni,0,1:Nk] * (sto_Hz[0:Ni,0,1:Nk] + sto_Hy[0:Ni,0,0:Nk-1] - sto_Hy[0:Ni,0,1:Nk] - B[0:Ni,0,1:Nk] * (sto_epsilon_x[0:Ni,0,1:Nk]+1/2*At*sto_sigma_x[0:Ni,0,1:Nk]) * m[0:Ni,0,1:Nk])
    m[0:Ni,1:Nj,1:Nk] = H_z[0:Ni,1:Nj,1:Nk] - H_z[0:Ni,0:Nj-1,1:Nk] + H_y[0:Ni,1:Nj,0:Nk-1] - H_y[0:Ni,1:Nj,1:Nk]
    E_x[0:Ni,1:Nj,1:Nk] = C_Ex[0:Ni,1:Nj,1:Nk] * E_x[0:Ni,1:Nj,1:Nk] + D_Ex[0:Ni,1:Nj,1:Nk] * m[0:Ni,1:Nj,1:Nk] /As
    sto_Ex[0:Ni,1:Nj,1:Nk] = C_Ex[0:Ni,1:Nj,1:Nk] * sto_Ex[0:Ni,1:Nj,1:Nk] + A[0:Ni,1:Nj,1:Nk] * (sigma_x[0:Ni,1:Nj,1:Nk]*sto_epsilon_x[0:Ni,1:Nj,1:Nk]-epsilon_x[0:Ni,1:Nj,1:Nk]*sto_sigma_x[0:Ni,1:Nj,1:Nk]) * E_x[0:Ni,1:Nj,1:Nk] + D_Ex[0:Ni,1:Nj,1:Nk]* (sto_Hz[0:Ni,1:Nj,1:Nk] - sto_Hz[0:Ni,0:Nj-1,1:Nk] + sto_Hy[0:Ni,1:Nj,0:Nk-1] - sto_Hy[0:Ni,1:Nj,1:Nk] - B[0:Ni,1:Nj,1:Nk] * (sto_epsilon_x[0:Ni,1:Nj,1:Nk]+1/2*At*sto_sigma_x[0:Ni,1:Nj,1:Nk]) * m[0:Ni,1:Nj,1:Nk])

def update_Ey(E_y,H_x,H_y,H_z,C_Ey,D_Ey,As,At,Ni,Nj,Nk,sto_Ey,sto_Hx,sto_Hy,sto_Hz,sto_sigma_y,sto_epsilon_y,sigma_y,epsilon_y):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Ey * C_Ey * As * As / At
    B = D_Ey / At
    #Calculo campo eléctrico Ey y su varianza
    m[0,0:Nj,0] = - H_z[0,0:Nj,0] + H_x[0,0:Nj,0]   
    E_y[0,0:Nj,0] = C_Ey[0,0:Nj,0] * E_y[0,0:Nj,0] + D_Ey[0,0:Nj,0] * m[0,0:Nj,0] / As
    sto_Ey[0,0:Nj,0] = C_Ey[0,0:Nj,0] * sto_Ey[0,0:Nj,0] + A[0,0:Nj,0] * (sigma_y[0,0:Nj,0]*sto_epsilon_y[0,0:Nj,0]-epsilon_y[0,0:Nj,0]*sto_sigma_y[0,0:Nj,0]) * E_y[0,0:Nj,0] + D_Ey[0,0:Nj,0] * (- sto_Hz[0,0:Nj,0] + sto_Hx[0,0:Nj,0] - B[0,0:Nj,0] * (sto_epsilon_y[0,0:Nj,0]+1/2*At*sto_sigma_y[0,0:Nj,0]) * m[0,0:Nj,0])
    m[1:Ni,0:Nj,0] = H_z[0:Ni-1,0:Nj,0] - H_z[1:Ni,0:Nj,0] + H_x[1:Ni,0:Nj,0]
    E_y[1:Ni,0:Nj,0] = C_Ey[1:Ni,0:Nj,0] * E_y[1:Ni,0:Nj,0] + D_Ey[1:Ni,0:Nj,0] * m[1:Ni,0:Nj,0] / As         
    sto_Ey[1:Ni,0:Nj,0] = C_Ey[1:Ni,0:Nj,0] * sto_Ey[1:Ni,0:Nj,0] + A[1:Ni,0:Nj,0] * (sigma_y[1:Ni,0:Nj,0]*sto_epsilon_y[1:Ni,0:Nj,0]-epsilon_y[1:Ni,0:Nj,0]*sto_sigma_y[1:Ni,0:Nj,0]) * E_y[1:Ni,0:Nj,0] + D_Ey[1:Ni,0:Nj,0] * (sto_Hz[0:Ni-1,0:Nj,0] - sto_Hz[1:Ni,0:Nj,0] + sto_Hx[1:Ni,0:Nj,0] - B[1:Ni,0:Nj,0] * (sto_epsilon_y[1:Ni,0:Nj,0]+1/2*At*sto_sigma_y[1:Ni,0:Nj,0]) * m[1:Ni,0:Nj,0])
    m[0,0:Nj,1:Nk] = - H_z[0,0:Nj,1:Nk] + H_x[0,0:Nj,1:Nk] - H_x[0,0:Nj,0:Nk-1]
    E_y[0,0:Nj,1:Nk] = C_Ey[0,0:Nj,1:Nk] * E_y[0,0:Nj,1:Nk] + D_Ey[0,0:Nj,1:Nk] * m[0,0:Nj,1:Nk] / As
    sto_Ey[0,0:Nj,1:Nk] = C_Ey[0,0:Nj,1:Nk] * sto_Ey[0,0:Nj,1:Nk] + A[0,0:Nj,1:Nk] * (sigma_y[0,0:Nj,1:Nk]*sto_epsilon_y[0,0:Nj,1:Nk]-epsilon_y[0,0:Nj,1:Nk]*sto_sigma_y[0,0:Nj,1:Nk]) * E_y[0,0:Nj,1:Nk] + D_Ey[0,0:Nj,1:Nk] * (- sto_Hz[0,0:Nj,1:Nk] + sto_Hx[0,0:Nj,1:Nk] - sto_Hx[0,0:Nj,0:Nk-1] - B[0,0:Nj,1:Nk] * (sto_epsilon_y[0,0:Nj,1:Nk]+1/2*At*sto_sigma_y[0,0:Nj,1:Nk]) * m[0,0:Nj,1:Nk])
    m[1:Ni,0:Nj,1:Nk] = H_z[0:Ni-1,0:Nj,1:Nk] - H_z[1:Ni,0:Nj,1:Nk] + H_x[1:Ni,0:Nj,1:Nk] - H_x[1:Ni,0:Nj,0:Nk-1]
    E_y[1:Ni,0:Nj,1:Nk] = C_Ey[1:Ni,0:Nj,1:Nk] * E_y[1:Ni,0:Nj,1:Nk] + D_Ey[1:Ni,0:Nj,1:Nk] * m[1:Ni,0:Nj,1:Nk] / As
    sto_Ey[1:Ni,0:Nj,1:Nk] = C_Ey[1:Ni,0:Nj,1:Nk] * sto_Ey[1:Ni,0:Nj,1:Nk] + A[1:Ni,0:Nj,1:Nk] * (sigma_y[1:Ni,0:Nj,1:Nk]*sto_epsilon_y[1:Ni,0:Nj,1:Nk]-epsilon_y[1:Ni,0:Nj,1:Nk]*sto_sigma_y[1:Ni,0:Nj,1:Nk]) * E_y[1:Ni,0:Nj,1:Nk] + D_Ey[1:Ni,0:Nj,1:Nk] * (sto_Hz[0:Ni-1,0:Nj,1:Nk] - sto_Hz[1:Ni,0:Nj,1:Nk] + sto_Hx[1:Ni,0:Nj,1:Nk] - sto_Hx[1:Ni,0:Nj,0:Nk-1] - B[1:Ni,0:Nj,1:Nk] * (sto_epsilon_y[1:Ni,0:Nj,1:Nk]+1/2*At*sto_sigma_y[1:Ni,0:Nj,1:Nk]) * m[1:Ni,0:Nj,1:Nk])

def update_Ez(E_z,H_x,H_y,H_z,C_Ez,D_Ez,As,At,Ni,Nj,Nk,sto_Ez,sto_Hx,sto_Hy,sto_Hz,sto_sigma_z,sto_epsilon_z,sigma_z,epsilon_z):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Ez * C_Ez * As * As / At
    B = D_Ez / At
    #Calculo campo eléctrico Ez y su varianza
    m[0,0,0:Nk] = H_y[0,0,0:Nk] - H_x[0,0,0:Nk] 
    E_z[0,0,0:Nk] = C_Ez[0,0,0:Nk] * E_z[0,0,0:Nk] + D_Ez[0,0,0:Nk] * m[0,0,0:Nk] / As 
    sto_Ez[0,0,0:Nk] = C_Ez[0,0,0:Nk] * sto_Ez[0,0,0:Nk] + A[0,0,0:Nk] * (sigma_z[0,0,0:Nk]*sto_epsilon_z[0,0,0:Nk]-epsilon_z[0,0,0:Nk]*sto_sigma_z[0,0,0:Nk]) * E_z[0,0,0:Nk] + D_Ez[0,0,0:Nk] * (sto_Hy[0,0,0:Nk] - sto_Hx[0,0,0:Nk] - B[0,0,0:Nk] * (sto_epsilon_z[0,0,0:Nk]-1/2*At*sto_sigma_z[0,0,0:Nk]) * m[0,0,0:Nk])
    m[1:Ni,0,0:Nk] = H_y[1:Ni,0,0:Nk] - H_y[0:Ni-1,0,0:Nk] - H_x[1:Ni,0,0:Nk]
    E_z[1:Ni,0,0:Nk] = C_Ez[1:Ni,0,0:Nk] * E_z[1:Ni,0,0:Nk] + D_Ez[1:Ni,0,0:Nk] * m[1:Ni,0,0:Nk] / As            
    sto_Ez[1:Ni,0,0:Nk] = C_Ez[1:Ni,0,0:Nk] * sto_Ez[1:Ni,0,0:Nk] + A[1:Ni,0,0:Nk] * (sigma_z[1:Ni,0,0:Nk]*sto_epsilon_z[1:Ni,0,0:Nk]-epsilon_z[1:Ni,0,0:Nk]*sto_sigma_z[1:Ni,0,0:Nk]) * E_z[1:Ni,0,0:Nk] + D_Ez[1:Ni,0,0:Nk] * (sto_Hy[1:Ni,0,0:Nk] - sto_Hy[0:Ni-1,0,0:Nk] - sto_Hx[1:Ni,0,0:Nk] - B[1:Ni,0,0:Nk] * (sto_epsilon_z[1:Ni,0,0:Nk]-1/2*At*sto_sigma_z[1:Ni,0,0:Nk]) * m[1:Ni,0,0:Nk])
    m[0,1:Nj,0:Nk] = H_y[0,1:Nj,0:Nk] + H_x[0,0:Nj-1,0:Nk] - H_x[0,1:Nj,0:Nk]
    E_z[0,1:Nj,0:Nk] = C_Ez[0,1:Nj,0:Nk] * E_z[0,1:Nj,0:Nk] + D_Ez[0,1:Nj,0:Nk] * m[0,1:Nj,0:Nk] / As
    sto_Ez[0,1:Nj,0:Nk] = C_Ez[0,1:Nj,0:Nk] * sto_Ez[0,1:Nj,0:Nk] + A[0,1:Nj,0:Nk] * (sigma_z[0,1:Nj,0:Nk]*sto_epsilon_z[0,1:Nj,0:Nk]-epsilon_z[0,1:Nj,0:Nk]*sto_sigma_z[0,1:Nj,0:Nk]) * E_z[0,1:Nj,0:Nk] + D_Ez[0,1:Nj,0:Nk] * (sto_Hy[0,1:Nj,0:Nk] + sto_Hx[0,0:Nj-1,0:Nk] - sto_Hx[0,1:Nj,0:Nk] - B[0,1:Nj,0:Nk] * (sto_epsilon_z[0,1:Nj,0:Nk]-1/2*At*sto_sigma_z[0,1:Nj,0:Nk]) * m[0,1:Nj,0:Nk])
    m[1:Ni,1:Nj,0:Nk] = H_y[1:Ni,1:Nj,0:Nk] - H_y[0:Ni-1,1:Nj,0:Nk] + H_x[1:Ni,0:Nj-1,0:Nk] - H_x[1:Ni,1:Nj,0:Nk]
    E_z[1:Ni,1:Nj,0:Nk] = C_Ez[1:Ni,1:Nj,0:Nk] * E_z[1:Ni,1:Nj,0:Nk] + D_Ez[1:Ni,1:Nj,0:Nk] * m[1:Ni,1:Nj,0:Nk] / As
    sto_Ez[1:Ni,1:Nj,0:Nk] = C_Ez[1:Ni,1:Nj,0:Nk] * sto_Ez[1:Ni,1:Nj,0:Nk] + A[1:Ni,1:Nj,0:Nk] * (sigma_z[1:Ni,1:Nj,0:Nk]*sto_epsilon_z[1:Ni,1:Nj,0:Nk]-epsilon_z[1:Ni,1:Nj,0:Nk]*sto_sigma_z[1:Ni,1:Nj,0:Nk]) * E_z[1:Ni,1:Nj,0:Nk] + D_Ez[1:Ni,1:Nj,0:Nk] * (sto_Hy[1:Ni,1:Nj,0:Nk] - sto_Hy[0:Ni-1,1:Nj,0:Nk] + sto_Hx[1:Ni,0:Nj-1,0:Nk] - sto_Hx[1:Ni,1:Nj,0:Nk] - B[1:Ni,1:Nj,0:Nk] * (sto_epsilon_z[1:Ni,1:Nj,0:Nk]-1/2*At*sto_sigma_z[1:Ni,1:Nj,0:Nk]) * m[1:Ni,1:Nj,0:Nk])

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
    #Actualización campo magnético H con condiciones (PEC) y su varianza
    update_Hx(H_x,E_x,E_y,E_z,C_Hx,D_Hx,As,At,Ni,Nj,Nk,sto_Hx,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_x,sto_mu_x,sigma_m_x,mu_x)
    update_Hy(H_y,E_x,E_y,E_z,C_Hy,D_Hy,As,At,Ni,Nj,Nk,sto_Hy,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_y,sto_mu_y,sigma_m_y,mu_y)
    update_Hz(H_z,E_x,E_y,E_z,C_Hz,D_Hz,As,At,Ni,Nj,Nk,sto_Hz,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_z,sto_mu_z,sigma_m_z,mu_z)
    #Actualización campo eléctrico E con condiciones (PEC) y su varianza
    update_Ex(E_x,H_x,H_y,H_z,C_Ex,D_Ex,As,At,Ni,Nj,Nk,sto_Ex,sto_Hx,sto_Hy,sto_Hz,sto_sigma_x,sto_epsilon_x,sigma_x,epsilon_x)
    update_Ey(E_y,H_x,H_y,H_z,C_Ey,D_Ey,As,At,Ni,Nj,Nk,sto_Ey,sto_Hx,sto_Hy,sto_Hz,sto_sigma_y,sto_epsilon_y,sigma_y,epsilon_y)
    update_Ez(E_z,H_x,H_y,H_z,C_Ez,D_Ez,As,At,Ni,Nj,Nk,sto_Ez,sto_Hx,sto_Hy,sto_Hz,sto_sigma_z,sto_epsilon_z,sigma_z,epsilon_z)   
    #Pulso gaussiano
    E_z[10,10,10] = E_z[10,10,10] + gauss          
    times.append(t)
    probe1.append(E_x[20,20,20]) 
    probe3.append(E_y[20,20,20]) 
    probe4.append(E_z[20,20,20]) 
       
    probe2.append(H_x[19,19,19])
    probe5.append(H_y[19,19,19])
    probe6.append(H_z[19,19,19])
    print(n,t,E_x[15,15,15],H_x[15,15,15])

# Gráfica de E_x[1,1,1] y H_x[1,1,1] frente al tiempo
plt.plot(times, probe1, label='E_x[15,15,15]')
plt.plot(times, probe3, label='E_y[15,15,15]')
plt.plot(times, probe4, label='E_z[15,15,15]')
#plt.plot(times, probe2, label='H_x[15,15,15]')
#plt.plot(times, probe5, label='H_y[15,15,15]')
#plt.plot(times, probe6, label='H_z[15,15,15]')
plt.xlabel('Tiempo')
plt.ylabel('Campo eléctrico y magnético')
plt.title('Evolución de E_x[15,15,15] y H_x[15,15,15] en el tiempo')
plt.legend()
plt.grid(True)
plt.show()
print('FIN')
