#Programa FDTD_3D fuente suave, PEC, S-FDTD
#7 segmentos de cable
#derivada normalizada gaussiana
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
#INICIALIZACION DE VARIABLES
#Constantes del medio
N_max = 60 
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
c = 299792458
#Constantes de paso
PPW = 20 
As = 10**-3 
At = As*0.9/(c*math.sqrt(3))
lim_step = 2500
#lim_step = 82
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
#Inicialización a 0 de los campos
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
#Carga e intensidad por unidad de longitud
I = np.zeros((Ni,Nj,Nk))
Q = np.zeros((Ni,Nj,Nk))
sto_I = np.zeros((Ni,Nj,Nk))
sto_Q = np.zeros((Ni,Nj,Nk))
L = np.zeros((Ni,Nj,Nk))
R = np.zeros((Ni,Nj,Nk))
sto_L = np.zeros((Ni,Nj,Nk))
sto_R = np.zeros((Ni,Nj,Nk))
l = 7*As
d = 2 * 2*10**-5
r = d/2
#d = 2 * 2*10**-6
#1%
#Rvar = np.array([0.999922979125512,1.005100543660772,0.9941299871337071,1.000778728593687,1.0034004400561682,1.0017477981385732,0.991150602110106]) 
#desvia = np.array([0.00999922979125512,0.01005100543660772,0.009941299871337071,0.010007787285936872,0.010034004400561682,0.010017477981385732,0.00991150602110106])
#10%
Rvar = np.array([1.0480821440721129,0.9247264298969866,1.0283193018954377,0.953509278393018,0.9577257685371066,0.8946043410020965,0.9735852636761192]) 
desvia = np.array([0.10480821440721129,0.09247264298969866,0.10283193018954379,0.09535092783930181,0.09577257685371066,0.08946043410020965,0.09735852636761193])
for k in range(0,7):
    R[30,27+k,30] = Rvar[k]/As
    L[30,27+k,30] = (7.462*10**-7)
    sto_R[30,27+k,30] = desvia[k]/As

#Funciones
def update_Hx(H_x,E_x,E_y,E_z,C_Hx,D_Hx,As,At,Ni,Nj,Nk,sto_Hx,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_x,sto_mu_x,sigma_m_x,mu_x):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Hx * C_Hx / At
    B = D_Hx / (As)
    #Calculo campo magnético Hx y su varianza
    m[0:Ni,0:Nj-1,0:Nk-1] = E_y[0:Ni,0:Nj-1,0:Nk-1] - E_y[0:Ni,0:Nj-1,1:Nk] + E_z[0:Ni,1:Nj,0:Nk-1] - E_z[0:Ni,0:Nj-1,0:Nk-1]  
    H_x[0:Ni,0:Nj-1,0:Nk-1] = C_Hx[0:Ni,0:Nj-1,0:Nk-1] * H_x[0:Ni,0:Nj-1,0:Nk-1] - D_Hx[0:Ni,0:Nj-1,0:Nk-1] * m[0:Ni,0:Nj-1,0:Nk-1] / As
    sto_Hx[0:Ni,0:Nj-1,0:Nk-1] = C_Hx[0:Ni,0:Nj-1,0:Nk-1] * sto_Hx[0:Ni,0:Nj-1,0:Nk-1] + A[0:Ni,0:Nj-1,0:Nk-1] * (sigma_m_x[0:Ni,0:Nj-1,0:Nk-1]*sto_mu_x[0:Ni,0:Nj-1,0:Nk-1]-mu_x[0:Ni,0:Nj-1,0:Nk-1]*sto_sigma_m_x[0:Ni,0:Nj-1,0:Nk-1]) * H_x[0:Ni,0:Nj-1,0:Nk-1] + D_Hx[0:Ni,0:Nj-1,0:Nk-1] * (sto_Ey[0:Ni,0:Nj-1,0:Nk-1] - sto_Ey[0:Ni,0:Nj-1,1:Nk] + sto_Ez[0:Ni,1:Nj,0:Nk-1] - sto_Ez[0:Ni,0:Nj-1,0:Nk-1] - B[0:Ni,0:Nj-1,0:Nk-1] * (sto_mu_x[0:Ni,0:Nj-1,0:Nk-1]+1/2*At*sto_sigma_m_x[0:Ni,0:Nj-1,0:Nk-1]) * m[0:Ni,0:Nj-1,0:Nk-1])/As
    m[0:Ni,0:Nj-1,Nk-1] = E_y[0:Ni,0:Nj-1,Nk-1] + E_z[0:Ni,1:Nj,Nk-1] - E_z[0:Ni,0:Nj-1,Nk-1]
    H_x[0:Ni,0:Nj-1,Nk-1] = C_Hx[0:Ni,0:Nj-1,Nk-1] * H_x[0:Ni,0:Nj-1,Nk-1] - D_Hx[0:Ni,0:Nj-1,Nk-1] * m[0:Ni,0:Nj-1,Nk-1] / As
    sto_Hx[0:Ni,0:Nj-1,Nk-1] = C_Hx[0:Ni,0:Nj-1,Nk-1] * sto_Hx[0:Ni,0:Nj-1,Nk-1] + A[0:Ni,0:Nj-1,Nk-1] * (sigma_m_x[0:Ni,0:Nj-1,Nk-1]*sto_mu_x[0:Ni,0:Nj-1,Nk-1]-mu_x[0:Ni,0:Nj-1,Nk-1]*sto_sigma_m_x[0:Ni,0:Nj-1,Nk-1]) * H_x[0:Ni,0:Nj-1,Nk-1] + D_Hx[0:Ni,0:Nj-1,Nk-1] * (sto_Ey[0:Ni,0:Nj-1,Nk-1] + sto_Ez[0:Ni,1:Nj,Nk-1] - sto_Ez[0:Ni,0:Nj-1,Nk-1] - B[0:Ni,0:Nj-1,Nk-1] * (sto_mu_x[0:Ni,0:Nj-1,Nk-1]+1/2*At*sto_sigma_m_x[0:Ni,0:Nj-1,Nk-1]) * m[0:Ni,0:Nj-1,Nk-1])/As
    m[0:Ni,Nj-1,0:Nk-1] = E_y[0:Ni,Nj-1,0:Nk-1] - E_y[0:Ni,Nj-1,1:Nk] - E_z[0:Ni,Nj-1,0:Nk-1]  
    H_x[0:Ni,Nj-1,0:Nk-1] = C_Hx[0:Ni,Nj-1,0:Nk-1] * H_x[0:Ni,Nj-1,0:Nk-1] - D_Hx[0:Ni,Nj-1,0:Nk-1] * m[0:Ni,Nj-1,0:Nk-1] / As   
    sto_Hx[0:Ni,Nj-1,0:Nk-1] = C_Hx[0:Ni,Nj-1,0:Nk-1] * sto_Hx[0:Ni,Nj-1,0:Nk-1] + A[0:Ni,Nj-1,0:Nk-1] * (sigma_m_x[0:Ni,Nj-1,0:Nk-1]*sto_mu_x[0:Ni,Nj-1,0:Nk-1]-mu_x[0:Ni,Nj-1,0:Nk-1]*sto_sigma_m_x[0:Ni,Nj-1,0:Nk-1]) * H_x[0:Ni,Nj-1,0:Nk-1] + D_Hx[0:Ni,Nj-1,0:Nk-1] * (sto_Ey[0:Ni,Nj-1,0:Nk-1] - sto_Ey[0:Ni,Nj-1,1:Nk] - sto_Ez[0:Ni,Nj-1,0:Nk-1]  - B[0:Ni,Nj-1,0:Nk-1] * (sto_mu_x[0:Ni,Nj-1,0:Nk-1]+1/2*At*sto_sigma_m_x[0:Ni,Nj-1,0:Nk-1]) * m[0:Ni,Nj-1,0:Nk-1])/As
    m[0:Ni,Nj-1,Nk-1] = E_y[0:Ni,Nj-1,Nk-1] - E_z[0:Ni,Nj-1,Nk-1]
    H_x[0:Ni,Nj-1,Nk-1] = C_Hx[0:Ni,Nj-1,Nk-1] * H_x[0:Ni,Nj-1,Nk-1] - D_Hx[0:Ni,Nj-1,Nk-1] * m[0:Ni,Nj-1,Nk-1] / As
    sto_Hx[0:Ni,Nj-1,Nk-1] = C_Hx[0:Ni,Nj-1,Nk-1] * sto_Hx[0:Ni,Nj-1,Nk-1] + A[0:Ni,Nj-1,Nk-1] * (sigma_m_x[0:Ni,Nj-1,Nk-1]*sto_mu_x[0:Ni,Nj-1,Nk-1]-mu_x[0:Ni,Nj-1,Nk-1]*sto_sigma_m_x[0:Ni,Nj-1,Nk-1]) * H_x[0:Ni,Nj-1,Nk-1] + D_Hx[0:Ni,Nj-1,Nk-1] * (sto_Ey[0:Ni,Nj-1,Nk-1] - sto_Ez[0:Ni,Nj-1,Nk-1] - B[0:Ni,Nj-1,Nk-1] * (sto_mu_x[0:Ni,Nj-1,Nk-1]+1/2*At*sto_sigma_m_x[0:Ni,Nj-1,Nk-1]) * m[0:Ni,Nj-1,Nk-1])/As

def update_Hy(H_y,E_x,E_y,E_z,C_Hy,D_Hy,As,At,Ni,Nj,Nk,sto_Hy,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_y,sto_mu_y,sigma_m_y,mu_y):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Hy * C_Hy / At
    B = D_Hy / (As)
    #Calculo campo magnético Hy y su varianza
    m[0:Ni-1,0:Nj,0:Nk-1] = E_x[0:Ni-1,0:Nj,1:Nk] - E_x[0:Ni-1,0:Nj,0:Nk-1] + E_z[0:Ni-1,0:Nj,0:Nk-1] - E_z[1:Ni,0:Nj,0:Nk-1]
    H_y[0:Ni-1,0:Nj,0:Nk-1] = C_Hy[0:Ni-1,0:Nj,0:Nk-1] * H_y[0:Ni-1,0:Nj,0:Nk-1] - D_Hy[0:Ni-1,0:Nj,0:Nk-1] * m[0:Ni-1,0:Nj,0:Nk-1] / As
    sto_Hy[0:Ni-1,0:Nj,0:Nk-1] = C_Hy[0:Ni-1,0:Nj,0:Nk-1] * sto_Hy[0:Ni-1,0:Nj,0:Nk-1] + A[0:Ni-1,0:Nj,0:Nk-1] * (sigma_m_y[0:Ni-1,0:Nj,0:Nk-1]*sto_mu_y[0:Ni-1,0:Nj,0:Nk-1] - mu_y[0:Ni-1,0:Nj,0:Nk-1]*sto_sigma_m_y[0:Ni-1,0:Nj,0:Nk-1]) * H_y[0:Ni-1,0:Nj,0:Nk-1] + D_Hy[0:Ni-1,0:Nj,0:Nk-1] * (sto_Ex[0:Ni-1,0:Nj,1:Nk] - sto_Ex[0:Ni-1,0:Nj,0:Nk-1] + sto_Ez[0:Ni-1,0:Nj,0:Nk-1] - sto_Ez[1:Ni,0:Nj,0:Nk-1] - B[0:Ni-1,0:Nj,0:Nk-1]*(sto_mu_y[0:Ni-1,0:Nj,0:Nk-1]+1/2*At*sto_sigma_m_y[0:Ni-1,0:Nj,0:Nk-1]) * m[0:Ni-1,0:Nj,0:Nk-1])/As
    m[0:Ni-1,0:Nj,Nk-1] = - E_x[0:Ni-1,0:Nj,Nk-1] + E_z[0:Ni-1,0:Nj,Nk-1] - E_z[1:Ni,0:Nj,Nk-1]
    H_y[0:Ni-1,0:Nj,Nk-1] = C_Hy[0:Ni-1,0:Nj,Nk-1] * H_y[0:Ni-1,0:Nj,Nk-1] - D_Hy[0:Ni-1,0:Nj,Nk-1] * m[0:Ni-1,0:Nj,Nk-1] / As
    sto_Hy[0:Ni-1,0:Nj,Nk-1] = C_Hy[0:Ni-1,0:Nj,Nk-1] * sto_Hy[0:Ni-1,0:Nj,Nk-1] + A[0:Ni-1,0:Nj,Nk-1] * (sigma_m_y[0:Ni-1,0:Nj,Nk-1]*sto_mu_y[0:Ni-1,0:Nj,Nk-1] - mu_y[0:Ni-1,0:Nj,Nk-1]*sto_sigma_m_y[0:Ni-1,0:Nj,Nk-1]) * H_y[0:Ni-1,0:Nj,Nk-1] + D_Hy[0:Ni-1,0:Nj,Nk-1] * ( - sto_Ex[0:Ni-1,0:Nj,Nk-1] + sto_Ez[0:Ni-1,0:Nj,Nk-1] - sto_Ez[1:Ni,0:Nj,Nk-1] - B[0:Ni-1,0:Nj,Nk-1]*(sto_mu_y[0:Ni-1,0:Nj,Nk-1]+1/2*At*sto_sigma_m_y[0:Ni-1,0:Nj,Nk-1]) * m[0:Ni-1,0:Nj,Nk-1])/As
    m[Ni-1,0:Nj,0:Nk-1] = E_x[Ni-1,0:Nj,1:Nk] - E_x[Ni-1,0:Nj,0:Nk-1] + E_z[Ni-1,0:Nj,0:Nk-1]
    H_y[Ni-1,0:Nj,0:Nk-1] = C_Hy[Ni-1,0:Nj,0:Nk-1] * H_y[Ni-1,0:Nj,0:Nk-1] - D_Hy[Ni-1,0:Nj,0:Nk-1] * m[Ni-1,0:Nj,0:Nk-1] / As
    sto_Hy[Ni-1,0:Nj,0:Nk-1] = C_Hy[Ni-1,0:Nj,0:Nk-1] * sto_Hy[Ni-1,0:Nj,0:Nk-1] + A[Ni-1,0:Nj,0:Nk-1] * (sigma_m_y[Ni-1,0:Nj,0:Nk-1]*sto_mu_y[Ni-1,0:Nj,0:Nk-1] - mu_y[Ni-1,0:Nj,0:Nk-1]*sto_sigma_m_y[Ni-1,0:Nj,0:Nk-1]) * H_y[Ni-1,0:Nj,0:Nk-1] + D_Hy[Ni-1,0:Nj,0:Nk-1] * (sto_Ex[Ni-1,0:Nj,1:Nk] - sto_Ex[Ni-1,0:Nj,0:Nk-1] + sto_Ez[Ni-1,0:Nj,0:Nk-1] - B[Ni-1,0:Nj,0:Nk-1]*(sto_mu_y[Ni-1,0:Nj,0:Nk-1]+1/2*At*sto_sigma_m_y[Ni-1,0:Nj,0:Nk-1]) * m[Ni-1,0:Nj,0:Nk-1])/As
    m[Ni-1,0:Nj,Nk-1] = - E_x[Ni-1,0:Nj,Nk-1] + E_z[Ni-1,0:Nj,Nk-1]
    H_y[Ni-1,0:Nj,Nk-1] = C_Hy[Ni-1,0:Nj,Nk-1] * H_y[Ni-1,0:Nj,Nk-1] - D_Hy[Ni-1,0:Nj,Nk-1] * m[Ni-1,0:Nj,Nk-1] / As
    sto_Hy[Ni-1,0:Nj,Nk-1] = C_Hy[Ni-1,0:Nj,Nk-1] * sto_Hy[Ni-1,0:Nj,Nk-1] + A[Ni-1,0:Nj,Nk-1] * (sigma_m_y[Ni-1,0:Nj,Nk-1]*sto_mu_y[Ni-1,0:Nj,Nk-1] - mu_y[Ni-1,0:Nj,Nk-1]*sto_sigma_m_y[Ni-1,0:Nj,Nk-1]) * H_y[Ni-1,0:Nj,Nk-1] + D_Hy[Ni-1,0:Nj,Nk-1] * (-sto_Ex[Ni-1,0:Nj,Nk-1] + sto_Ez[Ni-1,0:Nj,Nk-1] - B[Ni-1,0:Nj,Nk-1]*(sto_mu_y[Ni-1,0:Nj,Nk-1]+1/2*At*sto_sigma_m_y[Ni-1,0:Nj,Nk-1]) * m[Ni-1,0:Nj,Nk-1])/As

def update_Hz(H_z,E_x,E_y,E_z,C_Hz,D_Hz,As,At,Ni,Nj,Nk,sto_Hz,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_z,sto_mu_z,sigma_m_z,mu_z):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Hz * C_Hz / At
    B = D_Hz / (As)
    #Calculo campo magnético Hz y su varianza
    m[0:Ni-1,0:Nj-1,0:Nk] = E_x[0:Ni-1,0:Nj-1,0:Nk] - E_x[0:Ni-1,1:Nj,0:Nk] + E_y[1:Ni,0:Nj-1,0:Nk] - E_y[0:Ni-1,0:Nj-1,0:Nk]
    H_z[0:Ni-1,0:Nj-1,0:Nk] = C_Hz[0:Ni-1,0:Nj-1,0:Nk] * H_z[0:Ni-1,0:Nj-1,0:Nk] - D_Hz[0:Ni-1,0:Nj-1,0:Nk] * m[0:Ni-1,0:Nj-1,0:Nk]  / As  
    sto_Hz[0:Ni-1,0:Nj-1,0:Nk] = C_Hz[0:Ni-1,0:Nj-1,0:Nk] * sto_Hz[0:Ni-1,0:Nj-1,0:Nk] + A[0:Ni-1,0:Nj-1,0:Nk] * (sigma_m_z[0:Ni-1,0:Nj-1,0:Nk]*sto_mu_z[0:Ni-1,0:Nj-1,0:Nk]-mu_z[0:Ni-1,0:Nj-1,0:Nk]*sto_sigma_m_z[0:Ni-1,0:Nj-1,0:Nk]) * H_z[0:Ni-1,0:Nj-1,0:Nk] + D_Hz[0:Ni-1,0:Nj-1,0:Nk] * (sto_Ex[0:Ni-1,0:Nj-1,0:Nk] - sto_Ex[0:Ni-1,1:Nj,0:Nk] + sto_Ey[1:Ni,0:Nj-1,0:Nk] - sto_Ey[0:Ni-1,0:Nj-1,0:Nk]- B[0:Ni-1,0:Nj-1,0:Nk] * (sto_mu_z[0:Ni-1,0:Nj-1,0:Nk]+1/2*At*sto_sigma_m_z[0:Ni-1,0:Nj-1,0:Nk]) * m[0:Ni-1,0:Nj-1,0:Nk])/As
    m[0:Ni-1,Nj-1,0:Nk] = E_x[0:Ni-1,Nj-1,0:Nk] + E_y[1:Ni,Nj-1,0:Nk] - E_y[0:Ni-1,Nj-1,0:Nk]
    H_z[0:Ni-1,Nj-1,0:Nk] = C_Hz[0:Ni-1,Nj-1,0:Nk] * H_z[0:Ni-1,Nj-1,0:Nk] - D_Hz[0:Ni-1,Nj-1,0:Nk] * m[0:Ni-1,Nj-1,0:Nk]  / As
    sto_Hz[0:Ni-1,Nj-1,0:Nk] = C_Hz[0:Ni-1,Nj-1,0:Nk] * sto_Hz[0:Ni-1,Nj-1,0:Nk] + A[0:Ni-1,Nj-1,0:Nk] * (sigma_m_z[0:Ni-1,Nj-1,0:Nk]*sto_mu_z[0:Ni-1,Nj-1,0:Nk]-mu_z[0:Ni-1,Nj-1,0:Nk]*sto_sigma_m_z[0:Ni-1,Nj-1,0:Nk]) * H_z[0:Ni-1,Nj-1,0:Nk] + D_Hz[0:Ni-1,Nj-1,0:Nk] * (sto_Ex[0:Ni-1,Nj-1,0:Nk] + sto_Ey[1:Ni,Nj-1,0:Nk] - sto_Ey[0:Ni-1,Nj-1,0:Nk] - B[0:Ni-1,Nj-1,0:Nk] * (sto_mu_z[0:Ni-1,Nj-1,0:Nk]+1/2*At*sto_sigma_m_z[0:Ni-1,Nj-1,0:Nk]) * m[0:Ni-1,Nj-1,0:Nk])/As
    m[Ni-1,0:Nj-1,0:Nk] = E_x[Ni-1,0:Nj-1,0:Nk] - E_x[Ni-1,1:Nj,0:Nk] - E_y[Ni-1,0:Nj-1,0:Nk]
    H_z[Ni-1,0:Nj-1,0:Nk] = C_Hz[Ni-1,0:Nj-1,0:Nk] * H_z[Ni-1,0:Nj-1,0:Nk] - D_Hz[Ni-1,0:Nj-1,0:Nk] * m[Ni-1,0:Nj-1,0:Nk]  / As  
    sto_Hz[Ni-1,0:Nj-1,0:Nk] = C_Hz[Ni-1,0:Nj-1,0:Nk] * sto_Hz[Ni-1,0:Nj-1,0:Nk] + A[Ni-1,0:Nj-1,0:Nk] * (sigma_m_z[Ni-1,0:Nj-1,0:Nk]*sto_mu_z[Ni-1,0:Nj-1,0:Nk]-mu_z[Ni-1,0:Nj-1,0:Nk]*sto_sigma_m_z[Ni-1,0:Nj-1,0:Nk]) * H_z[Ni-1,0:Nj-1,0:Nk] + D_Hz[Ni-1,0:Nj-1,0:Nk] * (sto_Ex[Ni-1,0:Nj-1,0:Nk] - sto_Ex[Ni-1,1:Nj,0:Nk] - sto_Ey[Ni-1,0:Nj-1,0:Nk] - B[Ni-1,0:Nj-1,0:Nk] * (sto_mu_z[Ni-1,0:Nj-1,0:Nk]+1/2*At*sto_sigma_m_z[Ni-1,0:Nj-1,0:Nk]) * m[Ni-1,0:Nj-1,0:Nk])/As
    m[Ni-1,Nj-1,0:Nk] = E_x[Ni-1,Nj-1,0:Nk] - E_y[Ni-1,Nj-1,0:Nk]
    H_z[Ni-1,Nj-1,0:Nk] = C_Hz[Ni-1,Nj-1,0:Nk] * H_z[Ni-1,Nj-1,0:Nk] - D_Hz[Ni-1,Nj-1,0:Nk] * m[Ni-1,Nj-1,0:Nk]  / As
    sto_Hz[Ni-1,Nj-1,0:Nk] = C_Hz[Ni-1,Nj-1,0:Nk] * sto_Hz[Ni-1,Nj-1,0:Nk] + A[Ni-1,Nj-1,0:Nk] * (sigma_m_z[Ni-1,Nj-1,0:Nk]*sto_mu_z[Ni-1,Nj-1,0:Nk]-mu_z[Ni-1,Nj-1,0:Nk]*sto_sigma_m_z[Ni-1,Nj-1,0:Nk]) * H_z[Ni-1,Nj-1,0:Nk] + D_Hz[Ni-1,Nj-1,0:Nk] * (sto_Ex[Ni-1,Nj-1,0:Nk] - sto_Ey[Ni-1,Nj-1,0:Nk] - B[Ni-1,Nj-1,0:Nk] * (sto_mu_z[Ni-1,Nj-1,0:Nk]+1/2*At*sto_sigma_m_z[Ni-1,Nj-1,0:Nk]) * m[Ni-1,Nj-1,0:Nk])/As
   
def update_Ex(E_x,H_x,H_y,H_z,C_Ex,D_Ex,As,At,Ni,Nj,Nk,sto_Ex,sto_Hx,sto_Hy,sto_Hz,sto_sigma_x,sto_epsilon_x,sigma_x,epsilon_x):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Ex * C_Ex / At
    B = D_Ex / (As)
    #Calculo campo eléctrico Ex y su varianza
    m[0:Ni,0,0] = H_z[0:Ni,0,0] - H_y[0:Ni,0,0]
    E_x[0:Ni,0,0] = C_Ex[0:Ni,0,0] * E_x[0:Ni,0,0] + D_Ex[0:Ni,0,0] * m[0:Ni,0,0] / As 
    sto_Ex[0:Ni,0,0] = C_Ex[0:Ni,0,0] * sto_Ex[0:Ni,0,0] + A[0:Ni,0,0] * (sigma_x[0:Ni,0,0]*sto_epsilon_x[0:Ni,0,0]-epsilon_x[0:Ni,0,0]*sto_sigma_x[0:Ni,0,0]) * E_x[0:Ni,0,0] + D_Ex[0:Ni,0,0] * (sto_Hz[0:Ni,0,0] - sto_Hy[0:Ni,0,0] - B[0:Ni,0,0] * (sto_epsilon_x[0:Ni,0,0]+1/2*At*sto_sigma_x[0:Ni,0,0]) * m[0:Ni,0,0])/As
    m[0:Ni,1:Nj,0] = H_z[0:Ni,1:Nj,0] - H_z[0:Ni,0:Nj-1,0] - H_y[0:Ni,1:Nj,0]
    E_x[0:Ni,1:Nj,0] = C_Ex[0:Ni,1:Nj,0] * E_x[0:Ni,1:Nj,0] + D_Ex[0:Ni,1:Nj,0] * m[0:Ni,1:Nj,0] /As           
    sto_Ex[0:Ni,1:Nj,0] = C_Ex[0:Ni,1:Nj,0] * sto_Ex[0:Ni,1:Nj,0] + A[0:Ni,1:Nj,0] * (sigma_x[0:Ni,1:Nj,0]*sto_epsilon_x[0:Ni,1:Nj,0]-epsilon_x[0:Ni,1:Nj,0]*sto_sigma_x[0:Ni,1:Nj,0]) * E_x[0:Ni,1:Nj,0] + D_Ex[0:Ni,1:Nj,0] * (sto_Hz[0:Ni,1:Nj,0] - sto_Hz[0:Ni,0:Nj-1,0] - sto_Hy[0:Ni,1:Nj,0] - B[0:Ni,1:Nj,0] * (sto_epsilon_x[0:Ni,1:Nj,0]+1/2*At*sto_sigma_x[0:Ni,1:Nj,0]) * m[0:Ni,1:Nj,0])/As
    m[0:Ni,0,1:Nk] = H_z[0:Ni,0,1:Nk] + H_y[0:Ni,0,0:Nk-1] - H_y[0:Ni,0,1:Nk]
    E_x[0:Ni,0,1:Nk] = C_Ex[0:Ni,0,1:Nk] * E_x[0:Ni,0,1:Nk] + D_Ex[0:Ni,0,1:Nk] * m[0:Ni,0,1:Nk] /As
    sto_Ex[0:Ni,0,1:Nk] = C_Ex[0:Ni,0,1:Nk] * sto_Ex[0:Ni,0,1:Nk] + A[0:Ni,0,1:Nk] * (sigma_x[0:Ni,0,1:Nk]*sto_epsilon_x[0:Ni,0,1:Nk]-epsilon_x[0:Ni,0,1:Nk]*sto_sigma_x[0:Ni,0,1:Nk]) * E_x[0:Ni,0,1:Nk] + D_Ex[0:Ni,0,1:Nk] * (sto_Hz[0:Ni,0,1:Nk] + sto_Hy[0:Ni,0,0:Nk-1] - sto_Hy[0:Ni,0,1:Nk] - B[0:Ni,0,1:Nk] * (sto_epsilon_x[0:Ni,0,1:Nk]+1/2*At*sto_sigma_x[0:Ni,0,1:Nk]) * m[0:Ni,0,1:Nk])/As
    m[0:Ni,1:Nj,1:Nk] = H_z[0:Ni,1:Nj,1:Nk] - H_z[0:Ni,0:Nj-1,1:Nk] + H_y[0:Ni,1:Nj,0:Nk-1] - H_y[0:Ni,1:Nj,1:Nk]
    E_x[0:Ni,1:Nj,1:Nk] = C_Ex[0:Ni,1:Nj,1:Nk] * E_x[0:Ni,1:Nj,1:Nk] + D_Ex[0:Ni,1:Nj,1:Nk] * m[0:Ni,1:Nj,1:Nk] /As
    sto_Ex[0:Ni,1:Nj,1:Nk] = C_Ex[0:Ni,1:Nj,1:Nk] * sto_Ex[0:Ni,1:Nj,1:Nk] + A[0:Ni,1:Nj,1:Nk] * (sigma_x[0:Ni,1:Nj,1:Nk]*sto_epsilon_x[0:Ni,1:Nj,1:Nk]-epsilon_x[0:Ni,1:Nj,1:Nk]*sto_sigma_x[0:Ni,1:Nj,1:Nk]) * E_x[0:Ni,1:Nj,1:Nk] + D_Ex[0:Ni,1:Nj,1:Nk]* (sto_Hz[0:Ni,1:Nj,1:Nk] - sto_Hz[0:Ni,0:Nj-1,1:Nk] + sto_Hy[0:Ni,1:Nj,0:Nk-1] - sto_Hy[0:Ni,1:Nj,1:Nk] - B[0:Ni,1:Nj,1:Nk] * (sto_epsilon_x[0:Ni,1:Nj,1:Nk]+1/2*At*sto_sigma_x[0:Ni,1:Nj,1:Nk]) * m[0:Ni,1:Nj,1:Nk])/As

def update_Ey(E_y,H_x,H_y,H_z,C_Ey,D_Ey,As,At,Ni,Nj,Nk,sto_Ey,sto_Hx,sto_Hy,sto_Hz,sto_sigma_y,sto_epsilon_y,sigma_y,epsilon_y):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Ey * C_Ey / At
    B = D_Ey / (As)
    #Calculo campo eléctrico Ey y su varianza
    m[0,0:Nj,0] = - H_z[0,0:Nj,0] + H_x[0,0:Nj,0]   
    E_y[0,0:Nj,0] = C_Ey[0,0:Nj,0] * E_y[0,0:Nj,0] + D_Ey[0,0:Nj,0] * m[0,0:Nj,0] / As
    sto_Ey[0,0:Nj,0] = C_Ey[0,0:Nj,0] * sto_Ey[0,0:Nj,0] + A[0,0:Nj,0] * (sigma_y[0,0:Nj,0]*sto_epsilon_y[0,0:Nj,0]-epsilon_y[0,0:Nj,0]*sto_sigma_y[0,0:Nj,0]) * E_y[0,0:Nj,0] + D_Ey[0,0:Nj,0] * (- sto_Hz[0,0:Nj,0] + sto_Hx[0,0:Nj,0] - B[0,0:Nj,0] * (sto_epsilon_y[0,0:Nj,0]+1/2*At*sto_sigma_y[0,0:Nj,0]) * m[0,0:Nj,0])/As
    m[1:Ni,0:Nj,0] = H_z[0:Ni-1,0:Nj,0] - H_z[1:Ni,0:Nj,0] + H_x[1:Ni,0:Nj,0]
    E_y[1:Ni,0:Nj,0] = C_Ey[1:Ni,0:Nj,0] * E_y[1:Ni,0:Nj,0] + D_Ey[1:Ni,0:Nj,0] * m[1:Ni,0:Nj,0] / As         
    sto_Ey[1:Ni,0:Nj,0] = C_Ey[1:Ni,0:Nj,0] * sto_Ey[1:Ni,0:Nj,0] + A[1:Ni,0:Nj,0] * (sigma_y[1:Ni,0:Nj,0]*sto_epsilon_y[1:Ni,0:Nj,0]-epsilon_y[1:Ni,0:Nj,0]*sto_sigma_y[1:Ni,0:Nj,0]) * E_y[1:Ni,0:Nj,0] + D_Ey[1:Ni,0:Nj,0] * (sto_Hz[0:Ni-1,0:Nj,0] - sto_Hz[1:Ni,0:Nj,0] + sto_Hx[1:Ni,0:Nj,0] - B[1:Ni,0:Nj,0] * (sto_epsilon_y[1:Ni,0:Nj,0]+1/2*At*sto_sigma_y[1:Ni,0:Nj,0]) * m[1:Ni,0:Nj,0])/As
    m[0,0:Nj,1:Nk] = - H_z[0,0:Nj,1:Nk] + H_x[0,0:Nj,1:Nk] - H_x[0,0:Nj,0:Nk-1]
    E_y[0,0:Nj,1:Nk] = C_Ey[0,0:Nj,1:Nk] * E_y[0,0:Nj,1:Nk] + D_Ey[0,0:Nj,1:Nk] * m[0,0:Nj,1:Nk] / As
    sto_Ey[0,0:Nj,1:Nk] = C_Ey[0,0:Nj,1:Nk] * sto_Ey[0,0:Nj,1:Nk] + A[0,0:Nj,1:Nk] * (sigma_y[0,0:Nj,1:Nk]*sto_epsilon_y[0,0:Nj,1:Nk]-epsilon_y[0,0:Nj,1:Nk]*sto_sigma_y[0,0:Nj,1:Nk]) * E_y[0,0:Nj,1:Nk] + D_Ey[0,0:Nj,1:Nk] * (- sto_Hz[0,0:Nj,1:Nk] + sto_Hx[0,0:Nj,1:Nk] - sto_Hx[0,0:Nj,0:Nk-1] - B[0,0:Nj,1:Nk] * (sto_epsilon_y[0,0:Nj,1:Nk]+1/2*At*sto_sigma_y[0,0:Nj,1:Nk]) * m[0,0:Nj,1:Nk])/As
    m[1:Ni,0:Nj,1:Nk] = H_z[0:Ni-1,0:Nj,1:Nk] - H_z[1:Ni,0:Nj,1:Nk] + H_x[1:Ni,0:Nj,1:Nk] - H_x[1:Ni,0:Nj,0:Nk-1]
    E_y[1:Ni,0:Nj,1:Nk] = C_Ey[1:Ni,0:Nj,1:Nk] * E_y[1:Ni,0:Nj,1:Nk] + D_Ey[1:Ni,0:Nj,1:Nk] * m[1:Ni,0:Nj,1:Nk] / As
    sto_Ey[1:Ni,0:Nj,1:Nk] = C_Ey[1:Ni,0:Nj,1:Nk] * sto_Ey[1:Ni,0:Nj,1:Nk] + A[1:Ni,0:Nj,1:Nk] * (sigma_y[1:Ni,0:Nj,1:Nk]*sto_epsilon_y[1:Ni,0:Nj,1:Nk]-epsilon_y[1:Ni,0:Nj,1:Nk]*sto_sigma_y[1:Ni,0:Nj,1:Nk]) * E_y[1:Ni,0:Nj,1:Nk] + D_Ey[1:Ni,0:Nj,1:Nk] * (sto_Hz[0:Ni-1,0:Nj,1:Nk] - sto_Hz[1:Ni,0:Nj,1:Nk] + sto_Hx[1:Ni,0:Nj,1:Nk] - sto_Hx[1:Ni,0:Nj,0:Nk-1] - B[1:Ni,0:Nj,1:Nk] * (sto_epsilon_y[1:Ni,0:Nj,1:Nk]+1/2*At*sto_sigma_y[1:Ni,0:Nj,1:Nk]) * m[1:Ni,0:Nj,1:Nk])/As

def update_Ez(E_z,H_x,H_y,H_z,C_Ez,D_Ez,As,At,Ni,Nj,Nk,sto_Ez,sto_Hx,sto_Hy,sto_Hz,sto_sigma_z,sto_epsilon_z,sigma_z,epsilon_z,R):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A = C_Ez * C_Ez  / At
    B = D_Ez / (As)
    #Calculo campo eléctrico Ez y su varianza
    m[0,0,0:Nk] = H_y[0,0,0:Nk] - H_x[0,0,0:Nk] 
    E_z[0,0,0:Nk] = C_Ez[0,0,0:Nk] * E_z[0,0,0:Nk] + D_Ez[0,0,0:Nk] * m[0,0,0:Nk] / As 
    sto_Ez[0,0,0:Nk] = C_Ez[0,0,0:Nk] * sto_Ez[0,0,0:Nk] + A[0,0,0:Nk] * (sigma_z[0,0,0:Nk]*sto_epsilon_z[0,0,0:Nk]-epsilon_z[0,0,0:Nk]*sto_sigma_z[0,0,0:Nk]) * E_z[0,0,0:Nk] + D_Ez[0,0,0:Nk] * (sto_Hy[0,0,0:Nk] - sto_Hx[0,0,0:Nk] - B[0,0,0:Nk] * (sto_epsilon_z[0,0,0:Nk]+1/2*At*sto_sigma_z[0,0,0:Nk]) * m[0,0,0:Nk])/As
    m[1:Ni,0,0:Nk] = H_y[1:Ni,0,0:Nk] - H_y[0:Ni-1,0,0:Nk] - H_x[1:Ni,0,0:Nk]
    E_z[1:Ni,0,0:Nk] = C_Ez[1:Ni,0,0:Nk] * E_z[1:Ni,0,0:Nk] + D_Ez[1:Ni,0,0:Nk] * m[1:Ni,0,0:Nk] / As            
    sto_Ez[1:Ni,0,0:Nk] = C_Ez[1:Ni,0,0:Nk] * sto_Ez[1:Ni,0,0:Nk] + A[1:Ni,0,0:Nk] * (sigma_z[1:Ni,0,0:Nk]*sto_epsilon_z[1:Ni,0,0:Nk]-epsilon_z[1:Ni,0,0:Nk]*sto_sigma_z[1:Ni,0,0:Nk]) * E_z[1:Ni,0,0:Nk] + D_Ez[1:Ni,0,0:Nk] * (sto_Hy[1:Ni,0,0:Nk] - sto_Hy[0:Ni-1,0,0:Nk] - sto_Hx[1:Ni,0,0:Nk] - B[1:Ni,0,0:Nk] * (sto_epsilon_z[1:Ni,0,0:Nk]+1/2*At*sto_sigma_z[1:Ni,0,0:Nk]) * m[1:Ni,0,0:Nk])/As
    m[0,1:Nj,0:Nk] = H_y[0,1:Nj,0:Nk] + H_x[0,0:Nj-1,0:Nk] - H_x[0,1:Nj,0:Nk]
    E_z[0,1:Nj,0:Nk] = C_Ez[0,1:Nj,0:Nk] * E_z[0,1:Nj,0:Nk] + D_Ez[0,1:Nj,0:Nk] * m[0,1:Nj,0:Nk] / As
    sto_Ez[0,1:Nj,0:Nk] = C_Ez[0,1:Nj,0:Nk] * sto_Ez[0,1:Nj,0:Nk] + A[0,1:Nj,0:Nk] * (sigma_z[0,1:Nj,0:Nk]*sto_epsilon_z[0,1:Nj,0:Nk]-epsilon_z[0,1:Nj,0:Nk]*sto_sigma_z[0,1:Nj,0:Nk]) * E_z[0,1:Nj,0:Nk] + D_Ez[0,1:Nj,0:Nk] * (sto_Hy[0,1:Nj,0:Nk] + sto_Hx[0,0:Nj-1,0:Nk] - sto_Hx[0,1:Nj,0:Nk] - B[0,1:Nj,0:Nk] * (sto_epsilon_z[0,1:Nj,0:Nk]+1/2*At*sto_sigma_z[0,1:Nj,0:Nk]) * m[0,1:Nj,0:Nk])/As
    m[1:Ni,1:Nj,0:Nk] = H_y[1:Ni,1:Nj,0:Nk] - H_y[0:Ni-1,1:Nj,0:Nk] + H_x[1:Ni,0:Nj-1,0:Nk] - H_x[1:Ni,1:Nj,0:Nk]
    E_z[1:Ni,1:Nj,0:Nk] = C_Ez[1:Ni,1:Nj,0:Nk] * E_z[1:Ni,1:Nj,0:Nk] + D_Ez[1:Ni,1:Nj,0:Nk] * m[1:Ni,1:Nj,0:Nk] / As
    sto_Ez[1:Ni,1:Nj,0:Nk] = C_Ez[1:Ni,1:Nj,0:Nk] * sto_Ez[1:Ni,1:Nj,0:Nk] + A[1:Ni,1:Nj,0:Nk] * (sigma_z[1:Ni,1:Nj,0:Nk]*sto_epsilon_z[1:Ni,1:Nj,0:Nk]-epsilon_z[1:Ni,1:Nj,0:Nk]*sto_sigma_z[1:Ni,1:Nj,0:Nk]) * E_z[1:Ni,1:Nj,0:Nk] + D_Ez[1:Ni,1:Nj,0:Nk] * (sto_Hy[1:Ni,1:Nj,0:Nk] - sto_Hy[0:Ni-1,1:Nj,0:Nk] + sto_Hx[1:Ni,0:Nj-1,0:Nk] - sto_Hx[1:Ni,1:Nj,0:Nk] - B[1:Ni,1:Nj,0:Nk] * (sto_epsilon_z[1:Ni,1:Nj,0:Nk]+1/2*At*sto_sigma_z[1:Ni,1:Nj,0:Nk]) * m[1:Ni,1:Nj,0:Nk])/As
    
def update_Ez_thin_wire(E_z,H_x,H_y,H_z,C_Ez,D_Ez,As,At,c,Ni,Nj,Nk,sto_Ez,sto_Hx,sto_Hy,sto_Hz,sto_sigma_z,sto_epsilon_z,sigma_z,epsilon_z,I,Q,R,L,sto_Q,sto_I,sto_R,sto_L,l,r):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A = np.zeros((Ni,Nj,Nk))
    B = np.zeros((Ni,Nj,Nk))
    A1 = np.zeros((Ni,Nj,Nk))
    A2 = np.zeros((Ni,Nj,Nk))
    A3 = np.zeros((Ni,Nj,Nk))
    B1 = np.zeros((Ni,Nj,Nk))
    B2 = np.zeros((Ni,Nj,Nk))
    B3 = np.zeros((Ni,Nj,Nk))
    sigma = np.zeros((Ni,Nj,Nk))
    C_Ez_tw = np.copy(C_Ez)
    D_Ez_tw = np.copy(D_Ez)
    k = np.arange(0,7)
    sigma[30,27+k,30] = (1/R[30,27+k,30])*(1/(math.pi*(r)**2))
    C_Ez_tw[30,27+k,30] = (epsilon_z[30,27+k,30] - 0.5*sigma[30,27+k,30]*At)/(epsilon_z[30,27+k,30] + 0.5*sigma[30,27+k,30]*At)
    D_Ez_tw[30,27+k,30] = At/(epsilon_z[30,27+k,30] + 0.5*sigma[30,27+k,30]*At)
    A = C_Ez_tw * C_Ez_tw / At
    B = D_Ez_tw / (As)
    A1[30,27+k,30] = (L[30,27+k,30] - 1/2*R[30,27+k,30]*At)/(L[30,27+k,30] + 1/2*R[30,27+k,30]*At)
    A2[30,27+k,30] = (At*L[30,27+k,30]*c**2)/(As*(L[30,27+k,30] + 1/2*At*R[30,27+k,30]))
    A3[30,27+k,30] = At/(L[30,27+k,30] + 1/2*At*R[30,27+k,30])
    B1[30,27+k,30] = sto_L[30,27+k,30]*R[30,27+k,30]*At/(L[30,27+k,30]+1/2*R[30,27+k,30]*At)**2 - sto_R[30,27+k,30]*At*L[30,27+k,30]/(L[30,27+k,30]+1/2*R[30,27+k,30]*At)**2
    B2[30,27+k,30] = sto_L[30,27+k,30]*c**2*At*As*(1/2*At*R[30,27+k,30])/(As*(L[30,27+k,30]+1/2*At*R[30,27+k,30]))**2 - sto_R[30,27+k,30]*1/2*As*At**2*c**2*L[30,27+k,30]/(As*(L[30,27+k,30]+1/2*At*R[30,27+k,30]))**2
    B3[30,27+k,30] = -sto_L[30,27+k,30]*At/(L[30,27+k,30]+1/2*At*R[30,27+k,30])**2 - 1/2*sto_R[30,27+k,30]*At**2/(L[30,27+k,30]+1/2*At*R[30,27+k,30])**2
    #Calculo campo eléctrico Ez, la thin wire y sus varianzas
    #Q,I 
    Q[30,27+k,30] = Q[30,27+k,30] - (At/As) * (I[30,28+k,30] - I[30,27+k,30])
    I[30,27+k,30] = A1[30,27+k,30]*I[30,27+k,30] - A2[30,27+k,30]*(Q[30,27+k,30] - Q[30,26+k,30]) + A3[30,27+k,30]*E_z[30,27+k,30]
    #Varianza de Q,I
    sto_Q[30,27+k,30] = sto_Q[30,27+k,30] - (At/As) * (sto_I[30,28+k,30] - sto_I[30,27+k,30])
    sto_I[30,27+k,30] = A1[30,27+k,30]*sto_I[30,27+k,30]- A2[30,27+k,30]*(sto_Q[30,27+k,30] - sto_Q[30,26+k,30]) + B1[30,27+k,30]*I[30,27+k,30] - B2[30,27+k,30]*(Q[30,27+k,30] - Q[30,26+k,30]) + A3[30,27+k,30]*sto_Ez[30,27+k,30] + B3[30,27+k,30]*E_z[30,27+k,30]
    #E_z y su varianza
    m[0,0,0:Nk] = H_y[0,0,0:Nk] - H_x[0,0,0:Nk] 
    E_z[0,0,0:Nk] = C_Ez_tw[0,0,0:Nk] * E_z[0,0,0:Nk] + D_Ez_tw[0,0,0:Nk] * m[0,0,0:Nk] / As + D_Ez_tw[0,0,0:Nk] * I[0,0,0:Nk]/(As**2)
    sto_Ez[0,0,0:Nk] = C_Ez_tw[0,0,0:Nk] * sto_Ez[0,0,0:Nk] + A[0,0,0:Nk] * (sigma_z[0,0,0:Nk]*sto_epsilon_z[0,0,0:Nk]-epsilon_z[0,0,0:Nk]*sto_sigma_z[0,0,0:Nk]) * E_z[0,0,0:Nk] + D_Ez_tw[0,0,0:Nk] * (sto_I[0,0,0:Nk]/As + sto_Hy[0,0,0:Nk] - sto_Hx[0,0,0:Nk] - B[0,0,0:Nk] * (sto_epsilon_z[0,0,0:Nk]+1/2*At*sto_sigma_z[0,0,0:Nk]) * m[0,0,0:Nk])/As
    m[1:Ni,0,0:Nk] = H_y[1:Ni,0,0:Nk] - H_y[0:Ni-1,0,0:Nk] - H_x[1:Ni,0,0:Nk]
    E_z[1:Ni,0,0:Nk] = C_Ez_tw[1:Ni,0,0:Nk] * E_z[1:Ni,0,0:Nk] + D_Ez_tw[1:Ni,0,0:Nk] * m[1:Ni,0,0:Nk] / As + D_Ez_tw[1:Ni,0,0:Nk] * I[1:Ni,0,0:Nk]/(As**2)            
    sto_Ez[1:Ni,0,0:Nk] = C_Ez_tw[1:Ni,0,0:Nk] * sto_Ez[1:Ni,0,0:Nk] + A[1:Ni,0,0:Nk] * (sigma_z[1:Ni,0,0:Nk]*sto_epsilon_z[1:Ni,0,0:Nk]-epsilon_z[1:Ni,0,0:Nk]*sto_sigma_z[1:Ni,0,0:Nk]) * E_z[1:Ni,0,0:Nk] + D_Ez_tw[1:Ni,0,0:Nk] * (sto_I[0,0,0:Nk]/As + sto_Hy[1:Ni,0,0:Nk] - sto_Hy[0:Ni-1,0,0:Nk] - sto_Hx[1:Ni,0,0:Nk] - B[1:Ni,0,0:Nk] * (sto_epsilon_z[1:Ni,0,0:Nk]+1/2*At*sto_sigma_z[1:Ni,0,0:Nk]) * m[1:Ni,0,0:Nk])/As
    m[0,1:Nj,0:Nk] = H_y[0,1:Nj,0:Nk] + H_x[0,0:Nj-1,0:Nk] - H_x[0,1:Nj,0:Nk]
    E_z[0,1:Nj,0:Nk] = C_Ez_tw[0,1:Nj,0:Nk] * E_z[0,1:Nj,0:Nk] + D_Ez_tw[0,1:Nj,0:Nk] * m[0,1:Nj,0:Nk] / As + D_Ez_tw[0,1:Nj,0:Nk] * I[0,1:Nj,0:Nk]/(As**2)
    sto_Ez[0,1:Nj,0:Nk] = C_Ez_tw[0,1:Nj,0:Nk] * sto_Ez[0,1:Nj,0:Nk] + A[0,1:Nj,0:Nk] * (sigma_z[0,1:Nj,0:Nk]*sto_epsilon_z[0,1:Nj,0:Nk]-epsilon_z[0,1:Nj,0:Nk]*sto_sigma_z[0,1:Nj,0:Nk]) * E_z[0,1:Nj,0:Nk] + D_Ez_tw[0,1:Nj,0:Nk] * (sto_I[0,0,0:Nk]/As + sto_Hy[0,1:Nj,0:Nk] + sto_Hx[0,0:Nj-1,0:Nk] - sto_Hx[0,1:Nj,0:Nk] - B[0,1:Nj,0:Nk] * (sto_epsilon_z[0,1:Nj,0:Nk]+1/2*At*sto_sigma_z[0,1:Nj,0:Nk]) * m[0,1:Nj,0:Nk])/As
    m[1:Ni,1:Nj,0:Nk] = H_y[1:Ni,1:Nj,0:Nk] - H_y[0:Ni-1,1:Nj,0:Nk] + H_x[1:Ni,0:Nj-1,0:Nk] - H_x[1:Ni,1:Nj,0:Nk]
    E_z[1:Ni,1:Nj,0:Nk] = C_Ez_tw[1:Ni,1:Nj,0:Nk] * E_z[1:Ni,1:Nj,0:Nk] + D_Ez_tw[1:Ni,1:Nj,0:Nk] * m[1:Ni,1:Nj,0:Nk] / As + D_Ez_tw[1:Ni,1:Nj,0:Nk] * I[1:Ni,1:Nj,0:Nk]/(As**2) 
    sto_Ez[1:Ni,1:Nj,0:Nk] = C_Ez_tw[1:Ni,1:Nj,0:Nk] * sto_Ez[1:Ni,1:Nj,0:Nk] + A[1:Ni,1:Nj,0:Nk] * (sigma_z[1:Ni,1:Nj,0:Nk]*sto_epsilon_z[1:Ni,1:Nj,0:Nk]-epsilon_z[1:Ni,1:Nj,0:Nk]*sto_sigma_z[1:Ni,1:Nj,0:Nk]) * E_z[1:Ni,1:Nj,0:Nk] + D_Ez_tw[1:Ni,1:Nj,0:Nk] * (sto_I[0,0,0:Nk]/As + sto_Hy[1:Ni,1:Nj,0:Nk] - sto_Hy[0:Ni-1,1:Nj,0:Nk] + sto_Hx[1:Ni,0:Nj-1,0:Nk] - sto_Hx[1:Ni,1:Nj,0:Nk] - B[1:Ni,1:Nj,0:Nk] * (sto_epsilon_z[1:Ni,1:Nj,0:Nk]+1/2*At*sto_sigma_z[1:Ni,1:Nj,0:Nk]) * m[1:Ni,1:Nj,0:Nk])/As

#CALCULOS
times=[]
probe1=[]
probe2=[]

for n in range(1,lim_step+1):
    t = n*At
#    if t < 50*As/c:
    gauss = math.exp(-(t-to)**2/p**2)
#    gauss = math.sqrt(2*math.exp(1))*(t-to)*math.exp(-(t-to)**2/p**2)/p
    #Actualización campo magnético H con condiciones (PEC) y su varianza
    update_Hx(H_x,E_x,E_y,E_z,C_Hx,D_Hx,As,At,Ni,Nj,Nk,sto_Hx,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_x,sto_mu_x,sigma_m_x,mu_x)
    update_Hy(H_y,E_x,E_y,E_z,C_Hy,D_Hy,As,At,Ni,Nj,Nk,sto_Hy,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_y,sto_mu_y,sigma_m_y,mu_y)
    update_Hz(H_z,E_x,E_y,E_z,C_Hz,D_Hz,As,At,Ni,Nj,Nk,sto_Hz,sto_Ex,sto_Ey,sto_Ez,sto_sigma_m_z,sto_mu_z,sigma_m_z,mu_z)
    #Actualización campo eléctrico E con condiciones (PEC) y su varianza
    update_Ex(E_x,H_x,H_y,H_z,C_Ex,D_Ex,As,At,Ni,Nj,Nk,sto_Ex,sto_Hx,sto_Hy,sto_Hz,sto_sigma_x,sto_epsilon_x,sigma_x,epsilon_x)
    update_Ey(E_y,H_x,H_y,H_z,C_Ey,D_Ey,As,At,Ni,Nj,Nk,sto_Ey,sto_Hx,sto_Hy,sto_Hz,sto_sigma_y,sto_epsilon_y,sigma_y,epsilon_y)
    update_Ez_thin_wire(E_z,H_x,H_y,H_z,C_Ez,D_Ez,As,At,c,Ni,Nj,Nk,sto_Ez,sto_Hx,sto_Hy,sto_Hz,sto_sigma_z,sto_epsilon_z,sigma_z,epsilon_z,I,Q,R,L,sto_Q,sto_I,sto_R,sto_L,l,r)
    #Pulso gaussiano
    E_z[30,30,20] = E_z[30,30,20] + D_Ez[30,30,20] * gauss          
    times.append(t)
    probe1.append(I[30,30,30])
    probe2.append(abs(sto_I[30,30,30]))
    print(n,t,E_z[15,15,15],H_x[15,15,15])
         
#plt.plot(times, probe1, 'x--', label= 'Media perfil gaussiano FDTD 1%', color = 'green', markersize=3)
#plt.plot(times, probe1, label= 'Media perfil gaussiano FDTD 10%', color = 'green')
plt.plot(times, probe2, label='Desviación estándar S-FDTD 10%', color = 'blue')
#plt.plot(times, probe2, 's--', label='Desviación estándar perfil gaussiano S-FDTD 1%', color = 'red', markersize=3)

plt.xlabel('Tiempo [s]')
plt.ylabel('Intensidad [A]')
plt.title('Intensidad en el punto [30,30,30]')
plt.legend()
plt.grid(True)
plt.show()


print('FIN')
