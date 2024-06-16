#Programa FDTD_3D fuente suave, sin ciclo for, PEC, Monte-Carlo
#7 segmentos de cable
#gaussiana
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import os
# Directorio donde quieres guardar los archivos
directorio =  r'C:\Users\anjer\OneDrive\Documentos\Universidad de Granada\Año 7\Cuatrimestre 2\Python\Datos5'
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
lim_step = 500
#lim_step = 82
fmax = c/(PPW*As)
p = 1/(math.pi*fmax)
to = 5*p
t = 0
#Constantes algebraicas
C_Hx = (mu_x - 0.5*sigma_m_x*At)/(mu_x + 0.5*sigma_m_x*At)
D_Hx = At/(mu_x + 0.5*sigma_m_x*At)
C_Hy = (mu_y - 0.5*sigma_m_y*At)/(mu_y + 0.5*sigma_m_y*At)
D_Hy = At/(mu_y + 0.5*sigma_m_y*At)
C_Hz = (mu_z - 0.5*sigma_m_z*At)/(mu_z + 0.5*sigma_m_z*At)
D_Hz = At/(mu_z + 0.5*sigma_m_z*At)
C_Ex = (epsilon_x - 0.5*sigma_x*At)/(epsilon_x + 0.5*sigma_x*At)
D_Ex = At/(epsilon_x + 0.5*sigma_x*At)
C_Ey = (epsilon_y - 0.5*sigma_y*At)/(epsilon_y + 0.5*sigma_y*At)
D_Ey = At/(epsilon_y + 0.5*sigma_y*At)
C_Ez = (epsilon_z - 0.5*sigma_z*At)/(epsilon_z + 0.5*sigma_z*At)
D_Ez = At/(epsilon_z + 0.5*sigma_z*At)
L = np.zeros((Ni,Nj,Nk))
d = 2 * 2*10**-5
l = As
r = d/2
for k in range(0,7):
#    L[30,27+k,30] = 7.462*10**-7
    L[30,27+k,30] = (7.462*10**-7)/As

#Funciones
def update_Hx(H_x,E_x,E_y,E_z,C_Hx,D_Hx,As,At,Ni,Nj,Nk):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk))
    #Calculo campo magnético Hx y su varianza
    m[0:Ni,0:Nj-1,0:Nk-1] = E_y[0:Ni,0:Nj-1,0:Nk-1] - E_y[0:Ni,0:Nj-1,1:Nk] + E_z[0:Ni,1:Nj,0:Nk-1] - E_z[0:Ni,0:Nj-1,0:Nk-1]  
    H_x[0:Ni,0:Nj-1,0:Nk-1] = C_Hx[0:Ni,0:Nj-1,0:Nk-1] * H_x[0:Ni,0:Nj-1,0:Nk-1] - D_Hx[0:Ni,0:Nj-1,0:Nk-1] * m[0:Ni,0:Nj-1,0:Nk-1] / As
    m[0:Ni,0:Nj-1,Nk-1] = E_y[0:Ni,0:Nj-1,Nk-1] + E_z[0:Ni,1:Nj,Nk-1] - E_z[0:Ni,0:Nj-1,Nk-1]
    H_x[0:Ni,0:Nj-1,Nk-1] = C_Hx[0:Ni,0:Nj-1,Nk-1] * H_x[0:Ni,0:Nj-1,Nk-1] - D_Hx[0:Ni,0:Nj-1,Nk-1] * m[0:Ni,0:Nj-1,Nk-1] / As
    m[0:Ni,Nj-1,0:Nk-1] = E_y[0:Ni,Nj-1,0:Nk-1] - E_y[0:Ni,Nj-1,1:Nk] - E_z[0:Ni,Nj-1,0:Nk-1]  
    H_x[0:Ni,Nj-1,0:Nk-1] = C_Hx[0:Ni,Nj-1,0:Nk-1] * H_x[0:Ni,Nj-1,0:Nk-1] - D_Hx[0:Ni,Nj-1,0:Nk-1] * m[0:Ni,Nj-1,0:Nk-1] / As   
    m[0:Ni,Nj-1,Nk-1] = E_y[0:Ni,Nj-1,Nk-1] - E_z[0:Ni,Nj-1,Nk-1]
    H_x[0:Ni,Nj-1,Nk-1] = C_Hx[0:Ni,Nj-1,Nk-1] * H_x[0:Ni,Nj-1,Nk-1] - D_Hx[0:Ni,Nj-1,Nk-1] * m[0:Ni,Nj-1,Nk-1] / As

def update_Hy(H_y,E_x,E_y,E_z,C_Hy,D_Hy,As,At,Ni,Nj,Nk):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    #Calculo campo magnético Hy y su varianza
    m[0:Ni-1,0:Nj,0:Nk-1] = E_x[0:Ni-1,0:Nj,1:Nk] - E_x[0:Ni-1,0:Nj,0:Nk-1] + E_z[0:Ni-1,0:Nj,0:Nk-1] - E_z[1:Ni,0:Nj,0:Nk-1]
    H_y[0:Ni-1,0:Nj,0:Nk-1] = C_Hy[0:Ni-1,0:Nj,0:Nk-1] * H_y[0:Ni-1,0:Nj,0:Nk-1] - D_Hy[0:Ni-1,0:Nj,0:Nk-1] * m[0:Ni-1,0:Nj,0:Nk-1] / As
    m[0:Ni-1,0:Nj,Nk-1] = - E_x[0:Ni-1,0:Nj,Nk-1] + E_z[0:Ni-1,0:Nj,Nk-1] - E_z[1:Ni,0:Nj,Nk-1]
    H_y[0:Ni-1,0:Nj,Nk-1] = C_Hy[0:Ni-1,0:Nj,Nk-1] * H_y[0:Ni-1,0:Nj,Nk-1] - D_Hy[0:Ni-1,0:Nj,Nk-1] * m[0:Ni-1,0:Nj,Nk-1] / As
    m[Ni-1,0:Nj,0:Nk-1] = E_x[Ni-1,0:Nj,1:Nk] - E_x[Ni-1,0:Nj,0:Nk-1] + E_z[Ni-1,0:Nj,0:Nk-1]
    H_y[Ni-1,0:Nj,0:Nk-1] = C_Hy[Ni-1,0:Nj,0:Nk-1] * H_y[Ni-1,0:Nj,0:Nk-1] - D_Hy[Ni-1,0:Nj,0:Nk-1] * m[Ni-1,0:Nj,0:Nk-1] / As
    m[Ni-1,0:Nj,Nk-1] = - E_x[Ni-1,0:Nj,Nk-1] + E_z[Ni-1,0:Nj,Nk-1]
    H_y[Ni-1,0:Nj,Nk-1] = C_Hy[Ni-1,0:Nj,Nk-1] * H_y[Ni-1,0:Nj,Nk-1] - D_Hy[Ni-1,0:Nj,Nk-1] * m[Ni-1,0:Nj,Nk-1] / As

def update_Hz(H_z,E_x,E_y,E_z,C_Hz,D_Hz,As,At,Ni,Nj,Nk):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    #Calculo campo magnético Hz y su varianza
    m[0:Ni-1,0:Nj-1,0:Nk] = E_x[0:Ni-1,0:Nj-1,0:Nk] - E_x[0:Ni-1,1:Nj,0:Nk] + E_y[1:Ni,0:Nj-1,0:Nk] - E_y[0:Ni-1,0:Nj-1,0:Nk]
    H_z[0:Ni-1,0:Nj-1,0:Nk] = C_Hz[0:Ni-1,0:Nj-1,0:Nk] * H_z[0:Ni-1,0:Nj-1,0:Nk] - D_Hz[0:Ni-1,0:Nj-1,0:Nk] * m[0:Ni-1,0:Nj-1,0:Nk]  / As  
    m[0:Ni-1,Nj-1,0:Nk] = E_x[0:Ni-1,Nj-1,0:Nk] + E_y[1:Ni,Nj-1,0:Nk] - E_y[0:Ni-1,Nj-1,0:Nk]
    H_z[0:Ni-1,Nj-1,0:Nk] = C_Hz[0:Ni-1,Nj-1,0:Nk] * H_z[0:Ni-1,Nj-1,0:Nk] - D_Hz[0:Ni-1,Nj-1,0:Nk] * m[0:Ni-1,Nj-1,0:Nk]  / As
    m[Ni-1,0:Nj-1,0:Nk] = E_x[Ni-1,0:Nj-1,0:Nk] - E_x[Ni-1,1:Nj,0:Nk] - E_y[Ni-1,0:Nj-1,0:Nk]
    H_z[Ni-1,0:Nj-1,0:Nk] = C_Hz[Ni-1,0:Nj-1,0:Nk] * H_z[Ni-1,0:Nj-1,0:Nk] - D_Hz[Ni-1,0:Nj-1,0:Nk] * m[Ni-1,0:Nj-1,0:Nk]  / As  
    m[Ni-1,Nj-1,0:Nk] = E_x[Ni-1,Nj-1,0:Nk] - E_y[Ni-1,Nj-1,0:Nk]
    H_z[Ni-1,Nj-1,0:Nk] = C_Hz[Ni-1,Nj-1,0:Nk] * H_z[Ni-1,Nj-1,0:Nk] - D_Hz[Ni-1,Nj-1,0:Nk] * m[Ni-1,Nj-1,0:Nk]  / As
   
def update_Ex(E_x,H_x,H_y,H_z,C_Ex,D_Ex,As,At,Ni,Nj,Nk):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    #Calculo campo eléctrico Ex y su varianza
    m[0:Ni,0,0] = H_z[0:Ni,0,0] - H_y[0:Ni,0,0]
    E_x[0:Ni,0,0] = C_Ex[0:Ni,0,0] * E_x[0:Ni,0,0] + D_Ex[0:Ni,0,0] * m[0:Ni,0,0] / As 
    m[0:Ni,1:Nj,0] = H_z[0:Ni,1:Nj,0] - H_z[0:Ni,0:Nj-1,0] - H_y[0:Ni,1:Nj,0]
    E_x[0:Ni,1:Nj,0] = C_Ex[0:Ni,1:Nj,0] * E_x[0:Ni,1:Nj,0] + D_Ex[0:Ni,1:Nj,0] * m[0:Ni,1:Nj,0] /As           
    m[0:Ni,0,1:Nk] = H_z[0:Ni,0,1:Nk] + H_y[0:Ni,0,0:Nk-1] - H_y[0:Ni,0,1:Nk]
    E_x[0:Ni,0,1:Nk] = C_Ex[0:Ni,0,1:Nk] * E_x[0:Ni,0,1:Nk] + D_Ex[0:Ni,0,1:Nk] * m[0:Ni,0,1:Nk] /As
    m[0:Ni,1:Nj,1:Nk] = H_z[0:Ni,1:Nj,1:Nk] - H_z[0:Ni,0:Nj-1,1:Nk] + H_y[0:Ni,1:Nj,0:Nk-1] - H_y[0:Ni,1:Nj,1:Nk]
    E_x[0:Ni,1:Nj,1:Nk] = C_Ex[0:Ni,1:Nj,1:Nk] * E_x[0:Ni,1:Nj,1:Nk] + D_Ex[0:Ni,1:Nj,1:Nk] * m[0:Ni,1:Nj,1:Nk] /As

def update_Ey(E_y,H_x,H_y,H_z,C_Ey,D_Ey,As,At,Ni,Nj,Nk):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    #Calculo campo eléctrico Ey y su varianza
    m[0,0:Nj,0] = - H_z[0,0:Nj,0] + H_x[0,0:Nj,0]   
    E_y[0,0:Nj,0] = C_Ey[0,0:Nj,0] * E_y[0,0:Nj,0] + D_Ey[0,0:Nj,0] * m[0,0:Nj,0] / As
    m[1:Ni,0:Nj,0] = H_z[0:Ni-1,0:Nj,0] - H_z[1:Ni,0:Nj,0] + H_x[1:Ni,0:Nj,0]
    E_y[1:Ni,0:Nj,0] = C_Ey[1:Ni,0:Nj,0] * E_y[1:Ni,0:Nj,0] + D_Ey[1:Ni,0:Nj,0] * m[1:Ni,0:Nj,0] / As         
    m[0,0:Nj,1:Nk] = - H_z[0,0:Nj,1:Nk] + H_x[0,0:Nj,1:Nk] - H_x[0,0:Nj,0:Nk-1]
    E_y[0,0:Nj,1:Nk] = C_Ey[0,0:Nj,1:Nk] * E_y[0,0:Nj,1:Nk] + D_Ey[0,0:Nj,1:Nk] * m[0,0:Nj,1:Nk] / As
    m[1:Ni,0:Nj,1:Nk] = H_z[0:Ni-1,0:Nj,1:Nk] - H_z[1:Ni,0:Nj,1:Nk] + H_x[1:Ni,0:Nj,1:Nk] - H_x[1:Ni,0:Nj,0:Nk-1]
    E_y[1:Ni,0:Nj,1:Nk] = C_Ey[1:Ni,0:Nj,1:Nk] * E_y[1:Ni,0:Nj,1:Nk] + D_Ey[1:Ni,0:Nj,1:Nk] * m[1:Ni,0:Nj,1:Nk] / As
   
def update_Ez_thin_wire(E_z,H_x,H_y,H_z,C_Ez,D_Ez,As,At,c,Ni,Nj,Nk,sigma_z,epsilon_z,I,Q,R,L,l,r):
    #Declaración de variables
    m = np.zeros((Ni,Nj,Nk)) 
    A1 = np.zeros((Ni,Nj,Nk))
    A2 = np.zeros((Ni,Nj,Nk))
    A3 = np.zeros((Ni,Nj,Nk))
    sigma = np.zeros((Ni,Nj,Nk))
    C_Ez_tw = np.copy(C_Ez)
    D_Ez_tw = np.copy(D_Ez)
    k = np.arange(0,7)
    sigma[30,27+k,30] = (1/R[30,27+k,30])*(1/(math.pi*(r)**2))
    C_Ez_tw[30,27+k,30] = (epsilon_z[30,27+k,30] - 0.5*sigma[30,27+k,30]*At)/(epsilon_z[30,27+k,30] + 0.5*sigma[30,27+k,30]*At)
    D_Ez_tw[30,27+k,30] = At/(epsilon_z[30,27+k,30] + 0.5*sigma[30,27+k,30]*At)
    A1[30,27+k,30] = (L[30,27+k,30] - 1/2*R[30,27+k,30]*At)/(L[30,27+k,30] + 1/2*R[30,27+k,30]*At)
    A2[30,27+k,30] = (At*L[30,27+k,30]*c**2)/(As*(L[30,27+k,30] + 1/2*At*R[30,27+k,30]))
    A3[30,27+k,30] = At/(L[30,27+k,30] + 1/2*At*R[30,27+k,30])
    #Calculo campo eléctrico Ez, la thin wires
    #Q,I 
    Q[30,27+k,30] = Q[30,27+k,30] - (At/As) * (I[30,28+k,30] - I[30,27+k,30])
    I[30,27+k,30] = A1[30,27+k,30]*I[30,27+k,30] - A2[30,27+k,30]*(Q[30,27+k,30] - Q[30,26+k,30]) + A3[30,27+k,30]*E_z[30,27+k,30]   
    #E_z
    m[0,0,0:Nk] = H_y[0,0,0:Nk] - H_x[0,0,0:Nk] 
    E_z[0,0,0:Nk] = C_Ez_tw[0,0,0:Nk] * E_z[0,0,0:Nk] + D_Ez_tw[0,0,0:Nk] * m[0,0,0:Nk] / As + D_Ez_tw[0,0,0:Nk] * I[0,0,0:Nk]/(As**2)
    m[1:Ni,0,0:Nk] = H_y[1:Ni,0,0:Nk] - H_y[0:Ni-1,0,0:Nk] - H_x[1:Ni,0,0:Nk]
    E_z[1:Ni,0,0:Nk] = C_Ez_tw[1:Ni,0,0:Nk] * E_z[1:Ni,0,0:Nk] + D_Ez_tw[1:Ni,0,0:Nk] * m[1:Ni,0,0:Nk] / As + D_Ez_tw[1:Ni,0,0:Nk] * I[1:Ni,0,0:Nk]/(As**2)            
    m[0,1:Nj,0:Nk] = H_y[0,1:Nj,0:Nk] + H_x[0,0:Nj-1,0:Nk] - H_x[0,1:Nj,0:Nk]
    E_z[0,1:Nj,0:Nk] = C_Ez_tw[0,1:Nj,0:Nk] * E_z[0,1:Nj,0:Nk] + D_Ez_tw[0,1:Nj,0:Nk] * m[0,1:Nj,0:Nk] / As + D_Ez_tw[0,1:Nj,0:Nk] * I[0,1:Nj,0:Nk]/(As**2)
    m[1:Ni,1:Nj,0:Nk] = H_y[1:Ni,1:Nj,0:Nk] - H_y[0:Ni-1,1:Nj,0:Nk] + H_x[1:Ni,0:Nj-1,0:Nk] - H_x[1:Ni,1:Nj,0:Nk]
    E_z[1:Ni,1:Nj,0:Nk] = C_Ez_tw[1:Ni,1:Nj,0:Nk] * E_z[1:Ni,1:Nj,0:Nk] + D_Ez_tw[1:Ni,1:Nj,0:Nk] * m[1:Ni,1:Nj,0:Nk] / As + D_Ez_tw[1:Ni,1:Nj,0:Nk] * I[1:Ni,1:Nj,0:Nk]/(As**2)  

for o in range(1,101):  
    nombre_archivo = f"montecarlo_g_1_{o}.txt"
    archivo = open(nombre_archivo, 'w')
    #CALCULOS
#    times=[]
#    probe1=[]
    #Parametros que necesitan reinicializarse en cada iteracion de
    #la simulacion montecarlo
    #InicializaciÃ³n a 0 de los campos
    H_x = np.zeros((Ni,Nj,Nk))          
    H_y = np.zeros((Ni,Nj,Nk))
    H_z = np.zeros((Ni,Nj,Nk))
    E_x = np.zeros((Ni,Nj,Nk))
    E_y = np.zeros((Ni,Nj,Nk))
    E_z = np.zeros((Ni,Nj,Nk))
    x = np.zeros(7)
    #Carga,intensidad y resistencia por unidad de longitud
    I = np.zeros((Ni,Nj,Nk))
    Q = np.zeros((Ni,Nj,Nk))
    R = np.zeros((Ni,Nj,Nk))
    #Valores de las resistencias y sus desviaciones
    #1%
    Rvar = np.array([0.999922979125512,1.005100543660772,0.9941299871337071,1.000778728593687,1.0034004400561682,1.0017477981385732,0.991150602110106])
    desvia = np.array([0.00999922979125512,0.01005100543660772,0.009941299871337071,0.010007787285936872,0.010034004400561682,0.010017477981385732,0.00991150602110106])
    #10%
#    Rvar = np.array([1.0480821440721129,0.9247264298969866,1.0283193018954377,0.953509278393018,0.9577257685371066,0.8946043410020965,0.9735852636761192]) 
#    desvia = np.array([0.10480821440721129,0.09247264298969866,0.10283193018954379,0.09535092783930181,0.09577257685371066,0.08946043410020965,0.09735852636761193])
    #Actualizacion de los valores de la resistencia
    for k in range(0,7):
        x[k] = Rvar[k]-3*desvia[k]+o*6*desvia[k]/100
    for k in range(0,7):
        R[30,27+k,30] =(Rvar[k]/As)*math.exp(-((x[k]-Rvar[k])**2)/(2*desvia[k]**2))
#        R[30,27+k,30] =(Rvar[k]/As)*(1/(math.sqrt(2*math.pi)*desvia[k]))*math.exp(-((x[k]-Rvar[k])**2)/(2*desvia[k]**2))
    t = 0
    #Ciclo principal FDTD
    for n in range(1,lim_step+1):
        t = n*At
        gauss = math.exp(-(t-to)**2/p**2)
#        dgauss = math.sqrt(2*math.exp(1))*(t-to)*math.exp(-(t-to)**2/p**2)/p
        #Actualización campo magnético H con condiciones (PEC) y su varianza
        update_Hx(H_x,E_x,E_y,E_z,C_Hx,D_Hx,As,At,Ni,Nj,Nk)
        update_Hy(H_y,E_x,E_y,E_z,C_Hy,D_Hy,As,At,Ni,Nj,Nk)
        update_Hz(H_z,E_x,E_y,E_z,C_Hz,D_Hz,As,At,Ni,Nj,Nk)
        #Actualización campo eléctrico E con condiciones (PEC) y su varianza
        update_Ex(E_x,H_x,H_y,H_z,C_Ex,D_Ex,As,At,Ni,Nj,Nk)
        update_Ey(E_y,H_x,H_y,H_z,C_Ey,D_Ey,As,At,Ni,Nj,Nk)
        update_Ez_thin_wire(E_z,H_x,H_y,H_z,C_Ez,D_Ez,As,At,c,Ni,Nj,Nk,sigma_z,epsilon_z,I,Q,R,L,l,r)
        #Pulso gaussiano
        E_z[30,30,20] = E_z[30,30,20] + D_Ez[30,30,20] * gauss          
#        times.append(t)
#        probe1.append(I[30,30,30])
        archivo.write(f"{n} {t} {I[30,30,30]}\n")
    archivo.close()
    print(f'Simulacion {o} completada')
    
print('FIN')