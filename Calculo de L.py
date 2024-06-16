#Programa para calcular la autoinductancia del método de Holland et al
#mediante metodos numericos
import numpy as np
import math
from scipy.integrate import simpson

#Inicializacion de variables
mu = 4*math.pi*10**-7
As = 10**-3
#a = 2*10**-5
a = 2*10**-6

#Funciones
#Numerador
def f(x,y):
    return np.log(np.sqrt(x**2+y**2)/a)*np.sqrt(x**2+y**2)                                                
#Denominador
def g(x,y):
    return np.sqrt(x**2+y**2)       

#Puntos de integracion
x = np.linspace(a,As,100)      
y = np.linspace(a,As,100)   
#u = np.linspace(0,As,100)   #desde 0   
#v = np.linspace(0,As,100)   #desde 0  
  
#Cuadricula de puntos
X, Y = np.meshgrid(x,y)  
#U, V = np.meshgrid(u,v)     #desde 0 

#Evaluar los puntos de las cuadriculas
f_valores = f(X, Y)
g_valores = g(X, Y)         #desde 0  

#Integrales mediante el metodo de simpson
integral_numerador = simpson(simpson(f_valores,x),y) 
integral_denominador = simpson(simpson(g_valores,x),y)  #desde 0  

#Calculo de L   
L = (mu/(2*math.pi))*(integral_numerador/integral_denominador)

print("L = ",L)        