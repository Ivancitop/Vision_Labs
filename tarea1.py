#!/usr/bin/env python3
#shebang
#imports necesarios para desarrollar operaciones
import cv2 as cv # opencv-python para procesamiento de imagen
import numpy as np # manejar operaciones con arreglos 
import matplotlib.pyplot as plt

# Lectura de imagen del sistema

def histogram(path:str = "img/baboon.png"):#Type writting, definimos parametro default (recibe path de imagen y devuelve filas, columnas, frecuencias del histograma
    img = cv.imread(path)
    fil,col,__ = img.shape # Obtenemos las carácteristicas de la imagen (columnas, filas y canales)

    # Pasamos la imagen a escala de grises
    imgGS = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #A partir de esa imagen, contruimos el histograma
    frec = np.zeros(256) #Creamos un arreglo para guardar las frecuencias con las que aparece cada color 

    #Calculamos las frecuencias de cada color
    for x in range(fil):
        for y in range(col):
            hue = imgGS[x][y]
            frec[hue] += 1

    bins = [x for x in range(256)] #vector aplanado [0-255]

    fig,ax = plt.subplots(1,2) #creamos una figura
    fig.suptitle('Histograma y escala de grises', fontsize=16)
    ax[0].imshow(imgGS,cmap='gray') #muestra la imagen a escala de grise
    ax[0].set_title('Imagen Escala de Grises')
    ax[1].bar(bins,frec) #muestra el histograma
    ax[1].set_title('Histograma')
    plt.show()

    return fil,col,imgGS,frec


def Acumulative (fil:int,col:int,frec): # función que recibe filas columnas y frecuencias, y retorna la probabilidad acumulada para los niveles de gris
    #Calculamos el histograma acumulativo
    n = col*fil #Tamaño de la imagen (pixeles totales)
    bins = [x for x in range(256)] 
    px = np.zeros(256) # vector de probabilidad
    cdf = np.zeros(256) # vector de probabilidades acumuladas
    for i in range(256):#numero de ocurrencias en i sobre pixeles totales
        px[i]=frec[i]/n # cálculo de probabilidad de ocurrencia por nivel
        for j in range(i+1):
            cdf[i]+=px[j] # cálculo de probabilidad acumulada

    plt.bar(bins,cdf) # histograma acumulado
    plt.title('Histograma acomulativo', fontsize=16)
    plt.show() 

    return cdf


def equalize(fil:int,col:int,imgGS,cdf)->None: #Función que ecualiza la imagen dada la probabilidad de ocurrencia acumulada
    eq_image = np.zeros((fil,col)) # Matriz de ceros con el tamaño de la imagen

    # construimos la imagen ecualizada
    for x in range(fil): # for anidado para asignar a cada pixel el valor del cdf correspondiente al nivel del pixel por 255
        for y in range(col):
            eq_image[x][y] = int(cdf[imgGS[x][y]]*255)

    
    

    fig,ax = plt.subplots(1,2) # creamos una nueva figura
    fig.suptitle('Comparación de Ecualización de Histograma', fontsize=16)
    ax[0].imshow(eq_image,cmap='gray') # mostramos la imagen equalizada
    ax[0].set_title('Imagen Ecualizada')
    ax[1].imshow(imgGS,cmap='gray') # mostramos la imagen a escala de grise
    ax[1].set_title('Imagen Escala de grises')
    plt.show()



def __main__(): # main
    fil,col,imgGS,frec = histogram() # histograma
    cdf=Acumulative(fil,col,frec) # histograma acumulado
    equalize(fil,col,imgGS,cdf) # imagen ecualizada
    return 0

if __name__=="__main__": # entry-point
    __main__()
    pass



