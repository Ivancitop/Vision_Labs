#!/usr/bin/env python3

#imports necesarios para desarrollar operaciones
import cv2 as cv # opencv-python para procesamiento de imagen
import numpy as np # manejar operaciones con arreglos 
import matplotlib.pyplot as plt

# Lectura de imagen del sistema

def histogram(path:str = "img/baboon.png"):#Type writting, definimos parametro default
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

    bins = [x for x in range(256)] 

    fig,ax = plt.subplots(1,2)
    fig.suptitle('Histograma y escala de grises', fontsize=16)
    ax[0].imshow(imgGS,cmap='gray')
    ax[0].set_title('Imagen Escala de Grises')
    ax[1].bar(bins,frec)
    ax[1].set_title('Imagen Ecualizada')
    plt.show()

    return fil,col,imgGS,frec


def Acumulative (fil:int,col:int,frec):
    #Calculamos el histograma acumulativo
    n = col*fil #Tamaño de la imagen (pixeles totales)
    bins = [x for x in range(256)] 
    px = np.zeros(256) 
    cdf = np.zeros(256) 
    for i in range(256):#numero de ocurrencias en i sobre pixeles totales
        px[i]=frec[i]/n
        for j in range(i+1):
            cdf[i]+=px[j]

    plt.bar(bins,cdf)
    plt.title('Histograma acomulativo', fontsize=16)
    plt.show()

    return cdf


def equalize(fil:int,col:int,imgGS,cdf)->None:
    eq_image = np.zeros((fil,col))
    for x in range(fil):
        for y in range(col):
            eq_image[x][y] = int(cdf[imgGS[x][y]]*255)

    
    

    fig,ax = plt.subplots(1,2)
    fig.suptitle('Comparación de Ecualización de Histograma', fontsize=16)
    ax[0].imshow(eq_image,cmap='gray')
    ax[0].set_title('Imagen Ecualizada')
    ax[1].imshow(imgGS,cmap='gray')
    ax[1].set_title('Imagen Escala de grises')
    plt.show()



def __main__():
    fil,col,imgGS,frec = histogram()
    cdf=Acumulative(fil,col,frec)
    equalize(fil,col,imgGS,cdf)
    return 0

if __name__=="__main__":
    __main__()
    pass



