#!/usr/bin/env python3

# modulos necesarios
import cv2 as cv
import numpy as np

# filtro gaussiano
def gaussianFilter(mean : float,variance : float, size : int)->float:
    
    # Validamos el tamaño definido por el usuario
    if (size)%2 and size>1:
        kernel=np.zeros((size,size))
        print("válido")
    else:
        print("no válido")
        return None
    
    mean +=(size//2)#definimos la media en el centro con size//2 y a partir de ahí damos iffset con la media
    # Definimos cada elemento del filtro
    for x in range(size):
        for y in range(size):
            kernel[x][y]=(1/(np.pi*2*(variance**2)))*np.exp((-((x-mean)**2+(y-mean)**2))/(2*(variance**2)))#función de la campana de gauss en 2D normalizada a 
    
    return kernel # retorna el kernal


def __main__(): # función main

    kernel=gaussianFilter(1,5,11) # llamamos a la función para generar el filtro
    size,__=kernel.shape # obtenemos el tamaño del filtro
    
    img=cv.imread("img/kodakimagecollection/kodim01.png") # leemos una imagen
    imgSC=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # tranformamos la imagen a escala de grises
    fil,col,__ = img.shape # obtenemos la figura de la imagen

    imgSCfloat = imgSC.astype('float') # tranformamos el tipo de dato del array de la imagen a flotante
    imgNew=np.zeros((fil,col),dtype='float') # Creamos una imagen del tamaño de la original de tipo flotante

    # Aplicamos el filtro a la imagen y asignamos el valor computado a la nueva
    for m in range(size//2,fil-size//2):
        for n in range(size//2,col-size//2):
            aux = 0.0 # variable para almacenar la combinación lineal del filtrado
            for k in range(size):
                for l in range(size):
                    aux += kernel[k][l] * imgSCfloat[m+k-size//2][n+l-size//2] # computar cada valor vecino por el valor de la función definida en el filtro
            imgNew[m][n] = aux

    imgNew = imgNew.astype('uint8') # modificamos el tipo a entero de 8 bytes (8 niveles)

    # mostramos la imagen
    cv.imshow("Original", imgSC)
    cv.imshow("Filtrada", imgNew)

    #ciclo while
    while True:
        # Leemos del teclado
        key = cv.waitKey(1000)
        # Verificamos si la ventana es visible
        win =  cv.getWindowProperty('Original', cv.WND_PROP_VISIBLE)

        # si se preciona la tecla ESC
        if key == 27 or key == ord("q") or win < 1.0:
            break

    cv.destroyAllWindows() # llamamos al destructor!!!

# entry point
if __name__=='__main__':
    __main__()