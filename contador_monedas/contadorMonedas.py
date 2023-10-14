import cv2
import numpy as np

varGauss = 3
varKernel = 3

original=cv2.imread('/Users/Administrador/Desktop/ANGEL/UDEMY/UDEMY_PHYTON/contador_monedas/images.png')  #leemos img
gris = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY) #pasamos a grises
gauss = cv2.GaussianBlur(gris,(varGauss,varGauss),0)
canny = cv2.Canny(gauss,60,100)
kernel=np.ones((varKernel,varKernel),np.uint8)
cierre=cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel) #elimina ruido intraimg

contornos,jerarquia = cv2.findContours(cierre.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

print("monedas encontradas: {}".format(len(contornos)))

cv2.drawContours(original,contornos,-1,(0,0,255),2)
#mostramos resultados
#cv2.imshow("Grises",gris) # ver en gris
#cv2.imshow("Gauss",gauss)# ver desenfoque gaussiano (paso 1 quita-ruido)
#cv2.imshow("Canny",canny) # ver en bordes (contornos)
cv2.imshow("Cierre",cierre) # ver el cierre

cv2.imshow("Resultado",original) # ver el result




cv2.waitKey(0) #0 == img || 1 == video

