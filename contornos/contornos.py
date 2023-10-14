
import cv2


imagen=cv2.imread('/Users/Administrador/Desktop/ANGEL/UDEMY/UDEMY_PHYTON/contornos/contorno.jpg')
grises= cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
tipo_umbral,umbral=cv2.threshold(grises,100,255,cv2.THRESH_BINARY)
#tipoUmbral no lo uso en este ejercicio, pero necesito las dos vars (puedes cambiarlo por _)

contorno,jerarquia = cv2.findContours(umbral,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imagen,contorno,-1,(251,60,50),3)# -1 = todos los contornos, (puedes elegir 1 = contono 1 , 2 = cont2... ; )parentesis = color del contorno

cv2.imshow('imagen original',imagen)
cv2.imshow('imagen grises',grises)
cv2.imshow('imagen umbral',umbral)

cv2.waitKey(0) #0 = imagen || 1 = video 

cv2.destroyAllWindows()#pa destruir las imgs pulsando any

