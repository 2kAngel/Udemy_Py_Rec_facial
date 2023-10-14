import cv2 as cv
import numpy as np

def ordenarpuntos(puntos):
    n_puntos=np.concatenate([puntos[0],puntos[1],puntos[2],puntos[3]]).tolist()
    y_order=sorted(n_puntos,key=lambda n_puntos:n_puntos[1])
    x_order=y_order[:2]
    x_order=sorted(x_order,key=lambda x_order:x_order[0])
    x2_order=y_order[2:4]
    x2_order=sorted(x2_order, key=lambda x2_order:x2_order[0])
    return [x_order[0],x_order[1],x2_order[0],x2_order[1]]

def alineamiento(imagen,ancho,alto):
    imagen_alineada=None
    grises=cv.cvtColor(imagen,cv.COLOR_BGR2GRAY)
    tipoumbral,umbral = cv.threshold(grises,150,255,cv.THRESH_BINARY)
    cv.imshow("Umbral",umbral)
    contorno=cv.findContours(umbral,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]
    contorno=sorted(contorno,key=cv.contourArea,reverse=True)[:1]

    for c  in contorno:
        epsilon=0.01*cv.arcLength(c,True)
        aprox= cv.approxPolyDP(c,epsilon,True) #closed = True (cierra el circulo)
        if len(aprox)==4:
            puntos=ordenarpuntos(aprox)
            punto1=np.float32(puntos)
            punto2=np.float32([[0,0],[ancho,0],[0,alto],[ancho,alto]])
            M_perspectiva = cv.getPerspectiveTransform(punto1,punto2)
            imagen_alineada = cv.warpPerspective(imagen,M_perspectiva,(ancho,alto))

    return imagen_alineada
captura_video = cv.VideoCapture(0)

while True:
    tipoCam,cam=captura_video.read()
    if tipoCam==False:
        break

    imagen_A6 = alineamiento(cam,ancho=480,alto=677)

    if imagen_A6 is not None:
        puntos = []
        imagen_gris = cv.cvtColor(imagen_A6,cv.COLOR_BGR2GRAY)
        blur=cv.GaussianBlur(imagen_gris,(5,5),1)
        _,umbral2= cv.threshold(blur,0,255,cv.THRESH_OTSU+cv.THRESH_BINARY_INV)
        cv.imshow("uMBRAL",umbral2)
        contorno2 = cv.findContours(umbral2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[0]
        cv.drawContours(imagen_A6,contorno2,-1,(255,0,0),2)
        
        #moneda de peru 
        suma1= 0.0
        suma2= 0.0

        for c_2 in contorno2:
            area= cv.contourArea(c_2)
            Momentos = cv.moments(c_2)
            if (Momentos["m00"]==0):
                Momentos["m00"]=1.0
            x=int(Momentos["m10"]/Momentos["m00"])
            y=int(Momentos["m01"]/Momentos["m00"])

            if area<9300 and area>8000:
                font=cv.FONT_HERSHEY_SIMPLEX
                cv.putText(imagen_A6,"Soles: 0.2", (x,y), font, 0.75, (0,255,0), 2)
                suma1=suma1+0.2

            if area<7800 and area>6500:
                font=cv.FONT_HERSHEY_SIMPLEX
                cv.putText(imagen_A6,"Soles: 0.1", (x,y), font, 0.75, (0,255,0), 2)
                suma1=suma1+0.1
            
        total=suma1+suma2
        print("Total en cents -> ", round(total,2))
        cv.imshow("Imagen A6", imagen_A6)
        cv.imshow("cam", cam)
    if cv.waitKey(1) == ord('s'):
        break
captura_video.release()
cv.destroyAllWindows()




