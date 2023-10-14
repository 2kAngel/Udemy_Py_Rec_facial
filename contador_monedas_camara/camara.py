import cv2 as cv
capturarVideo=cv.VideoCapture(0) #0-> cam interna | 1 - cam externa
if not capturarVideo.isOpened():
    print("Camara not found")
    exit()
while True:
    tipocamara,camara=capturarVideo.read()
    grises=cv.cvtColor(camara, cv.COLOR_BGR2GRAY)
    cv.imshow("VIVO" , grises)
    if cv.waitKey(1) == ord("q"):
        break

capturarVideo.release()
cv.destroyAllWindows()