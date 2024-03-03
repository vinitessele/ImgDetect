import cv2

classificador = cv2.CascadeClassifier(r"D:\ImagensDigitaisPy\exemploDeteccaoImg\cascades\haarcascade_frontalface_default.xml")
imagem = cv2.imread(r"D:\ImagensDigitaisPy\exemploDeteccaoImg\3.jpg")

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original",imagem)
cv2.imshow("cinza",imagemCinza)
#cv2.waitKey(2000)
facesDeectadas = classificador.detectMultiScale(imagemCinza)
print(len(facesDeectadas))
print(facesDeectadas)

for(x,y,l,a) in facesDeectadas:
    print(x,y,l,a)
    cv2.rectangle(imagem,(x,y),(x+l,y+a),(255,0,255),2)

cv2.imshow("Encontradas",imagem)
cv2.waitKey(0)