import cv2

classificador = cv2.CascadeClassifier(r"D:\ImagensDigitaisPy\exemploDeteccaoImg\cascades\haarcascade_frontalface_default.xml")
classificadorOlhos = cv2.CascadeClassifier(r"D:\ImagensDigitaisPy\exemploDeteccaoImg\cascades\haarcascade_eye.xml")

imagem = cv2.imread(r"D:\ImagensDigitaisPy\exemploDeteccaoImg\2.png")

imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Original",imagem)
#cv2.imshow("cinza",imagemCinza)
#cv2.waitKey(2000)
facesDeectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.09,minNeighbors=8,minSize=(30,30))
#print(len(facesDeectadas))
#print(facesDeectadas)

for(x,y,l,a) in facesDeectadas:
    print(x,y,l,a)
    cv2.rectangle(imagem,(x,y),(x+l,y+a),(100,100,255),2)
    regiao =  imagem[y:y + a, x:x +l]
    regiaoCinzaOlhos = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    olhosDetectados = classificadorOlhos.detectMultiScale(regiaoCinzaOlhos,scaleFactor=1.03,minNeighbors=3,minSize=(60,60))
    print(olhosDetectados)
    for(ox,oy,ol,oa) in olhosDetectados:
        cv2.rectangle(imagem,(ox,oy),(ox+ol,oy+oa),(100,500,255),2)

cv2.imshow("Encontradas",imagem)
cv2.waitKey(0)