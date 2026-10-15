from scripts.face_mesh import face_meshing
import cv2 as cv

# Inicialização da Câmera
camera = cv.VideoCapture(0)

# Verifica se foi possível abrir a câmera
if not camera.isOpened():
    print("Não foi possível abrir a câmera!")
    exit()

# Enquanto a tecla 'Q' não for pressionada, a captura de imagem continua
while True:
    sucesso, frame = camera.read()

    # Se a captura não estiver ocorrendo corretamente, quebra o loop
    if not sucesso:
        break

    frame = cv.flip(frame, 1)   # Inverte a câmera horizontalmente, para a sensação de espelho
    frame = face_meshing(frame)        # Detecta Malha Facial 

    cv.imshow("Detector de Emocoes", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv.destroyAllWindows()