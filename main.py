import numpy as np
import cv2 as cv
 
camera = cv.VideoCapture(0)

if not camera.isOpened():
    print("Não foi possível abrir a câmera!")
    exit()

while True:
    sucesso, frame = camera.read()

    if not sucesso:
        break

    frame = cv.flip(frame, 1)

    # Desenha um retângulo de exemplo
    cv.rectangle(
        frame,
        (150, 100),
        (450, 400),
        (0, 255, 0),
        2
    )

    # Adiciona um texto
    cv.putText(
        frame,
        "FELIZ",
        (150, 85),
        cv.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv.imshow("Detector de Emocoes", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv.destroyAllWindows()