import cv2 as cv

# Classificador de Faces
face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_classifier.empty():
    raise RuntimeError("Não foi possível carregar o classificador de faces.")

def face_tracking(frame):
    # Transforma retângulo na escala de cinza
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detecta as faces
    face = face_classifier.detectMultiScale(
    gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40) 
    )
    
    # Cria retângulo ao redor da face
    for (x, y, w, h) in face:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

