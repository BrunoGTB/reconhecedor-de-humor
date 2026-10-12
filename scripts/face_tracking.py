import cv2 as cv
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scripts.visualization_utils import visualize

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "detector.tflite"
)

# Criando um Objeto detector de faces
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH), 
                                  delegate=python.BaseOptions.Delegate.CPU
)
opcoes = vision.FaceDetectorOptions(base_options=base_options)
reconhecedor_faces = vision.FaceDetector.create_from_options(opcoes)

def face_tracking(frame):
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # Detecta faces no frame
    faces = reconhecedor_faces.detect(frame_mp)

    frame_copia = np.copy(frame_mp.numpy_view())
    frame_anotado = visualize(frame_copia, faces)
    return cv.cvtColor(frame_anotado, cv.COLOR_BGR2RGB)

