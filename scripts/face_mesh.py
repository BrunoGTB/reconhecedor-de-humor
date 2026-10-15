import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from scripts.visualization_utils import draw_landmarks_on_image

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
        / "models"
        / "face_landmarker_v2_with_blendshapes.task"
)

# Criando um Objeto de Malha Facial
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
opcoes = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
reconhecedor_malha_facial = vision.FaceLandmarker.create_from_options(opcoes)

def face_meshing(frame):
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # Detecta faces no frame
    faces = reconhecedor_malha_facial.detect(frame_mp)

    frame_anotado = draw_landmarks_on_image(frame_mp.numpy_view(), faces)

    return cv.cvtColor(frame_anotado, cv.COLOR_RGB2BGR)

