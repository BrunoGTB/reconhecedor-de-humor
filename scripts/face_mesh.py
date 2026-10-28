import cv2 as cv
import pickle
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from visualization_utils import draw_landmarks_on_image

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
        / "models"
        / "face_landmarker_v2_with_blendshapes.task"
)

PKL_DIR = Path("data/pickle")


# Criando um Objeto de Malha Facial
base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH),
    delegate=python.BaseOptions.Delegate.CPU
)
opcoes = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.1,
    min_face_presence_confidence=0.1
)
reconhecedor_malha_facial = vision.FaceLandmarker.create_from_options(opcoes)

def preparar(imagem, pad=16, tamanho=256):
    # borda ao redor do rosto, para o detector enxergar "contexto"
    imagem = cv.copyMakeBorder(imagem, pad, pad, pad, pad, cv.BORDER_REPLICATE)
    imagem = cv.resize(imagem, (tamanho, tamanho), interpolation=cv.INTER_CUBIC)
    return cv.cvtColor(imagem, cv.COLOR_GRAY2RGB)

def variacoes(img):
    h, w = img.shape
    out = {"orig": img, "flip": cv.flip(img, 1)}
    for ang in (-10, 10):
        M = cv.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out[f"rot{ang}"] = cv.warpAffine(img, M, (w, h), borderMode=cv.BORDER_REPLICATE)
    return out

def face_meshing(frame):
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Detecta faces no frame
    faces = reconhecedor_malha_facial.detect(frame_mp)

    frame_anotado = draw_landmarks_on_image(frame_mp.numpy_view(), faces)

    return cv.cvtColor(frame_anotado, cv.COLOR_RGB2BGR)

def get_face_blendshape(frame):
    frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detecta faces no frame
    faces = reconhecedor_malha_facial.detect(frame_mp)

    if not faces.face_blendshapes:
        return None

    # Apresentação dos BlendShapes
    face_blendshape = []
    for categoria in faces.face_blendshapes[0]:
        face_blendshape.append(categoria.score)

    lm = np.array([[p.x, p.y, p.z] for p in faces.face_landmarks[0]])
    lm = lm - lm[1]               
    lm = lm / np.linalg.norm(lm[33, :2] - lm[263, :2])
    
    return np.concatenate([face_blendshape, lm.flatten()]).astype(np.float32)



