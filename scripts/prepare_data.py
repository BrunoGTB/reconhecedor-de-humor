import csv, random
import cv2 as cv
import pandas as pd
from pathlib import Path
from face_mesh import get_face_blendshape, variacoes

RAIZ = Path(__file__).resolve().parent.parent / "data"
DATASETS = RAIZ / "datasets"
CSV = RAIZ / "csv"
TREINO = DATASETS / "train"
TESTE = DATASETS / "test"

colunas = [
    '_neutral',
    'browDownLeft',
    'browDownRight',
    'browInnerUp',
    'browOuterUpLeft',
    'browOuterUpRight',
    'cheekPuff',
    'cheekSquintLeft',
    'cheekSquintRight',
    'eyeBlinkLeft',
    'eyeBlinkRight',
    'eyeLookDownLeft',
    'eyeLookDownRight',
    'eyeLookInLeft',
    'eyeLookInRight',
    'eyeLookOutLeft',
    'eyeLookOutRight',
    'eyeLookUpLeft',
    'eyeLookUpRight',
    'eyeSquintLeft',
    'eyeSquintRight',
    'eyeWideLeft',
    'eyeWideRight',
    'jawForward',
    'jawLeft',
    'jawOpen',
    'jawRight',
    'mouthClose',
    'mouthDimpleLeft',
    'mouthDimpleRight',
    'mouthFrownLeft',
    'mouthFrownRight',
    'mouthFunnel',
    'mouthLeft',
    'mouthLowerDownLeft',
    'mouthLowerDownRight',
    'mouthPressLeft',
    'mouthPressRight',
    'mouthPucker',
    'mouthRight',
    'mouthRollLower',
    'mouthRollUpper',
    'mouthShrugLower',
    'mouthShrugUpper',
    'mouthSmileLeft',
    'mouthSmileRight',
    'mouthStretchLeft',
    'mouthStretchRight',
    'mouthUpperUpLeft',
    'mouthUpperUpRight',
    'noseSneerLeft',
    'noseSneerRight',
    'emotion'
]

colunas_bs = [c.rstrip(":") for c in colunas if c != "emotion"]
colunas_lm = [f"lm{i}_{eixo}" for i in range(478) for eixo in "xyz"]
colunas = colunas_bs + colunas_lm + ["emotion"]

from tqdm import tqdm

# Lista todas as imagens dos datasets
imagens = [
    (dataset, emocao, caminho_imagem)
    for dataset in DATASETS.iterdir()
    if dataset.is_dir()
    for emocao in dataset.iterdir()
    if emocao.is_dir()
    for caminho_imagem in emocao.iterdir()
    if caminho_imagem.is_file()
]

dados_por_dataset = {dataset.name: [] for dataset in DATASETS.iterdir()
                     if dataset.is_dir()}

dados, falhas = [], 0
tentativas = 0

for dataset, emocao, caminho_imagem in tqdm(
    imagens,
    desc="Processando imagens",
    unit="imagem"
):
    imagem = cv.imread(str(caminho_imagem), cv.IMREAD_GRAYSCALE)

    if imagem is None:
        continue

    imagem = cv.resize(imagem, (256, 256), interpolation=cv.INTER_CUBIC)
    versoes = variacoes(imagem) if dataset.name == "train" else {"orig": imagem}

    for tipo, v in versoes.items():
        tentativas += 1
        face_blendshape = get_face_blendshape(cv.cvtColor(v, cv.COLOR_GRAY2BGR))

        if face_blendshape is None:
            falhas += 1
            continue

        # Se a função retorna uma lista de valores
        linha = list(face_blendshape)
        linha.append(emocao.name)

        dados_por_dataset[dataset.name].append(linha)

print(f"Falhas: {falhas}/{tentativas} ({100 * falhas / tentativas:.1f}%)")

for nome_dataset, linhas in dados_por_dataset.items():
    assert all(len(l) == len(colunas) for l in linhas), "linha com tamanho diferente do cabeçalho"

# Salva os CSVs depois do processamento
for nome_dataset, dados in dados_por_dataset.items():
    saida = CSV / f"{nome_dataset}.csv"

    with open(saida, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(colunas)
        escritor.writerows(dados)

    print(f"CSV criado em {saida}: {len(dados)} registros")
