# %%
import os
import cv2
import numpy as np
import pandas as pd

MAIN_PATH = os.path.dirname(os.path.abspath(''))
IMAGES_PATH = os.path.join(MAIN_PATH, 'images/calibracao')
RESULTS_PATH = os.path.join(MAIN_PATH, 'results')

# %%


def extract_and_convert_to_float(column):
    """função que toma apenas a parte numerica de uma coluna de um dataframe

    Args:
        column (string): _description_

    Returns:
        dataframe: columna apenas com os numero no formato float
    """
    return column.str.extract('([\d.]+)').astype(float)


def intervals(li, lf, ci, cf, imgi, imgf):
    """função para definir os intervalos de linhas, colunas e imagens

    Args:
        li (int32): indice da linha inicial
        lf (int32): indice da linha final
        ci (int32): indice da coluna inicial
        cf (int32): indice da coluna final
        imgi (int32): numero da imagem inicial
        imgf (int32): numero da imagem final

    Returns:
        tuple: retorna os indices iniciais e finais e também um array desse intervalo
    """
    IntervaloLinhas = np.linspace(li, lf, num=lf-li+1, dtype=int)
    IntervaloColunas = np.linspace(ci, cf, num=cf-ci+1, dtype=int)
    IntervaloImagens = np.linspace(imgi, imgf, imgf-imgi+1, dtype=int)

    return li, lf, ci, cf, imgi, imgf, IntervaloLinhas, IntervaloColunas, IntervaloImagens

# %%


li, lf, ci, cf, imgi, imgf, IntervaloLinhas, IntervaloColunas, IntervaloImagens = intervals(
    1, 2048, 1, 2048, 1, 206)

# %%
arquivo_txt = None

arquivos_na_pasta = os.listdir(IMAGES_PATH)

for arquivo in arquivos_na_pasta:
    if arquivo.endswith('.txt'):
        arquivo_txt = os.path.join(IMAGES_PATH, arquivo)
        break

df_raw = pd.read_csv(arquivo_txt, sep='\t', skiprows=1, header=None)
df_raw.columns = ['imagem', 'temp_k', 'h_oe_dc',
                  'h_oe_ac', 'tempo', 'exp_us', 'ganho', 'offset']


info = df_raw.copy()

info['temp_k'] = extract_and_convert_to_float(info['temp_k'])
info['h_oe_dc'] = extract_and_convert_to_float(info['h_oe_dc'])
info['h_oe_ac'] = extract_and_convert_to_float(info['h_oe_ac'])


# %%

grau = 2

I = np.empty((len(IntervaloLinhas), len(
    IntervaloColunas), len(IntervaloImagens)))
coefs = np.empty((len(IntervaloLinhas), len(IntervaloColunas), grau + 1))

# %%

arquivos_tiff = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.tiff')]
arquivos_tiff.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
arquivos_tiff = arquivos_tiff[0:imgf]


for n, nome_arquivo in enumerate(arquivos_tiff):
    caminho_completo = os.path.join(IMAGES_PATH, nome_arquivo)
    # Carregue a imagem em escala de cinza, ajuste conforme necessário
    imagem = cv2.imread(caminho_completo, cv2.IMREAD_GRAYSCALE)
    imagem = np.array(imagem)
    I[:, :, n] = imagem

# %%

x = info.loc[0:imgf-1, 'h_oe_dc'].to_numpy()

for i in range(len(IntervaloLinhas)):
    for j in range(len(IntervaloColunas)):
        # Transforme I(i, j, :) em uma matriz unidimensional
        dados = I[i, j, :].squeeze()

        # Realize a regressão polinomial de segunda ordem (polinômio quadrático)
        coefs[i, j, :] = np.polyfit(x, dados, grau)

    print(i)  # Exibe o progresso

# %%

np.save(os.path.join(RESULTS_PATH, 'calibracao/coefs.npy'), coefs)
