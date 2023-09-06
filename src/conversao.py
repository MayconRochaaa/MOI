# %%
import os
import cv2
import PIL
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


nome_medida = 'medida_5k'
MAIN_PATH = os.path.dirname(os.path.abspath(''))
IMAGES_PATH = os.path.join(MAIN_PATH, 'images/'+nome_medida)
RESULTS_PATH = os.path.join(MAIN_PATH, 'results/'+nome_medida)
CALIBRACAO_PATH = os.path.join(MAIN_PATH, 'results/calibracao')

# %%


def extract_and_convert_to_float(column):
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
coefs = np.load(os.path.join(CALIBRACAO_PATH, 'coefs.npy'))

# %%

# Suponha que 'IntervaloLinhas', 'IntervaloColunas', 'I', 'coefs' e 'imgf' estejam definidos
B0 = np.zeros((2048, 2048))
# B=np.empty((2048,2048))
for i in range(0, imgf):
    # Define o nome das imagens a serem salvas
    FileNameTiff = f'B_NbN_pe_{i+1:03d}.tiff'
    B = np.empty((2048, 2048))
    for j in range(len(IntervaloLinhas)):
        for k in range(len(IntervaloColunas)):

            a = coefs[j, k, 0]
            b = coefs[j, k, 1]
            c = coefs[j, k, 2] - I[j, k, i]

            discriminant = b**2-4*a*c

            B[j, k] = np.real((-b+np.sqrt(discriminant))/(2*a))

    B = B - B0

    if i == 0:
        B0 = B

    caminho_arquivo = f'{RESULTS_PATH}/{FileNameTiff}'
    # Salve a matriz B como uma imagem .tiff usando tifffile
    im2 = PIL.Image.fromarray(B)
    im2.save(caminho_arquivo)

    print(i)  # Opcional: exibe o contador no terminal

# %%
xx = np.linspace(0, 2048, 2048)
yy = B[1024, :]

# %%

plt.scatter(xx, yy)
# %%
