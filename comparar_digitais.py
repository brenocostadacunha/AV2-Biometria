import cv2
import numpy as np
import os
import time
import concurrent.futures

def carregar_descritores_arquivo(caminho_arquivo):
    try:
        with open(caminho_arquivo, 'r') as arquivo_txt:
            linhas = arquivo_txt.readlines()
            descritores = np.array([[float(valor) for valor in linha.split()] for linha in linhas], dtype=np.float32)
        return descritores
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {caminho_arquivo}")
        return None

def encontrar_melhor_correspondencia_por_banco(descritores_biometria, pasta_base):
    melhor_similaridade = 0
    melhor_caminho_imagem = None
    melhor_porcentagem = 0

    for nome_arquivo in os.listdir(pasta_base):
        if nome_arquivo.endswith(('.tif')):
            caminho_imagem = os.path.join(pasta_base, nome_arquivo)
            fingerprint_database_image = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

            sift = cv2.SIFT_create()
            keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

            # Comparar descritores usando FLANN
            matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict()).knnMatch(descritores_biometria, descriptors_2, k=2)
            match_points = [m for m, n in matches if m.distance < 0.75 * n.distance]

            similaridade = len(match_points)
            porcentagem_similaridade = (similaridade / len(descritores_biometria)) * 100

            if similaridade > melhor_similaridade:
                melhor_similaridade = similaridade
                melhor_caminho_imagem = caminho_imagem
                melhor_porcentagem = porcentagem_similaridade

    return melhor_similaridade, melhor_caminho_imagem, melhor_porcentagem

def processar_biometria(nome_arquivo_biometria, pastas_bases_dados):
    caminho_biometria = os.path.join("Banco", "biometria", nome_arquivo_biometria)
    descritores_biometria = carregar_descritores_arquivo(caminho_biometria)
    if descritores_biometria is not None:
        melhor_resultado = None
        for pasta_base in pastas_bases_dados:
            melhor_similaridade, melhor_caminho_imagem, melhor_porcentagem = encontrar_melhor_correspondencia_por_banco(
                descritores_biometria, pasta_base
            )
            if melhor_caminho_imagem:
                resultado = (
                    f"{nome_arquivo_biometria:<10} | {pasta_base.split('/')[1]:<15} | "
                    f"{melhor_caminho_imagem.split('/')[1]:<25} | {melhor_porcentagem:>.2f}%"
                )
                if melhor_resultado is None or melhor_similaridade > melhor_resultado[0]:
                    melhor_resultado = (melhor_similaridade, resultado)
        if melhor_resultado:
            return melhor_resultado[1]
    return None

 
pastas_bases_dados = [
    'Banco/BD1',
    'Banco/BD2',
    'Banco/BD3',
    'Banco/BD4'
]

 
inicio = time.time()

 
header = f"{'Biometria':<10} | {'Banco de Dados':<15} | {'Imagem Correspondente':<25} | Similaridade"
separador = f"{'-'*10} | {'-'*15} | {'-'*25} | {'-'*12}"
print(header)
print(separador)

 
resultados_finais = []

 
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(processar_biometria, f"file{i}.txt", pastas_bases_dados) for i in range(1, 9)]
    for future in concurrent.futures.as_completed(futures):
        resultado = future.result()
        if resultado:
            resultados_finais.append(resultado)
            print(resultado)
    print(separador)

 
with open('resultados_biometria.txt', 'w') as arquivo:
    arquivo.write(header + "\n")
    arquivo.write(separador + "\n")
    for resultado in resultados_finais:
        arquivo.write(resultado + "\n")

 
fim = time.time()
tempo_total = fim - inicio
print(f"Tempo total de execução: {tempo_total:.2f} segundos")
