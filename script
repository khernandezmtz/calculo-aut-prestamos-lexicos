# Lingüística computacional 2026
# Primer borrador proyecto final
# Kevin Efrén Hernández Martínez

import pandas as pd
import numpy as np
from collections import Counter
import re  

class DetectorPrestamosNahuatl:
    def __init__(self, vocabulario, umbral_tau=0.5):
        """
        Constructor de la clase. Inicializa el vocabulario limpio, define el umbral 
        de clasificación y calcula las líneas base estadísticas del corpus.
        """
        self.vocabulario = [w.lower().strip() for w in vocabulario if len(w) > 0]
        self.umbral_tau = umbral_tau
        self.ngram_freqs = self._calcular_frecuencias_ngrams()
        self.transition_probs = self._calcular_probabilidades_transicion()

    def _calcular_frecuencias_ngrams(self, n_min=2, n_max=13):
        """
        Calcula la frecuencia relativa de n-gramas de caracteres (desde n_min hasta n_max)
        en todo el vocabulario para establecer qué combinaciones son comunes o raras.
        """
        total_ngrams = 0
        contador_global = Counter()
        for palabra in self.vocabulario:
            # Extrae n-gramas respetando la longitud máxima de cada palabra
            for n in range(n_min, min(len(palabra) + 1, n_max + 1)):
                ngrams = [palabra[i:i+n] for i in range(len(palabra) - n + 1)]
                contador_global.update(ngrams)
                total_ngrams += len(ngrams)
        # Retorna un diccionario con las probabilidades relativas de cada n-grama
        return {ng: count / total_ngrams for ng, count in contador_global.items()} if total_ngrams > 0 else {}

    def _calcular_probabilidades_transicion(self):
        """
        Calcula las probabilidades de transición condicional P(c2 | c1) entre 
        caracteres consecutivos utilizando delimitadores de inicio (^) y fin ($).
        """
        bigramas_contador = Counter()
        unigramas_contador = Counter()
        for palabra in self.vocabulario:
            padded = "^" + palabra + "$"
            for i in range(len(padded) - 1):
                c1, c2 = padded[i], padded[i+1]
                bigramas_contador[(c1, c2)] += 1
                unigramas_contador[c1] += 1
        return {(c1, c2): count / unigramas_contador[c1] for (c1, c2), count in bigramas_contador.items()}

    def extraer_caracteristicas(self, palabra):
        """
        Extrae métricas cuantitativas internas para una palabra:
        1. Rare n-gram score (anomalía de subcadenas).
        2. Rare transition score (anomalía de transiciones fonotácticas).
        """
        # 1. Cálculo de n-gramas raros (hasta n=13)
        ngrams_palabra = []
        for n in range(2, min(len(palabra) + 1, 14)):
            ngrams_palabra.extend([palabra[i:i+n] for i in range(len(palabra) - n + 1)])

        rare_ngram = 0.0
        if ngrams_palabra:
            for ng in ngrams_palabra:
                p_ng = self.ngram_freqs.get(ng, 0.0)
                # Penalización fuerte si el n-grama es muy infrecuente en el corpus
                if p_ng < 0.005:
                    rare_ngram += 1.0 * (0.005 - p_ng)
                elif 0.005 <= p_ng < 0.02:
                    rare_ngram += 0.5 * (0.02 - p_ng)
            rare_ngram /= len(ngrams_palabra)

        # 2. Cálculo de transiciones raras
        padded = "^" + palabra + "$"
        transiciones = [(padded[i], padded[i+1]) for i in range(len(padded) - 1)]
        rare_trans = 0.0
        if transiciones:
            for t in transiciones:
                p_trans = self.transition_probs.get(t, 0.0)
                # Penalización si la transición de caracteres es atípica
                if p_trans < 0.01:
                    rare_trans += 100 * (0.01 - p_trans)
                elif 0.01 <= p_trans < 0.05:
                    rare_trans += 20 * (0.05 - p_trans)
            rare_trans /= len(transiciones)

        return rare_ngram, rare_trans

    def _sigmoide(self, x, gamma=2.0, theta=0.5):
        """Función de mapeo sigmoideo para acotar puntuaciones a probabilidades entre [0, 1]."""
        return 1 / (1 + np.exp(-gamma * (x - theta)))

    def clasificar_vocabulario(self):
        """
        Ejecuta el pipeline completo: extracción, normalización Min-Max, puntuación 
        compuesta, mapeo de probabilidad y clasificación binaria (0 = nativa, 1 = préstamo).
        """
        resultados = []

        # Paso 1: Extracción de características crudas por palabra
        features_raw = []
        for palabra in self.vocabulario:
            rn, rt = self.extraer_caracteristicas(palabra)
            features_raw.append({'palabra': palabra, 'rn': rn, 'rt': rt})

        df_feats = pd.DataFrame(features_raw)

        # Paso 2: Normalización Min-Max para estandarizar las escalas de ambas métricas
        df_feats['rn_norm'] = (df_feats['rn'] - df_feats['rn'].min()) / (df_feats['rn'].max() - df_feats['rn'].min() + 1e-9)
        df_feats['rt_norm'] = (df_feats['rt'] - df_feats['rt'].min()) / (df_feats['rt'].max() - df_feats['rt'].min() + 1e-9)

        # Paso 3: Ponderación de características (alpha_1 y alpha_2)
        alpha_1 = 0.6  # Peso para n-gramas raros
        alpha_2 = 0.4  # Peso para transiciones raras

        for idx, row in df_feats.iterrows():
            # Puntuación compuesta s(w)
            score_s = (alpha_1 * row['rn_norm']) + (alpha_2 * row['rt_norm'])

            # Conversión a probabilidad p(w) vía sigmoide
            probabilidad_p = self._sigmoide(score_s)

            # Clasificación binaria basada en el umbral tau
            etiqueta_binaria = 1 if probabilidad_p >= self.umbral_tau else 0

            resultados.append({
                'palabra': row['palabra'],
                'rn': round(row['rn'], 6),
                'rt': round(row['rt'], 6),
                'score_compuesto': round(score_s, 4),
                'probabilidad': round(probabilidad_p, 4),
                'prediccion_binaria': etiqueta_binaria
            })

        df_resultados = pd.DataFrame(resultados)
        # Ordenar el DataFrame de mayor a menor según el puntaje de anomalía compuesto
        df_resultados = df_resultados.sort_values(by='score_compuesto', ascending=False)
        return df_resultados

def load_words_from_file(filepath):
    """
    Lee un archivo de texto, extrae palabras mediante regex, las convierte a minúsculas
    y devuelve una lista con los tipos léxicos únicos (types).
    """
    with open(ruta, 'r', encoding='utf-8') as f:
        text = f.read()
    words = re.findall(r'\b\w+\b', text.lower())
    unique_words = list(set(words))
    return unique_words

# --- Bloque de Ejecución y Prueba ---
ruta = 'tequila zongolica asr.txt'

# Carga de palabras y ejecución del clasificador
corpus_prueba_from_file = load_words_from_file(file_path)

detector = DetectorPrestamosNahuatl(corpus_prueba_from_file, umbral_tau=0.5)
df_clasificacion = detector.clasificar_vocabulario()
print(df_clasificacion)

# Exportación de resultados a un archivo CSV
df_clasificacion.to_csv('clasificacion_palabras.csv', index=False)
print("Resultados guardados en 'clasificacion_palabras.csv'")
