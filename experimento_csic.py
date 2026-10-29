"""
Experimento com o dataset CSIC 2010 (requisições HTTP rotuladas Normal/Anomalous).

Coloque este arquivo na mesma pasta de csic_database.csv e execute:
    python experimento_csic.py [caminho_do_csv]

Atributos: extraídos apenas da URL, do método e do corpo da requisição
(cabeçalhos como User-Agent e Cookie NÃO são usados, para evitar que o modelo
aprenda particularidades de como o dataset foi coletado).

Cenários:
  A) Não supervisionado "realista": o modelo é treinado sobre uma mistura em que
     os ataques são raros (5%), sem usar rótulos, e avaliado sobre a mesma mistura.
  B) Detecção de novidade: o modelo é treinado só com metade do tráfego normal
     e avaliado em tráfego normal inédito + todos os ataques.
  C) Não supervisionado com a proporção real do dataset (41% de ataques).
Todos repetidos com sementes diferentes; o rótulo é usado SOMENTE na avaliação.
"""
import sys
import json
import time
import math
from collections import Counter
from urllib.parse import urlsplit, unquote_plus

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

CAMINHO = sys.argv[1] if len(sys.argv) > 1 else "csic_database.csv"
ESPECIAIS = set("'\"<>;()%*|\\/`~{}[]$!#^@,:?")
COLUNAS = ["len_caminho", "len_query", "n_parametros", "len_conteudo", "metodo_post",
           "metodo_put", "len_decodificado", "n_especiais", "prop_especiais",
           "n_percent", "prop_digitos", "entropia", "max_len_valor", "log_freq_caminho"]


def entropia(s):
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def carregar(caminho):
    df = pd.read_csv(caminho, low_memory=False)
    rot = df.iloc[:, 0].astype(str)                      # Normal / Anomalous
    y = (rot.str.lower() == "anomalous").astype(int).values
    return df, y


def extrair_base(df):
    """Atributos que dependem só da própria requisição (sem estatísticas do conjunto)."""
    linhas, caminhos = [], []
    for metodo, url, cont in zip(df["Method"], df["URL"], df["content"].fillna("")):
        url = str(url).rsplit(" HTTP/", 1)[0]
        sp = urlsplit(url)
        caminho, query = sp.path, sp.query
        carga_bruta = query if query else str(cont)
        decod = unquote_plus(carga_bruta, encoding="latin-1")
        n_esp = sum(ch in ESPECIAIS for ch in decod)
        params = [p for p in carga_bruta.split("&") if p]
        valores = [unquote_plus(p.split("=", 1)[1], encoding="latin-1") if "=" in p else "" for p in params]
        linhas.append([
            len(caminho), len(query), len(params), len(str(cont)),
            int(metodo == "POST"), int(metodo == "PUT"), len(decod), n_esp,
            n_esp / (len(decod) + 1), carga_bruta.count("%"),
            sum(ch.isdigit() for ch in decod) / (len(decod) + 1), entropia(decod),
            max((len(v) for v in valores), default=0),
        ])
        caminhos.append(caminho)
    return np.array(linhas, dtype=float), np.array(caminhos)


def montar(base, caminhos, idx_treino, idx_alvo):
    """Adiciona o atributo de raridade do caminho, calculado só com o conjunto de treino."""
    freq = Counter(caminhos[idx_treino])
    def f(idx):
        lf = np.array([math.log1p(freq.get(c, 0)) for c in caminhos[idx]])
        return np.column_stack([base[idx], lf])
    return f(idx_treino), f(idx_alvo)


def metricas(y, pred, escore):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precisao=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), fpr=fp / (fp + tn),
                auc=roc_auc_score(y, escore), ap=average_precision_score(y, escore),
                tp=tp, fp=fp, fn=fn, tn=tn)


def resumir(lista):
    df = pd.DataFrame(lista)
    return {c: [float(df[c].mean()), float(df[c].std())] for c in df.columns}


def treinar_auditoria_local(X_treino, X_alvo, contaminacao, seed):
    modelo = IsolationForest(n_estimators=100, contamination=contaminacao,
                             random_state=seed, n_jobs=-1)
    modelo.fit(X_treino)
    escore = -modelo.decision_function(X_alvo)           # maior = mais anômalo
    pred = (modelo.predict(X_alvo) == -1).astype(int)
    return pred, escore


def main(n_rep=30):
    t0 = time.perf_counter()
    df, y = carregar(CAMINHO)
    base, caminhos = extrair_base(df)
    t_extr = time.perf_counter() - t0
    normais = np.where(y == 0)[0]
    anom = np.where(y == 1)[0]
    out = {"n_total": int(len(y)), "n_normal": int(len(normais)), "n_anomalo": int(len(anom)),
           "tempo_extracao_s": t_extr, "n_repeticoes": n_rep}

    # ---- Cenário A: ataques raros (5%) misturados ao tráfego normal, treino sem rótulos
    cont_A = {"0,05": 0.05, "0,10": 0.10, "auto": "auto"}
    resA = {k: [] for k in cont_A}
    for s in range(n_rep):
        rng = np.random.default_rng(s)
        n_a = int(round(len(normais) * 0.05 / 0.95))
        idx = np.concatenate([normais, rng.choice(anom, n_a, replace=False)])
        Xtr, _ = montar(base, caminhos, idx, idx)
        for nome, c in cont_A.items():
            pred, esc = treinar_auditoria_local(Xtr, Xtr, c, s)
            resA[nome].append(metricas(y[idx], pred, esc))
    out["A"] = {k: resumir(v) for k, v in resA.items()}
    out["A_n_anomalos"] = int(round(len(normais) * 0.05 / 0.95))

    # ---- Cenário B: treino só com tráfego normal; teste = normal inédito + todos os ataques
    cont_B = {"0,01": 0.01, "0,05": 0.05, "0,10": 0.10, "auto": "auto"}
    resB = {k: [] for k in cont_B}
    for s in range(n_rep):
        rng = np.random.default_rng(100 + s)
        perm = rng.permutation(normais)
        treino, teste_n = perm[: len(perm) // 2], perm[len(perm) // 2:]
        idx_teste = np.concatenate([teste_n, anom])
        Xtr, Xte = montar(base, caminhos, treino, idx_teste)
        for nome, c in cont_B.items():
            pred, esc = treinar_auditoria_local(Xtr, Xte, c, s)
            resB[nome].append(metricas(y[idx_teste], pred, esc))
    out["B"] = {k: resumir(v) for k, v in resB.items()}
    out["B_n_treino"] = int(len(normais) // 2)
    out["B_n_teste_normal"] = int(len(normais) - len(normais) // 2)

    # ---- Cenário C: todo o dataset, sem rótulos, 41% de ataques (cont. = proporção real)
    resC = []
    idx = np.arange(len(y))
    Xtr, _ = montar(base, caminhos, idx, idx)
    prop = float(y.mean())
    for s in range(5):
        pred, esc = treinar_auditoria_local(Xtr, Xtr, prop, s)
        resC.append(metricas(y, pred, esc))
    out["C"] = resumir(resC)
    out["C_prop_ataque"] = prop

    # ---- Tempo: extração + treino + predição no dataset completo (um cenário B típico)
    t1 = time.perf_counter()
    Xtr, Xte = montar(base, caminhos, normais[: len(normais) // 2], idx)
    treinar_auditoria_local(Xtr, Xte, 0.05, 0)
    out["tempo_modelo_s"] = time.perf_counter() - t1

    # ---- Importância descritiva: AUC de cada atributo isolado (cenário completo)
    Xall, _ = montar(base, caminhos, idx, idx)
    out["auc_por_atributo"] = {c: float(max(roc_auc_score(y, Xall[:, i]), 1 - roc_auc_score(y, Xall[:, i])))
                               for i, c in enumerate(COLUNAS)}
    json.dump(out, open("resultados_csic.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
