"""
Comparação do Isolation Forest com outros algoritmos no CSIC 2010.

Requer experimento_csic.py na mesma pasta (reaproveita a extração de atributos).
Uso:  python experimento_comparacao.py [caminho_do_csv]
Saídas: resultados_comparacao.json e roc_curvas.png

Divisão (repetida com sementes diferentes, por grupos de requisições gêmeas): cerca de
metade do tráfego normal e metade dos ataques vai para o treino; a outra metade de cada um forma o teste, idêntico para
todos os modelos.
  - Isolation Forest, Local Outlier Factor e One-Class SVM: treinados SÓ com o
    tráfego normal do treino (sem usar ataques nem rótulos de ataque).
  - Random Forest (supervisionado): treinado com normais E ataques rotulados do
    treino; serve de referência de "teto" com acesso a rótulos.
Medidas: AUC-ROC, precisão média (AP) e recall com 5% de falsos positivos.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import experimento_csic as ec

CAMINHO = sys.argv[1] if len(sys.argv) > 1 else "csic_database.csv"
N_REP = 10
N_TREINO_OCSVM = 5000          # o One-Class SVM é quadrático; usa-se subamostra do treino


def recall_em_fpr(y, esc, fpr_alvo=0.05):
    fpr, tpr, _ = roc_curve(y, esc)
    return float(np.interp(fpr_alvo, fpr, tpr))


def avaliar(y, esc):
    return dict(auc=roc_auc_score(y, esc), ap=average_precision_score(y, esc),
                recall_fpr5=recall_em_fpr(y, esc))


def main():
    df, y = ec.carregar(CAMINHO)
    base, caminhos = ec.extrair_base(df)
    chaves = ec.chaves_grupo(df)
    tam = []
    normais, anom = np.where(y == 0)[0], np.where(y == 1)[0]
    modelos = ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Random Forest (supervisionado)"]
    res = {m: [] for m in modelos}
    tempos = {m: [] for m in modelos}
    roc0 = {}

    for s in range(N_REP):
        rng = np.random.default_rng(500 + s)
        # divisão por grupo: requisições gêmeas (GET/POST equivalentes) ficam do mesmo lado
        uni = np.unique(chaves); rng.shuffle(uni); sel = set(uni[: len(uni) // 2])
        no_tr = np.array([k in sel for k in chaves])
        tr_n, te_n = normais[no_tr[normais]], normais[~no_tr[normais]]
        tr_a, te_a = anom[no_tr[anom]], anom[~no_tr[anom]]
        tam.append((len(tr_n), len(tr_a), len(te_n), len(te_a)))
        idx_te = np.concatenate([te_n, te_a])
        yte = y[idx_te]

        # --- modelos treinados só com tráfego normal
        Xtr, Xte = ec.montar(base, caminhos, tr_n, idx_te)
        sc = StandardScaler().fit(Xtr)
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

        t = time.perf_counter()
        m = IsolationForest(n_estimators=100, random_state=s, n_jobs=-1).fit(Xtr)
        e_if = -m.decision_function(Xte); tempos[modelos[0]].append(time.perf_counter() - t)

        t = time.perf_counter()
        m = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(Ztr)
        e_lof = -m.decision_function(Zte); tempos[modelos[1]].append(time.perf_counter() - t)

        t = time.perf_counter()
        sub = rng.choice(len(Ztr), N_TREINO_OCSVM, replace=False)
        m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(Ztr[sub])
        e_svm = -m.decision_function(Zte); tempos[modelos[2]].append(time.perf_counter() - t)

        # --- supervisionado: treina com normais e ataques rotulados do treino
        idx_tr_sup = np.concatenate([tr_n, tr_a])
        Xtr_s, Xte_s = ec.montar(base, caminhos, idx_tr_sup, idx_te)
        t = time.perf_counter()
        m = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1).fit(Xtr_s, y[idx_tr_sup])
        e_rf = m.predict_proba(Xte_s)[:, 1]; tempos[modelos[3]].append(time.perf_counter() - t)

        for nome, esc in zip(modelos, [e_if, e_lof, e_svm, e_rf]):
            res[nome].append(avaliar(yte, esc))
        if s == 0:
            for nome, esc in zip(modelos, [e_if, e_lof, e_svm, e_rf]):
                f, tp, _ = roc_curve(yte, esc)
                roc0[nome] = (f, tp)

    resumo = {}
    for nome in modelos:
        d = pd.DataFrame(res[nome])
        resumo[nome] = {c: [float(d[c].mean()), float(d[c].std())] for c in d.columns}
        resumo[nome]["tempo_s"] = float(np.mean(tempos[nome]))
    t = np.array(tam).mean(axis=0)
    out = dict(n_repeticoes=N_REP, n_treino_normal=float(t[0]), n_treino_ataque_supervisionado=float(t[1]),
               n_teste_normal=float(t[2]), n_teste_ataque=float(t[3]), resumo=resumo)
    json.dump(out, open("resultados_comparacao.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

    # --- figura: curvas ROC (primeira repetição)
    plt.figure(figsize=(5.2, 4.4), dpi=200)
    estilos = ["-", "--", "-.", ":"]
    for (nome, (f, tp)), est in zip(roc0.items(), estilos):
        plt.plot(f, tp, est, color="black", lw=1.4, label=f"{nome} (AUC = {roc_auc_score_from(f, tp):.2f})".replace(".", ","))
    plt.plot([0, 1], [0, 1], color="gray", lw=0.8)
    plt.axvline(0.05, color="gray", lw=0.6, ls=":")
    plt.xlabel("Taxa de falsos positivos"); plt.ylabel("Taxa de verdadeiros positivos (recall)")
    plt.legend(loc="lower right", fontsize=7.5, frameon=False)
    plt.xlim(0, 1); plt.ylim(0, 1.01); plt.tight_layout()
    plt.savefig("roc_curvas.png")


def roc_auc_score_from(f, tp):
    return float(np.trapezoid(tp, f)) if hasattr(np, "trapezoid") else float(np.trapz(tp, f))


if __name__ == "__main__":
    main()
