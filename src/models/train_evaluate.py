"""
Etapas 3.3 e 3.4 — Treinamento, tuning, persistência e validação contra baselines.

Protocolo:
  * Split 80/20 AGRUPADO por incidente (GroupShuffleSplit): candidatos do mesmo
    incidente nunca ficam em treino e teste ao mesmo tempo (evita vazamento).
  * Tuning: RandomizedSearchCV com GroupKFold(5) no conjunto de treino.
  * Avaliação pontual (por candidato): RMSE, MAE, R² e, com limiar
    LIMIAR_VIAVEL, Acurácia, Precisão, Recall, F1 e matriz de confusão
    (classe positiva = "conflito crítico").
  * Avaliação de decisão (por incidente): score realizado da opção escolhida
    por cada estratégia, regret vs. oráculo, Top-k e teste de Wilcoxon pareado.

Uso: python -m src.models.train_evaluate
"""
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform, wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV

from src import config
from src.baseline.greedy_baseline import avaliar_decisoes, escolher_por_score, estrategias_referencia
from src.utils.plot_style import AZUL, CORES, SEQ, TEXTO2, aplicar_estilo, salvar

RS = config.RANDOM_STATE
np.random.seed(RS)


def regra_v1(df: pd.DataFrame) -> np.ndarray:
    """Score determinístico original do projeto (v1), usado como preditor heurístico."""
    pen = (np.where(df["fator_ocupacao_sala"] > 1, (df["fator_ocupacao_sala"] - 1) * 40, 0)
           + df["indice_sobreposicao_trilha"] * 25 + df["horario_pico"] * 10)
    return np.clip(100 - pen, 0, 100)


def metricas_pontuais(nome, y, pred):
    yc, pc = (y < config.LIMIAR_VIAVEL).astype(int), (pred < config.LIMIAR_VIAVEL).astype(int)
    return {
        "modelo": nome,
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
        "R2": r2_score(y, pred),
        "Acuracia": accuracy_score(yc, pc),
        "Precisao": precision_score(yc, pc, zero_division=0),
        "Recall": recall_score(yc, pc, zero_division=0),
        "F1": f1_score(yc, pc, zero_division=0),
    }, confusion_matrix(yc, pc, labels=[0, 1])


def tunar(nome, est, dist, X, y, g, n_iter):
    t0 = time.time()
    busca = RandomizedSearchCV(est, dist, n_iter=n_iter, cv=GroupKFold(n_splits=5),
                               scoring="neg_root_mean_squared_error", random_state=RS,
                               n_jobs=-1, refit=True)
    busca.fit(X, y, groups=g)
    print(f"  {nome}: RMSE-CV = {-busca.best_score_:.3f} | {time.time() - t0:.0f}s | {busca.best_params_}")
    return busca


def bootstrap_ic(x, n=2000):
    rng = np.random.default_rng(RS)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def main():
    aplicar_estilo()
    df = pd.read_csv(config.PROCESSED_FILE)
    df["_idx"] = df.index
    X, y, g = df[config.FEATURES], df[config.TARGET], df[config.GROUP]

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=RS).split(X, y, g))
    Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], g.iloc[tr]
    Xte, yte = X.iloc[te], y.iloc[te]
    test = df.iloc[te].copy()
    print(f"Treino: {len(tr)} candidatos ({gtr.nunique()} incidentes) | "
          f"Teste: {len(te)} candidatos ({test[config.GROUP].nunique()} incidentes)")

    # ---------------- Tuning ----------------
    print("Ajuste de hiperparâmetros (RandomizedSearchCV + GroupKFold=5)...")
    rf = tunar("Random Forest", RandomForestRegressor(random_state=RS, n_jobs=1), {
        "n_estimators": randint(150, 400), "max_depth": [None, 12, 18, 24],
        "min_samples_leaf": randint(1, 20), "max_features": [0.4, 0.6, 0.8, 1.0],
    }, Xtr, ytr, gtr, n_iter=15)
    xg = tunar("XGBoost", xgb.XGBRegressor(random_state=RS, n_jobs=1, tree_method="hist"), {
        "n_estimators": randint(200, 800), "max_depth": randint(3, 10),
        "learning_rate": uniform(0.02, 0.18), "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 20),
        "reg_lambda": uniform(0.0, 5.0),
    }, Xtr, ytr, gtr, n_iter=30)
    modelos = {"Random Forest": rf.best_estimator_, "XGBoost": xg.best_estimator_}

    # ---------------- Avaliação pontual ----------------
    preds = {
        "Média (Dummy)": DummyRegressor().fit(Xtr, ytr).predict(Xte),
        "Regras v1 (manual)": regra_v1(Xte),
        # A gulosa trata qualquer slot livre como viável -> score 100 para todos
        "Baseline Gulosa": np.full(len(Xte), 100.0),
    }
    for nome, m in modelos.items():
        preds[nome] = m.predict(Xte)
    linhas, cms = [], {}
    for nome, p in preds.items():
        met, cm = metricas_pontuais(nome, yte.values, p)
        if nome == "Baseline Gulosa":
            met.update(RMSE=np.nan, MAE=np.nan, R2=np.nan)  # não produz score: só decisão
        linhas.append(met)
        cms[nome] = cm
    tab = pd.DataFrame(linhas)

    # ---------------- Avaliação de decisão ----------------
    for nome in modelos:
        test[f"pred_{nome}"] = preds[nome]
    estr = estrategias_referencia(test)
    for nome in modelos:
        estr[nome] = escolher_por_score(test, f"pred_{nome}")
    ordem = ["Aleatória", "Baseline Gulosa", "Menor sala que cabe", "Random Forest", "XGBoost", "Oráculo"]
    dec, por_inc = avaliar_decisoes(test, {k: estr[k] for k in ordem})
    melhor = min(modelos, key=lambda n: tab.set_index("modelo").loc[n, "RMSE"])
    ganho = por_inc[melhor] - por_inc["Baseline Gulosa"]
    ic = bootstrap_ic(ganho.values)
    stat = {n: wilcoxon(por_inc[melhor], por_inc[n]).pvalue for n in
            ["Baseline Gulosa", "Aleatória", "Menor sala que cabe"]}
    dec["p_wilcoxon_vs_" + melhor.replace(" ", "")] = dec["estrategia"].map(stat)

    tab.to_csv(config.RESULTS_FILE, index=False, float_format="%.4f")
    dec.to_csv(config.DECISION_RESULTS_FILE, index=False, float_format="%.4g")
    joblib.dump({"modelo": modelos[melhor], "nome": melhor, "features": config.FEATURES,
                 "limiar_viavel": config.LIMIAR_VIAVEL, "params": (rf if melhor == "Random Forest" else xg).best_params_},
                config.MODEL_FILE)
    print(f"\nMelhor modelo: {melhor} -> {config.MODEL_FILE}")

    # ---------------- Figuras ----------------
    # 1) Matrizes de confusão
    nomes_cm = ["Baseline Gulosa", "Random Forest", "XGBoost"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, n in zip(axes, nomes_cm):
        cm = cms[n]
        ax.imshow(cm, cmap=SEQ)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}\n({v / cm.sum():.0%})", ha="center", va="center", fontsize=9,
                    color="white" if v > cm.max() * 0.55 else "#0b0b0b")
        ax.set_xticks([0, 1], ["viável", "crítico"]); ax.set_yticks([0, 1], ["viável", "crítico"])
        ax.set(title=f"{n} (F1={tab.set_index('modelo').loc[n, 'F1']:.3f})", xlabel="previsto", ylabel="real")
    salvar(fig, config.FIG_DIR, "res_01_matrizes_confusao.png")

    # 2) Predito vs real
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, n in zip(axes, modelos):
        ax.hexbin(yte, preds[n], gridsize=45, cmap=SEQ, mincnt=1, bins="log")
        ax.plot([0, 100], [0, 100], color=TEXTO2, ls="--", lw=1)
        r = tab.set_index("modelo").loc[n]
        ax.set(title=f"{n}: RMSE={r.RMSE:.2f}, R²={r.R2:.3f}", xlabel="score real", ylabel="score predito")
    salvar(fig, config.FIG_DIR, "res_02_predito_vs_real.png")

    # 3) Métricas pontuais comparativas
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    t2 = tab.set_index("modelo")
    for ax, cols, tit in [(axes[0], ["RMSE", "MAE"], "Erro de regressão (menor é melhor)"),
                          (axes[1], ["F1", "Acuracia"], "Classificação de conflito crítico")]:
        ms = [m for m in t2.index if not t2.loc[m, cols].isna().any()]
        w = 0.8 / len(ms)
        for k, m in enumerate(ms):
            xs = np.arange(len(cols)) + k * w - 0.4 + w / 2
            b = ax.bar(xs, t2.loc[m, cols], w * 0.92, color=CORES[m], label=m)
            ax.bar_label(b, fmt="%.2f", fontsize=7, color=TEXTO2, padding=2)
        ax.set_xticks(range(len(cols)), cols); ax.set_title(tit)
        ax.grid(axis="x", visible=False)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    salvar(fig, config.FIG_DIR, "res_03_metricas_pontuais.png")

    # 4) Qualidade da decisão por incidente
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    d2 = dec.set_index("estrategia")
    cis = np.array([bootstrap_ic(por_inc[e].values) for e in ordem])
    medias = d2.loc[ordem, "score_realizado_medio"].values
    b = axes[0].barh(ordem, medias, color=[CORES[e] for e in ordem], height=0.6,
                     xerr=[medias - cis[:, 0], cis[:, 1] - medias], error_kw={"ecolor": TEXTO2, "lw": 1})
    axes[0].bar_label(b, fmt="%.1f", padding=6, fontsize=8, color=TEXTO2)
    axes[0].set(title="Score realizado médio da opção escolhida (IC 95%)", xlabel="score_viabilidade", xlim=(0, 105))
    axes[0].grid(axis="y", visible=False); axes[0].invert_yaxis()
    taxa = d2.loc[ordem, "taxa_conflito_critico"].values * 100
    b = axes[1].barh(ordem, taxa, color=[CORES[e] for e in ordem], height=0.6)
    axes[1].bar_label(b, fmt="%.1f%%", padding=4, fontsize=8, color=TEXTO2)
    axes[1].set(title=f"Incidentes resolvidos com conflito crítico (score < {config.LIMIAR_VIAVEL:.0f})",
                xlabel="% dos incidentes de teste")
    axes[1].grid(axis="y", visible=False); axes[1].invert_yaxis()
    salvar(fig, config.FIG_DIR, "res_04_decisao_estrategias.png")

    # 5) Ganho pareado ML - Gulosa por incidente
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ganho, bins=50, color=AZUL, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=TEXTO2, lw=1)
    ax.axvline(ganho.mean(), color=CORES["Baseline Gulosa"], ls="--", lw=1.5)
    ax.set(title=f"Ganho por incidente: {melhor} − Gulosa (média {ganho.mean():+.1f}, "
                 f"IC95% [{ic[0]:+.1f}, {ic[1]:+.1f}])", xlabel="Δ score realizado", ylabel="incidentes")
    salvar(fig, config.FIG_DIR, "res_05_ganho_pareado.png")

    # 6) Importância (contribuições SHAP nativas do XGBoost)
    contrib = xg.best_estimator_.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=config.FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b = ax.barh(imp.index, imp.values, color=AZUL, height=0.6)
    ax.bar_label(b, fmt="%.2f", padding=3, fontsize=8, color=TEXTO2)
    ax.set(title="Importância dos atributos — XGBoost (média |SHAP|)", xlabel="impacto médio no score predito")
    ax.grid(axis="y", visible=False)
    salvar(fig, config.FIG_DIR, "res_06_importancia_shap.png")

    # 7) Erro por faixa de ocupação
    faixas = pd.cut(test["fator_ocupacao_sala"], [0, 0.5, 0.8, 1.0, 1.25, 1.5, np.inf],
                    labels=["<0,5", "0,5–0,8", "0,8–1,0", "1,0–1,25", "1,25–1,5", ">1,5"])
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in modelos:
        err = (pd.Series(preds[n], index=test.index) - yte).abs().groupby(faixas, observed=False).mean()
        ax.plot(err.index.astype(str), err.values, marker="o", ms=6, color=CORES[n], label=n)
    ax.set(title="MAE por faixa de ocupação da sala", xlabel="fator de ocupação", ylabel="MAE")
    ax.legend()
    salvar(fig, config.FIG_DIR, "res_07_erro_por_ocupacao.png")

    # ---------------- Resumo ----------------
    pd.set_option("display.width", 200)
    print("\n=== Métricas pontuais (teste, por candidato) ===")
    print(tab.round(4).to_string(index=False))
    print("\n=== Qualidade da decisão (teste, por incidente) ===")
    print(dec.round(4).to_string(index=False))
    print(f"\nGanho médio {melhor} vs Gulosa: {ganho.mean():+.2f} pts (IC95% {ic.round(2)}); "
          f"vence em {(ganho > 0).mean():.1%}, empata em {(ganho == 0).mean():.1%} dos incidentes")


if __name__ == "__main__":
    main()
