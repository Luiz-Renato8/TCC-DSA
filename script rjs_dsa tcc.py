# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# ==============================================================

# ANÁLISE DE RECUPERAÇÕES JUDICIAIS NO AGRONEGÓCIO BRASILEIRO

# Impacto da Lei 14.112/2020 e Determinantes Setoriais

# TCC — MBA em Data Science & Analytics

# ==============================================================

 

 

# ==============================================================

# SEÇÃO 0 — IMPORTAÇÕES

# Todas as bibliotecas utilizadas no script, agrupadas por função

# ==============================================================

 

# Manipulação de dados

import pandas as pd

import numpy as np

 

# Modelagem estatística e econométrica

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.nonparametric.smoothers_lowess import lowess

 

# Visualização

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from matplotlib.patches import Patch

import seaborn as sns

 

# Testes estatísticos

from scipy import stats

 

# Configurações gerais

import warnings

warnings.filterwarnings("ignore")

 

# Paleta de cores padronizada para todas as visualizações

COR_OBSERVADO = "#2C6FAC"   # azul escuro — série observada

COR_AJUSTADO  = "#E07B39"   # laranja — série ajustada

COR_LEI       = "#C0392B"   # vermelho — linha da lei

COR_RESIDUO_P = "#2C6FAC"   # azul — resíduo positivo

COR_RESIDUO_N = "#E07B39"   # laranja — resíduo negativo

COR_MA1       = "#2C6FAC"   # azul — modelos do painel

COR_MA4       = "#27AE60"   # verde — modelos do agro

 

 

# ==============================================================

# SEÇÃO 1 — CARREGAMENTO E INSPEÇÃO INICIAL DOS DADOS

# Objetivo: carregar o dataset e verificar sua estrutura básica

# ==============================================================

 

df = pd.read_excel(r"C:\Users\PC\OneDrive\Desktop\DSA TCC\database_rj setorial.xlsx", engine="openpyxl")

 

# Padronizar nomes de colunas: minúsculas e sem espaços

df.columns = df.columns.str.lower().str.replace(" ", "_")

 

# Ordenar por setor e ano para garantir a ordem correta dos lags

df = df.sort_values(["setor", "ano"]).reset_index(drop=True)

 

print("=" * 55)

print("INSPEÇÃO INICIAL DOS DADOS")

print("=" * 55)

print(f"Dimensões: {df.shape[0]} linhas × {df.shape[1]} colunas")

print(f"Setores: {df['setor'].unique()}")

print(f"Período: {df['ano'].min()} a {df['ano'].max()}")

print("\nPrimeiras linhas:")

print(df.head())

print("\nTipos de dados:")

print(df.dtypes)

 

 

# ==============================================================

# SEÇÃO 2 — LIMPEZA E PADRONIZAÇÃO

# Objetivo: garantir que todas as colunas estejam no tipo

# numérico correto antes de qualquer transformação

# ==============================================================

 

# Converter Selic de string percentual para decimal numérico

# Exemplo: "10,75%" → 0.1075

df["selic"] = (

    df["selic"]

    .astype(str)

    .str.replace(",", ".", regex=False)

    .str.replace("%", "", regex=False)

    .astype(float) / 100

)

 

# Converter colunas que podem estar como string com vírgula decimal

for col in ["tx_cambio", "preço_soja", "preço_milho"]:

    df[col] = (

        df[col]

        .astype(str)

        .str.strip()

        .str.replace(",", ".", regex=False)

        .replace("", np.nan)

        .astype(float)

    )

 

print("\n\nTipos após limpeza:")

print(df[["selic", "tx_cambio", "preço_soja", "preço_milho"]].dtypes)

 

 

# ==============================================================

# SEÇÃO 3 — ENGENHARIA DE VARIÁVEIS

# Objetivo: criar todas as variáveis analíticas necessárias

# para a modelagem. Dividido em 4 etapas lógicas:

#   3.1 Variável dependente

#   3.2 Variáveis de identificação (dummies e interações)

#   3.3 Variáveis setoriais específicas do agro

#   3.4 Variáveis defasadas (lags)

# O dropna() é executado UMA ÚNICA VEZ ao final,

# após todas as variáveis terem sido criadas.

# ==============================================================

 

# ── 3.1 Variável dependente ──────────────────────────────────

# Log do número de CNPJs em recuperação judicial por setor/ano

# A transformação logarítmica reduz a assimetria da distribuição

# e permite interpretar coeficientes como semi-elasticidades

df["log_rj"] = np.log(df["qtd_cnpjs"])

 

# ── 3.2 Dummies de identificação ─────────────────────────────

# Dummy de grupo de tratamento: agro = 1, demais setores = 0

df["agro"] = (df["setor"] == "agro").astype(int)

 

# Dummy de período pós-lei: anos a partir de 2021 = 1

# A Lei 14.112/2020 foi promulgada em dezembro de 2020

# e seus efeitos práticos começam a partir de 2021

df["pos_lei"] = (df["ano"] >= 2021).astype(int)

 

# Variável de Diferenças-em-Diferenças (DID):

# interação entre grupo de tratamento e período pós-lei

# Captura o efeito da lei EXCLUSIVAMENTE para o setor agro

# após sua vigência

df["did"] = df["agro"] * df["pos_lei"]

 

# ── 3.3 Variáveis setoriais do agro ──────────────────────────

# Interações que ativam preços de commodities e crédito

# apenas para o setor agro (zero para indústria e serviços)

df["soja_agro"]         = df["preço_soja"]         * df["agro"]

df["milho_agro"]        = df["preço_milho"]         * df["agro"]

df["credito_agro_real"] = df["credito_concedido"]   * df["agro"]

 

# Log do crédito: np.log1p(x) = log(1+x) evita log(0)

# quando o crédito é zero para setores não-agro

df["log_credito"] = np.log1p(df["credito_agro_real"])

 

# ── 3.4 Variáveis defasadas (lags por setor) ─────────────────

# Lags calculados DENTRO de cada setor (groupby)

# para evitar que o último ano de um setor vire

# o "ano anterior" do próximo setor na ordenação

df["selic_lag1"]        = df.groupby("setor")["selic"].shift(1)

df["pib_lag1"]          = df.groupby("setor")["var_pib"].shift(1)

df["credito_lag1"]      = df.groupby("setor")["credito_agro_real"].shift(1)

df["log_credito_lag1"]  = df.groupby("setor")["log_credito"].shift(1)

 

# ── Remoção de valores ausentes ──────────────────────────────

# Executado UMA VEZ após todas as variáveis terem sido criadas

# Os NaN surgem principalmente do primeiro ano de cada setor

# (sem valor defasado disponível)

df = df.dropna().reset_index(drop=True)

 

print("=" * 55)

print("DATASET FINAL APÓS ENGENHARIA DE VARIÁVEIS")

print("=" * 55)

print(f"Observações: {len(df)} ({df['setor'].nunique()} setores × ~{len(df)//df['setor'].nunique()} anos)")

print(f"\nEstatísticas descritivas:")

print(df[[

    "log_rj", "selic_lag1", "pib_lag1", "tx_cambio",

    "soja_agro", "milho_agro", "log_credito_lag1", "did"

]].describe().round(4))

 

 

# ==============================================================

# SEÇÃO 4 — ANÁLISE EXPLORATÓRIA: PAINEL COMPLETO

# Objetivo: examinar correlações entre variáveis no painel

# com os 3 setores, identificando relações brutas e possíveis

# problemas de multicolinearidade antes da modelagem

# ==============================================================

 

cols_corr_painel = [

    "log_rj", "selic_lag1", "pib_lag1", "tx_cambio",

    "soja_agro", "milho_agro", "log_credito_lag1", "did"

]

 

corr_painel = df[cols_corr_painel].corr().round(3)

 

print("\n\nMATRIZ DE CORRELAÇÃO — PAINEL COMPLETO")

print(corr_painel)

 

plt.figure(figsize=(10, 7))

sns.heatmap(corr_painel, annot=True, cmap="coolwarm",

            fmt=".2f", linewidths=0.5, square=True)

plt.title("Matriz de Correlação — Painel Completo (3 Setores)",

          fontsize=13, fontweight="bold")

plt.tight_layout()

plt.savefig("eda_correlacao_painel.png", dpi=150, bbox_inches="tight")

plt.show()

 

 

# ==============================================================

# SEÇÃO 5 — DIAGNÓSTICO DE MULTICOLINEARIDADE: PAINEL

# Objetivo: calcular o Fator de Inflação da Variância (VIF)

# para o conjunto de regressores do modelo de painel.

# VIF > 10 indica multicolinearidade severa.

# ==============================================================

 

X_vif_painel = df[[

    "selic_lag1", "pib_lag1", "tx_cambio",

    "log_credito_lag1", "did"

]]

X_vif_painel_const = sm.add_constant(X_vif_painel)

 

vif_painel = pd.DataFrame({

    "variavel": X_vif_painel_const.columns,

    "VIF": [

        variance_inflation_factor(X_vif_painel_const.values, i)

        for i in range(X_vif_painel_const.shape[1])

    ]

})

 

print("\n\nVIF — PAINEL COMPLETO")

print(vif_painel.round(3))

 

 

# ==============================================================

# SEÇÃO 6 — MODELAGEM: PAINEL COMPLETO

# Objetivo: estimar o modelo de regressão com os 3 setores,

# testando três estruturas de erros padrão:

#   M1 — OLS padrão (baseline, sem correção)

#   M2 — HC3 (robusto à heterocedasticidade)

#   M3 — HAC/Newey-West (robusto à heterocedasticidade

#         e autocorrelação) → modelo principal

#

# A especificação inclui efeitos fixos de setor via C(setor),

# controlando diferenças estruturais permanentes entre setores.

# A variável DID captura o efeito da Lei 14.112/2020

# especificamente para o agro no período pós-vigência.

# ==============================================================

 

formula_painel = (

    "log_rj ~ selic_lag1 + pib_lag1 + tx_cambio "

    "+ log_credito_lag1 + did + C(setor)"

)

 

# M1 — OLS padrão (baseline)

m1 = smf.ols(formula_painel, data=df).fit()

 

# M2 — Erros HC3 (robusto à heterocedasticidade)

# Nota: instável em amostras pequenas com observações de alta alavancagem

m2 = smf.ols(formula_painel, data=df).fit(cov_type="HC3")

 

# M3 — Erros HAC / Newey-West (modelo principal)

# maxlags=2 é escolha conservadora para séries anuais curtas

m3 = smf.ols(formula_painel, data=df).fit(

    cov_type="HAC", cov_kwds={"maxlags": 2}

)

 

# ── Sumários individuais ──

for nome, mod in [("M1 — OLS Padrão (baseline)", m1),

                  ("M2 — Erros HC3", m2),

                  ("M3 — Erros HAC / Newey-West [MODELO PRINCIPAL]", m3)]:

    print("\n" + "=" * 65)

    print(nome)

    print("=" * 65)

    print(mod.summary())

 

# ── Tabela comparativa de coeficientes e p-valores ──

def sig_stars(p):

    if p < 0.01:   return "***"

    elif p < 0.05: return "**"

    elif p < 0.10: return "*"

    else:          return "n.s."

 

variaveis_painel = [

    "Intercept",

    "C(setor)[T.industria]",

    "C(setor)[T.serviços]",

    "selic_lag1", "pib_lag1", "tx_cambio",

    "log_credito_lag1", "did"

]

 

print("\n\n" + "=" * 90)

print("TABELA COMPARATIVA — PAINEL COMPLETO: M1 vs M2 (HC3) vs M3 (HAC)")

print("=" * 90)

print(f"{'Variável':<30} {'Coef':>8} | {'p (M1)':>8} {'p (HC3)':>8} "

      f"{'p (HAC)':>8} | {'M1':>6} {'HC3':>7} {'HAC':>7}")

print("=" * 90)

 

for var in variaveis_painel:

    coef  = m1.params[var]

    p_m1  = m1.pvalues[var]

    p_hc3 = m2.pvalues[var]

    p_hac = m3.pvalues[var]

    print(

        f"{var:<30} {coef:>8.4f} | {p_m1:>8.3f} {p_hc3:>8.3f} {p_hac:>8.3f} | "

        f"{sig_stars(p_m1):>6} {sig_stars(p_hc3):>7} {sig_stars(p_hac):>7}"

    )

 

print("=" * 90)

print("Legenda: *** p<0.01 | ** p<0.05 | * p<0.10 | n.s. = não significativo")

print("Nota: M3-HAC é o modelo principal. M2-HC3 é instável nesta amostra.")

 

# ── Gráfico comparativo de coeficientes e IC 95% ──

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

vars_plot_painel = [

    "selic_lag1", "pib_lag1", "tx_cambio", "log_credito_lag1", "did"

]

 

for ax, (mod, titulo, cor) in zip(axes, [

    (m1, "M1 — OLS Padrão",      "steelblue"),

    (m2, "M2 — HC3",             "darkorange"),

    (m3, "M3 — HAC [Principal]", "seagreen")

]):

    coefs  = mod.params[vars_plot_painel]

    ci_low = mod.conf_int().loc[vars_plot_painel, 0]

    ci_up  = mod.conf_int().loc[vars_plot_painel, 1]

    erros  = [coefs - ci_low, ci_up - coefs]

 

    ax.barh(vars_plot_painel, coefs, xerr=erros,

            color=cor, alpha=0.7, capsize=5,

            error_kw={"elinewidth": 1.5, "ecolor": "black"})

    ax.axvline(0, color=COR_LEI, linestyle="--", linewidth=1)

    ax.set_title(titulo, fontsize=11, fontweight="bold")

    ax.set_xlabel("Coeficiente")

    ax.grid(True, alpha=0.3, axis="x")

 

axes[0].set_ylabel("Variável")

plt.suptitle("Coeficientes e IC 95% — Painel Completo: M1 vs HC3 vs HAC",

             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()

plt.savefig("painel_coeficientes_comparativo.png", dpi=150, bbox_inches="tight")

plt.show()

 

 

# ==============================================================

# SEÇÃO 7 — ANÁLISE EXPLORATÓRIA: SETOR AGRO

# Objetivo: repetir a análise exploratória isolando apenas

# o setor agro, onde variáveis como preço de soja, milho

# e crédito têm variabilidade real (não são zero)

# ==============================================================

 

df_agro = df[df["setor"] == "agro"].copy().reset_index(drop=True)

 

print("=" * 55)

print("DATASET — SETOR AGRO")

print("=" * 55)

print(f"Observações: {len(df_agro)}")

print(f"Período: {df_agro['ano'].min()} a {df_agro['ano'].max()}")

print(df_agro[[

    "ano", "qtd_cnpjs", "log_rj", "selic_lag1", "pib_lag1",

    "tx_cambio", "soja_agro", "milho_agro",

    "log_credito_lag1", "pos_lei"

]].to_string(index=False))

 

print("\n\nESTATÍSTICAS DESCRITIVAS — AGRO")

print(df_agro[[

    "log_rj", "selic_lag1", "pib_lag1", "tx_cambio",

    "soja_agro", "milho_agro", "log_credito_lag1", "pos_lei"

]].describe().round(4))

 

# ── Matriz de correlação do agro ──

cols_corr_agro = [

    "log_rj", "selic_lag1", "pib_lag1", "tx_cambio",

    "soja_agro", "milho_agro", "log_credito_lag1", "pos_lei"

]

 

corr_agro = df_agro[cols_corr_agro].corr().round(3)

 

print("\n\nMATRIZ DE CORRELAÇÃO — AGRO")

print(corr_agro)

 

plt.figure(figsize=(9, 7))

sns.heatmap(corr_agro, annot=True, cmap="coolwarm",

            fmt=".2f", linewidths=0.5, square=True)

plt.title("Matriz de Correlação — Setor Agro",

          fontsize=13, fontweight="bold")

plt.tight_layout()

plt.savefig("eda_correlacao_agro.png", dpi=150, bbox_inches="tight")

plt.show()

 

 

# ==============================================================

# SEÇÃO 8 — DIAGNÓSTICO DE MULTICOLINEARIDADE: AGRO

# Objetivo: calcular VIF para as variáveis candidatas

# do modelo específico do agro, identificando quais

# combinações de variáveis geram instabilidade numérica

# ==============================================================

 

X_vif_agro = df_agro[[

    "selic_lag1", "pib_lag1", "tx_cambio",

    "soja_agro", "log_credito_lag1", "pos_lei"

]].dropna()

 

X_vif_agro_const = sm.add_constant(X_vif_agro)

 

vif_agro = pd.DataFrame({

    "variavel": X_vif_agro_const.columns,

    "VIF": [

        variance_inflation_factor(X_vif_agro_const.values, i)

        for i in range(X_vif_agro_const.shape[1])

    ]

})

 

print("\n\nVIF — SETOR AGRO")

print(vif_agro.round(3))

print("\nNota: soja_agro apresenta VIF elevado (>10) devido à correlação")

print("com tx_cambio e pos_lei. Resultado esperado dado o confundimento")

print("temporal entre câmbio alto, alta de commodities e período pós-lei.")

 

 

# ==============================================================

# SEÇÃO 9 — MODELAGEM: SETOR AGRO

# Objetivo: estimar modelos específicos para o agro,

# incorporando variáveis setoriais (commodities, crédito)

# que são zero para os demais setores no painel.

# Quatro especificações progressivas:

#   MA1 — Apenas variáveis macro (baseline agro)

#   MA2 — Macro + preços de commodities

#   MA3 — Macro + crédito rural

#   MA4 — Macro + soja + crédito + dummy lei [modelo principal agro]

# Todos estimados com erros HAC (maxlags=1, dado N=13)

# ==============================================================

 

# MA1 — Baseline: apenas variáveis macroeconômicas

ma1 = smf.ols(

    "log_rj ~ selic_lag1 + pib_lag1 + tx_cambio",

    data=df_agro

).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

 

# MA2 — Macro + preços de commodities

# Objetivo: testar se soja e milho têm poder explicativo

# além do câmbio (com o qual são correlacionados)

ma2 = smf.ols(

    "log_rj ~ selic_lag1 + pib_lag1 + tx_cambio + soja_agro + milho_agro",

    data=df_agro

).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

 

# MA3 — Macro + crédito rural

# Objetivo: testar o papel do crédito como fator protetor

ma3 = smf.ols(

    "log_rj ~ selic_lag1 + pib_lag1 + tx_cambio + log_credito_lag1",

    data=df_agro

).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

 

# MA4 — Modelo mais completo: macro + soja + crédito + lei

# Modelo principal do agro: captura efeito da lei controlando

# por câmbio, rentabilidade (soja) e acesso a crédito

ma4 = smf.ols(

    "log_rj ~ selic_lag1 + pib_lag1 + tx_cambio "

    "+ soja_agro + log_credito_lag1 + pos_lei",

    data=df_agro

).fit(cov_type="HAC", cov_kwds={"maxlags": 1})

 

# ── Sumários individuais ──

modelos_agro = {"MA1": ma1, "MA2": ma2, "MA3": ma3, "MA4": ma4}

 

for nome, mod in modelos_agro.items():

    print("\n" + "=" * 65)

    print(f"MODELO {nome}" +

          (" [MODELO PRINCIPAL AGRO]" if nome == "MA4" else ""))

    print("=" * 65)

    print(mod.summary())

 

# ── Tabela comparativa — agro ──

todas_vars_agro = [

    "Intercept", "selic_lag1", "pib_lag1", "tx_cambio",

    "soja_agro", "milho_agro", "log_credito_lag1", "pos_lei"

]

 

print("\n\n" + "=" * 95)

print("TABELA COMPARATIVA — MODELOS AGRO: MA1 a MA4")

print("=" * 95)

print(f"{'Variável':<25} "

      f"{'MA1 coef':>10} {'p':>7} | "

      f"{'MA2 coef':>10} {'p':>7} | "

      f"{'MA3 coef':>10} {'p':>7} | "

      f"{'MA4 coef':>10} {'p':>7}")

print("=" * 95)

 

for var in todas_vars_agro:

    linha = f"{var:<25}"

    for nome, mod in modelos_agro.items():

        if var in mod.params:

            c = mod.params[var]

            p = mod.pvalues[var]

            linha += f" {c:>10.4f} {sig_stars(p):>7} |"

        else:

            linha += f" {'---':>10} {'---':>7} |"

    print(linha)

 

print("=" * 95)

print("Legenda: *** p<0.01 | ** p<0.05 | * p<0.10 | n.s. = não significativo")

 

# ── Métricas de ajuste ──

print("\n\nMÉTRICAS DE AJUSTE — MODELOS AGRO")

print("-" * 55)

print(f"{'Modelo':<8} {'R²':>8} {'R² Adj':>10} {'AIC':>10} {'BIC':>10}")

print("-" * 55)

for nome, mod in modelos_agro.items():

    print(f"{nome:<8} {mod.rsquared:>8.4f} {mod.rsquared_adj:>10.4f} "

          f"{mod.aic:>10.4f} {mod.bic:>10.4f}")

print("-" * 55)

print("Critério de seleção: menor AIC e maior R² Ajustado → MA4")

 

 

# ==============================================================

# SEÇÃO 10 — DIAGNÓSTICO DOS MODELOS PRINCIPAIS

# Objetivo: verificar as premissas dos modelos selecionados

# (M3-HAC e MA4) por meio de:

#   - Resíduos ao longo do tempo (padrão temporal)

#   - QQ-Plot (normalidade dos resíduos)

#   - Resíduo vs Fitted (homocedasticidade)

# ==============================================================

 

# ── Resíduos ao longo do tempo ──

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

 

for ax, (nome, mod, dados) in zip(axes, [

    ("MA1 — Agro",      ma1, df_agro),

    ("MA4 — Agro",      ma4, df_agro),

    ("M3 HAC — Painel", m3,  df)

]):

    res  = mod.resid

    anos = dados["ano"]

    cores_bar = [COR_RESIDUO_P if r >= 0 else COR_RESIDUO_N for r in res]

 

    ax.bar(anos, res, color=cores_bar, alpha=0.8,

           edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=1.2)

    ax.axvline(2021, color=COR_LEI, linestyle="--",

               linewidth=1.5, label="Lei 14.112/2020")

 

    z     = np.polyfit(anos, res, 1)

    p_fit = np.poly1d(z)

    ax.plot(sorted(anos), p_fit(sorted(anos)),

            color="gray", linestyle="-.", linewidth=1.2,

            label="Tendência linear")

 

    ax.set_title(f"Resíduos — {nome}", fontsize=11, fontweight="bold")

    ax.set_xlabel("Ano")

    ax.set_ylabel("Resíduo")

    ax.legend(fontsize=8)

    ax.grid(True, alpha=0.3, axis="y")

 

plt.suptitle("Diagnóstico — Resíduos ao Longo do Tempo",

             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()

plt.savefig("diag_residuos_tempo.png", dpi=150, bbox_inches="tight")

plt.show()

 

# ── QQ-Plot + Resíduo vs Fitted ──

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

 

for row, (nome, mod) in enumerate([

    ("MA4 — Agro",      ma4),

    ("M3 HAC — Painel", m3)

]):

    res    = mod.resid

    fitted = mod.fittedvalues

 

    # QQ-Plot

    ax = axes[row, 0]

    (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm")

    ax.scatter(osm, osr, color=COR_OBSERVADO, alpha=0.8, s=50, zorder=3)

    linha_x = np.array([min(osm), max(osm)])

    ax.plot(linha_x, slope * linha_x + intercept,

            color=COR_LEI, linewidth=1.5, linestyle="--")

    ax.annotate(f"R = {r:.3f}", xy=(0.05, 0.92),

                xycoords="axes fraction", fontsize=9,

                bbox=dict(boxstyle="round,pad=0.3",

                          facecolor="lightyellow"))

    ax.set_title(f"QQ-Plot — {nome}", fontsize=10, fontweight="bold")

    ax.set_xlabel("Quantis Teóricos (Normal)")

    ax.set_ylabel("Quantis Observados")

    ax.grid(True, alpha=0.3)

 

    # Resíduo vs Fitted

    ax = axes[row, 1]

    ax.scatter(fitted, res, color=COR_OBSERVADO,

               alpha=0.8, s=60, zorder=3)

    ax.axhline(0, color=COR_LEI, linestyle="--", linewidth=1.5)

    try:

        smooth = lowess(res, fitted, frac=0.6)

        ax.plot(smooth[:, 0], smooth[:, 1],

                color="darkorange", linewidth=2, label="LOWESS")

        ax.legend(fontsize=8)

    except Exception:

        pass

    ax.set_title(f"Resíduo vs Fitted — {nome}",

                 fontsize=10, fontweight="bold")

    ax.set_xlabel("Valores Ajustados")

    ax.set_ylabel("Resíduo")

    ax.grid(True, alpha=0.3)

 

plt.suptitle("Diagnóstico — QQ-Plot e Resíduo vs Fitted",

             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()

plt.savefig("diag_qqplot_fitted.png", dpi=150, bbox_inches="tight")

plt.show()

 

 

# ==============================================================

# SEÇÃO 11 — VISUALIZAÇÕES FINAIS PARA O TCC

# Objetivo: gerar as figuras de apresentação dos resultados,

# organizadas em 4 blocos temáticos:

#   Bloco A — Evolução temporal por setor

#   Bloco B — Coeficientes com IC 95% (modelos principais)

#   Bloco C — Contrafactual: efeito estimado da lei

#   Bloco D — Comparação de métricas entre modelos

# ==============================================================

 

# ── Bloco A — Evolução temporal ──────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

 

ax = axes[0]

for setor, cor, marker in [

    ("agro",      COR_OBSERVADO, "o"),

    ("industria", "#8E44AD",     "s"),

    ("serviços",  "#16A085",     "^")

]:

    dados_s = df[df["setor"] == setor]

    ax.plot(dados_s["ano"], dados_s["log_rj"],

            marker=marker, color=cor, label=setor.capitalize(),

            linewidth=2)

ax.axvline(2021, color=COR_LEI, linestyle="--",

           linewidth=1.5, label="Lei 14.112/2020")

ax.set_title("Evolução de log(RJ) por Setor",

             fontsize=11, fontweight="bold")

ax.set_xlabel("Ano")

ax.set_ylabel("log(qtd_cnpjs)")

ax.legend(fontsize=8)

ax.grid(True, alpha=0.3)

 

for ax, (nome, mod, cor) in zip(axes[1:], [

    ("MA1 — Baseline Agro", ma1, COR_MA1),

    ("MA4 — Principal Agro", ma4, COR_MA4)

]):

    ax.plot(df_agro["ano"], df_agro["log_rj"],

            marker="o", color=COR_OBSERVADO,

            label="Observado", linewidth=2)

    ax.plot(df_agro["ano"], mod.fittedvalues,

            marker="s", color=cor, label=f"Ajustado {nome[:3]}",

            linewidth=2, linestyle="--", alpha=0.8)

    ax.axvline(2021, color=COR_LEI, linestyle=":",

               linewidth=1.5, label="Lei 14.112/2020")

    ax.set_title(f"Agro — Observado vs Ajustado ({nome[:3]})",

                 fontsize=11, fontweight="bold")

    ax.set_xlabel("Ano")

    ax.set_ylabel("log(qtd_cnpjs)")

    ax.legend(fontsize=8)

    ax.grid(True, alpha=0.3)

 

plt.suptitle("Evolução Temporal: Observado vs Ajustado por Setor",

             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()

plt.savefig("fig_evolucao_temporal.png", dpi=150, bbox_inches="tight")

plt.show()

 

# ── Bloco B — Coeficientes com IC 95% ────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

 

configs_coef = [

    (m3,  "M3 HAC — Painel Completo",

     ["selic_lag1", "pib_lag1", "tx_cambio", "log_credito_lag1", "did"],

     ["Selic (lag1)", "PIB (lag1)", "Câmbio", "Log Crédito (lag1)", "DID (Lei)"]),

    (ma4, "MA4 HAC — Setor Agro",

     ["selic_lag1", "pib_lag1", "tx_cambio",

      "soja_agro", "log_credito_lag1", "pos_lei"],

     ["Selic (lag1)", "PIB (lag1)", "Câmbio",

      "Preço Soja", "Log Crédito (lag1)", "Dummy Lei"])

]

 

for ax, (mod, titulo, vars_sel, labels) in zip(axes, configs_coef):

    coefs  = mod.params[vars_sel].values

    ci_low = mod.conf_int().loc[vars_sel, 0].values

    ci_up  = mod.conf_int().loc[vars_sel, 1].values

    erros  = np.array([coefs - ci_low, ci_up - coefs])

    pvals  = mod.pvalues[vars_sel].values

 

    bar_cores = [

        "#1A5276" if p < 0.01 else

        "#2E86C1" if p < 0.05 else

        "#85C1E9" if p < 0.10 else

        "#BDC3C7"

        for p in pvals

    ]

 

    ax.barh(labels, coefs, xerr=erros, color=bar_cores,

            alpha=0.85, capsize=5,

            error_kw={"elinewidth": 1.5, "ecolor": "black"})

    ax.axvline(0, color=COR_LEI, linestyle="--", linewidth=1.5)

    ax.set_title(titulo, fontsize=11, fontweight="bold")

    ax.set_xlabel("Coeficiente (escala log)")

    ax.grid(True, alpha=0.3, axis="x")

 

    for i, (c, p) in enumerate(zip(coefs, pvals)):

        stars = ("***" if p < 0.01 else "**" if p < 0.05

                 else "*" if p < 0.10 else "")

        if stars:

            offset = max(erros[1][i] * 0.1, 0.05)

            ax.text(c + erros[1][i] + offset, i, stars,

                    va="center", fontsize=10,

                    color="#1A5276", fontweight="bold")

 

legenda_sig = [

    Patch(color="#1A5276", label="p < 0.01 (***)"),

    Patch(color="#2E86C1", label="p < 0.05 (**)"),

    Patch(color="#85C1E9", label="p < 0.10 (*)"),

    Patch(color="#BDC3C7", label="n.s.")

]

fig.legend(handles=legenda_sig, loc="lower center", ncol=4,

           fontsize=9, bbox_to_anchor=(0.5, -0.05),

           frameon=True, title="Nível de Significância")

 

plt.suptitle("Coeficientes com Intervalos de Confiança 95%",

             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()

plt.savefig("fig_coeficientes_ic.png", dpi=150, bbox_inches="tight")

plt.show()

 

# ── Bloco C — Contrafactual: efeito da lei ───────────────────

fig, ax = plt.subplots(figsize=(11, 6))

 

rj_obs     = df_agro["log_rj"]

fitted_ma1 = ma1.fittedvalues

fitted_ma4 = ma4.fittedvalues

 

ax.plot(df_agro["ano"], rj_obs,

        marker="o", color=COR_OBSERVADO, linewidth=2.5,

        label="Observado", zorder=5)

ax.plot(df_agro["ano"], fitted_ma1,

        marker="s", color="gray", linewidth=2,

        linestyle="--", alpha=0.7,

        label="Contrafactual (MA1 — sem lei)")

ax.plot(df_agro["ano"], fitted_ma4,

        marker="^", color=COR_MA4, linewidth=2,

        linestyle="-.", alpha=0.9,

        label="Ajustado MA4 (com lei)")

ax.fill_between(df_agro["ano"], fitted_ma1, rj_obs,

                where=(df_agro["ano"] >= 2021),

                alpha=0.15, color=COR_LEI,

                label="Efeito estimado da lei")

ax.axvline(2021, color=COR_LEI, linestyle=":",

           linewidth=2, label="Vigência Lei 14.112/2020")

 

lei_ef_pct = round((np.exp(ma4.params["pos_lei"]) - 1) * 100)

ax.annotate(

    f"pos_lei = +{ma4.params['pos_lei']:.2f}\n(+{lei_ef_pct}% RJs)",

    xy=(2022, df_agro[df_agro["ano"] == 2022]["log_rj"].values[0]),

    xytext=(2019.2, 6.0),

    fontsize=10, color=COR_LEI, fontweight="bold",

    arrowprops=dict(arrowstyle="->", color=COR_LEI, lw=1.5)

)

 

ax.set_title("Efeito da Lei 14.112/2020: Observado vs Contrafactual (Agro)",

             fontsize=12, fontweight="bold")

ax.set_xlabel("Ano", fontsize=11)

ax.set_ylabel("log(qtd_cnpjs)", fontsize=11)

ax.legend(fontsize=9, loc="upper left")

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("fig_contrafactual_lei.png", dpi=150, bbox_inches="tight")

plt.show()

 

# ── Bloco D — Comparação de métricas ─────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

 

nomes_mod  = ["M1\n(Painel)", "M3-HAC\n(Painel)",

              "MA1\n(Agro)",  "MA3\n(Agro)", "MA4\n(Agro)"]

r2_vals    = [m1.rsquared,     m3.rsquared,

              ma1.rsquared,    ma3.rsquared,  ma4.rsquared]

r2adj_vals = [m1.rsquared_adj, m3.rsquared_adj,

              ma1.rsquared_adj, ma3.rsquared_adj, ma4.rsquared_adj]

aic_vals   = [m1.aic, m3.aic, ma1.aic, ma3.aic, ma4.aic]

 

cores_mod = ["#2C6FAC", "#2C6FAC", "#27AE60", "#27AE60", "#27AE60"]

hatch_mod = ["", "//", "", "//", "xx"]

 

for ax, (vals, titulo, ylabel) in zip(axes, [

    (r2_vals,    "R²",                  "R²"),

    (r2adj_vals, "R² Ajustado",         "R² Ajustado"),

    (aic_vals,   "AIC (menor = melhor)", "AIC")

]):

    bars = ax.bar(nomes_mod, vals, color=cores_mod,

                  alpha=0.8, edgecolor="white", linewidth=1.2)

    for bar, h in zip(bars, hatch_mod):

        bar.set_hatch(h)

    for bar, val in zip(bars, vals):

        offset = 0.01 if titulo != "AIC (menor = melhor)" else 0.3

        ax.text(bar.get_x() + bar.get_width() / 2,

                bar.get_height() + offset,

                f"{val:.3f}" if titulo != "AIC (menor = melhor)"

                else f"{val:.1f}",

                ha="center", va="bottom",

                fontsize=9, fontweight="bold")

    ax.set_title(titulo, fontsize=11, fontweight="bold")

    ax.set_ylabel(ylabel)

    if titulo != "AIC (menor = melhor)":

        ax.set_ylim(0, 1.05)

    ax.grid(True, alpha=0.3, axis="y")

 

leg_grupos = [

    Patch(color="#2C6FAC", label="Modelos — Painel Completo"),

    Patch(color="#27AE60", label="Modelos — Setor Agro")

]

fig.legend(handles=leg_grupos, loc="lower center", ncol=2,

           fontsize=9, bbox_to_anchor=(0.5, -0.08))

 

plt.suptitle("Comparação de Métricas entre Modelos",

             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()

plt.savefig("fig_metricas_modelos.png", dpi=150, bbox_inches="tight")

plt.show()

 

 

# ==============================================================

# SEÇÃO 12 — TABELA RESUMO FINAL

# Síntese dos resultados para o TCC

# ==============================================================

 

print("\n")

print("=" * 75)

print("TABELA RESUMO FINAL — RESULTADOS PARA O TCC")

print("=" * 75)

print(f"\n{'Modelo':<22} {'Escopo':<18} {'R² Adj':>8} {'AIC':>8} {'Obs':>5}")

print("-" * 75)

for nome, escopo, mod, n in [

    ("M1 (baseline)",     "Painel 3 setores", m1,  39),

    ("M3-HAC (principal)","Painel 3 setores", m3,  39),

    ("MA1 (agro base)",   "Setor agro",       ma1, 13),

    ("MA4 (agro princ.)", "Setor agro",       ma4, 13),

]:

    print(f"{nome:<22} {escopo:<18} "

          f"{mod.rsquared_adj:>8.3f} {mod.aic:>8.2f} {n:>5}")

print("-" * 75)

 

print("\n\nCOEFICIENTES-CHAVE — EFEITO DA LEI 14.112/2020")

print("-" * 60)

print(f"{'Modelo':<22} {'Coef':>8} {'p-valor':>10} "

      f"{'Sig':>6} {'Efeito %':>10}")

print("-" * 60)

 

for label, mod, var in [

    ("M3-HAC (DID)",  m3,  "did"),

    ("MA4 (pos_lei)", ma4, "pos_lei")

]:

    coef = mod.params[var]

    pval = mod.pvalues[var]

    ef   = round((np.exp(coef) - 1) * 100, 1)

    sig  = ("***" if pval < 0.01 else "**" if pval < 0.05

            else "*" if pval < 0.10 else "n.s.")

    print(f"{label:<22} {coef:>8.4f} {pval:>10.3f} {sig:>6} {ef:>9.1f}%")

 

print("-" * 60)

print("\n*** p<0.01 | ** p<0.05 | * p<0.10 | n.s. = não significativo")

print("Efeito % = (exp(coef) - 1) × 100")

print("=" * 75)

 

 

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import numpy as np

 

# =============================================================

# TABELA VIF — PAINEL COMPLETO (para exportação como imagem)

# =============================================================

 

# Dados do VIF

vif_data = {

    'Variável': ['selic_lag1', 'pib_lag1', 'tx_cambio', 'log_credito_lag1', 'did'],

    'VIF': [1.081, 1.128, 1.289, 1.726, 1.829]

}

 

vif_df = pd.DataFrame(vif_data)

 

# Configuração da figura

fig, ax = plt.subplots(figsize=(7, 3))

ax.axis('off')

 

# Cabeçalho e dados

col_labels = ['Variável', 'VIF', 'Diagnóstico']

rows = []

for _, row in vif_df.iterrows():

    vif_val = row['VIF']

    if vif_val < 5:

        diag = 'Sem multicolinearidade'

    elif vif_val < 10:

        diag = 'Atenção'

    else:

        diag = 'Severo'

    rows.append([row['Variável'], f"{vif_val:.3f}", diag])

 

# Criação da tabela

table = ax.table(

    cellText=rows,

    colLabels=col_labels,

    cellLoc='center',

    loc='center'

)

 

# Estilo geral

table.auto_set_font_size(False)

table.set_fontsize(11)

table.scale(1.4, 2.0)

 

# Estilo do cabeçalho

for j in range(len(col_labels)):

    table[0, j].set_facecolor('#2c3e50')

    table[0, j].set_text_props(color='white', fontweight='bold')

 

# Estilo das linhas de dados

for i in range(1, len(rows) + 1):

    for j in range(len(col_labels)):

        vif_val = vif_df.iloc[i - 1]['VIF']

        # Cor de fundo alternada

        if i % 2 == 0:

            table[i, j].set_facecolor('#f2f2f2')

        else:

            table[i, j].set_facecolor('#ffffff')

        # Cor da coluna de diagnóstico

        if j == 2:

            if vif_val < 5:

                table[i, j].set_text_props(color='#27ae60', fontweight='bold')

            elif vif_val < 10:

                table[i, j].set_text_props(color='#e67e22', fontweight='bold')

            else:

                table[i, j].set_text_props(color='#e74c3c', fontweight='bold')

 

# Título

plt.title(

    'Tabela 1 — Fatores de Inflação da Variância (VIF)\nModelo de Painel Completo (3 setores)',

    fontsize=12,

    fontweight='bold',

    pad=20,

    loc='center'

)

 

# Nota de rodapé

fig.text(

    0.5, 0.02,

    'Nota: VIF < 5 indica ausência de multicolinearidade relevante. '

    'Limiar crítico: VIF > 10.',

    ha='center',

    fontsize=9,

    color='gray',

    style='italic'

)

 

plt.tight_layout()

plt.savefig('tabela_vif_painel.png', dpi=200, bbox_inches='tight')

plt.show()

 

print("Tabela salva como: tabela_vif_painel.png")

 

# TABELA 3 — COMPARATIVO M1 vs M2 (HC3) vs M3 (HAC)

# MODELAGEM DO PAINEL COMPLETO

# =============================================================

 

# Dados da tabela

dados = {

    'Variável':           ['Intercepto', 'C(setor)[T.industria]', 'C(setor)[T.servicos]',

                           'selic_lag1', 'pib_lag1', 'tx_cambio', 'log_credito_lag1', 'did'],

    'Coeficiente':        [1.991, 2.594, 3.178, 7.090, 0.102, 0.208, 0.027, 1.045],

    'p (M1)':             [0.001, 0.000, 0.000, 0.005, 0.957, 0.016, 0.209, 0.002],

    'Sig M1':             ['***', '***', '***', '***', 'n.s.', '**', 'n.s.', '***'],

    'p (HC3)':            [0.978, 0.971, 0.964, 0.043, 0.976, 0.012, 0.992, 0.608],

    'Sig HC3':            ['n.s.', 'n.s.', 'n.s.', '**', 'n.s.', '**', 'n.s.', 'n.s.'],

    'p (HAC)':            [0.000, 0.000, 0.000, 0.002, 0.956, 0.001, 0.108, 0.079],

    'Sig HAC':            ['***', '***', '***', '***', 'n.s.', '***', 'n.s.', '*'],

}

 

df_tab = pd.DataFrame(dados)

 

# =============================================================

# FUNÇÃO AUXILIAR — COR POR SIGNIFICÂNCIA

# =============================================================

 

def cor_sig(sig):

    if sig == '***':

        return '#1a5276'

    elif sig == '**':

        return '#2980b9'

    elif sig == '*':

        return '#85c1e9'

    else:

        return '#95a5a6'

 

# =============================================================

# MONTAGEM DA FIGURA

# =============================================================

 

fig, ax = plt.subplots(figsize=(13, 6.5))

ax.axis('off')

 

col_labels = [

    'Variável', 'Coeficiente',

    'p-valor\n(M1)', 'Sig.\n(M1)',

    'p-valor\n(HC3)', 'Sig.\n(HC3)',

    'p-valor\n(HAC)', 'Sig.\n(HAC)'

]

 

rows = []

for _, row in df_tab.iterrows():

    rows.append([

        row['Variável'],

        f"{row['Coeficiente']:.3f}",

        f"{row['p (M1)']:.3f}",

        row['Sig M1'],

        f"{row['p (HC3)']:.3f}",

        row['Sig HC3'],

        f"{row['p (HAC)']:.3f}",

        row['Sig HAC'],

    ])

 

# Criação da tabela

table = ax.table(

    cellText=rows,

    colLabels=col_labels,

    cellLoc='center',

    loc='center'

)

 

table.auto_set_font_size(False)

table.set_fontsize(10)

table.scale(1.3, 2.1)

 

# --- Cabeçalho ---

for j in range(len(col_labels)):

    table[0, j].set_facecolor('#2c3e50')

    table[0, j].set_text_props(color='white', fontweight='bold')

 

# --- Cores por bloco de modelo ---

cores_modelo = {

    2: '#d6eaf8',

    3: '#d6eaf8',

    4: '#fdebd0',

    5: '#fdebd0',

    6: '#d5f5e3',

    7: '#d5f5e3',

}

 

# --- Estilo das células de dados ---

for i in range(1, len(rows) + 1):

    eh_did = df_tab.iloc[i - 1]['Variável'] == 'did'

 

    for j in range(len(col_labels)):

 

        # Fundo base alternado

        if i % 2 == 0:

            table[i, j].set_facecolor('#f8f9fa')

        else:

            table[i, j].set_facecolor('#ffffff')

 

        # Sobreposição de cor por bloco de modelo

        if j in cores_modelo:

            r, g, b, _ = plt.matplotlib.colors.to_rgba(cores_modelo[j])

            table[i, j].set_facecolor((r, g, b, 0.35))

 

        # Destaque da linha did — sem sobrepor colunas de sig

        if eh_did:

            table[i, j].set_facecolor('#fef9e7')

 

        # Cor do texto nas colunas de significância

        if j in [3, 5, 7]:

            sig_val = rows[i - 1][j]

            table[i, j].set_text_props(

                color=cor_sig(sig_val),

                fontweight='bold'

            )

 

# =============================================================

# TÍTULO

# =============================================================

 

plt.title(

    'Tabela 3 — Comparativo de Modelos: M1 (OLS), M2 (HC3) e M3 (HAC)\n'

    'Modelagem do Painel Completo — Variável dependente: log(RJ)',

    fontsize=12,

    fontweight='bold',

    pad=22,

    loc='center'

)

 

# =============================================================

# NOTA DE RODAPÉ

# =============================================================

 

fig.text(

    0.5, 0.01,

    'Nota: Coeficientes idênticos entre M1, M2 e M3 — a estrutura de erros afeta apenas a inferência (p-valores e ICs), não os parâmetros estimados.\n'

    'Significância: *** p<0,01 | ** p<0,05 | * p<0,10 | n.s. = não significativo. '

    'M2 (HC3) descartado como modelo principal por instabilidade em amostras pequenas. '

    'M3 (HAC) adotado como modelo principal.',

    ha='center',

    fontsize=8.5,

    color='gray',

    style='italic'

)

 

# =============================================================

# LEGENDA — POSICIONADA ACIMA DA NOTA, FORA DA TABELA

# =============================================================

 

leg_m1  = mpatches.Patch(facecolor='#d6eaf8', edgecolor='gray', label='M1 — OLS padrão')

leg_hc3 = mpatches.Patch(facecolor='#fdebd0', edgecolor='gray', label='M2 — HC3 (descartado)')

leg_hac = mpatches.Patch(facecolor='#d5f5e3', edgecolor='gray', label='M3 — HAC (modelo principal)')

 

fig.legend(

    handles=[leg_m1, leg_hc3, leg_hac],

    loc='lower center',

    bbox_to_anchor=(0.5, 0.10),

    ncol=3,

    fontsize=9,

    framealpha=0.9

)

 

plt.tight_layout(rect=[0, 0.15, 1, 1])

plt.savefig('tabela2_comparativo_m1_m2_m3.png', dpi=200, bbox_inches='tight')

plt.show()

 

print("Tabela salva como: tabela2_comparativo_m1_m2_m3.png")

 

# TABELA 5 — VIF DO SETOR AGROPECUÁRIO

# =============================================================

 

def gerar_tabela5():

 

    vif_data = {

        'Variável':    ['selic_lag1', 'pib_lag1', 'tx_cambio', 'soja_agro', 'log_credito_lag1', 'pos_lei'],

        'VIF':         [2.51, 1.20, 6.00, 16.11, 2.14, 8.90],

        'Diagnóstico': ['Sem multicolinearidade', 'Sem multicolinearidade', 'Moderado',

                        'Severo', 'Sem multicolinearidade', 'Elevado']

    }

 

    df_vif = pd.DataFrame(vif_data)

 

    def cor_diag(diag):

        if diag == 'Sem multicolinearidade':

            return '#27ae60'

        elif diag == 'Moderado':

            return '#e67e22'

        elif diag == 'Elevado':

            return '#e67e22'

        else:

            return '#e74c3c'

 

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axis('off')

 

    col_labels = ['Variável', 'VIF', 'Diagnóstico']

    rows = [[row['Variável'], f"{row['VIF']:.2f}", row['Diagnóstico']]

            for _, row in df_vif.iterrows()]

 

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')

    table.auto_set_font_size(False)

    table.set_fontsize(11)

    table.scale(1.5, 2.1)

 

    # Cabeçalho

    for j in range(3):

        table[0, j].set_facecolor('#2c3e50')

        table[0, j].set_text_props(color='white', fontweight='bold')

 

    # Células de dados

    for i in range(1, len(rows) + 1):

        for j in range(3):

            table[i, j].set_facecolor('#f8f9fa' if i % 2 == 0 else '#ffffff')

        # Cor do diagnóstico

        diag = rows[i - 1][2]

        table[i, 2].set_text_props(color=cor_diag(diag), fontweight='bold')

        # Destaque linha severa

        if diag == 'Severo':

            for j in range(3):

                table[i, j].set_facecolor('#fdf2f2')

 

    plt.title(

        'Tabela 5 — Fatores de Inflação da Variância (VIF)\nRegressores do Modelo Agropecuário',

        fontsize=12, fontweight='bold', pad=20, loc='center'

    )

 

    fig.text(

        0.5, 0.02,

        'Nota: VIF < 5 = sem multicolinearidade relevante | VIF 5–10 = atenção | VIF > 10 = severo.\n'

        'O VIF elevado de soja_agro e pos_lei reflete confundimento temporal entre o período pós-lei,\n'

        'câmbio depreciado e commodities valorizadas. Coeficientes individuais devem ser interpretados com cautela.',

        ha='center', fontsize=8.5, color='gray', style='italic'

    )

 

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    plt.savefig('tabela4_vif_agro.png', dpi=200, bbox_inches='tight')

    plt.show()

    print("Tabela 5 salva como: tabela4_vif_agro.png")

 

 

# =============================================================

# TABELA 6 — MÉTRICAS DE AJUSTE DOS MODELOS DO AGRO

# =============================================================

 

def gerar_tabela6():

 

    dados = {

        'Modelo':        ['MA1', 'MA2', 'MA3', 'MA4'],

        'Especificação': [

            'Macro baseline (Selic, PIB, Câmbio)',

            'MA1 + Soja + Milho',

            'MA1 + Crédito rural',

            'MA1 + Soja + Crédito + pos_lei'

        ],

        'R²':            [0.772, 0.814, 0.789, 0.888],

        'R² Ajustado':   [0.696, 0.681, 0.684, 0.775],

        'AIC':           [27.50, 28.87, 28.48, 24.31],

        'BIC':           [29.75, 32.26, 31.30, 28.26],

    }

 

    df_met = pd.DataFrame(dados)

 

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.axis('off')

 

    col_labels = ['Modelo', 'Especificação', 'R²', 'R² Ajustado', 'AIC', 'BIC']

    rows = [

        [

            row['Modelo'],

            row['Especificação'],

            f"{row['R²']:.3f}",

            f"{row['R² Ajustado']:.3f}",

            f"{row['AIC']:.2f}",

            f"{row['BIC']:.2f}",

        ]

        for _, row in df_met.iterrows()

    ]

 

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')

    table.auto_set_font_size(False)

    table.set_fontsize(10)

    table.scale(1.3, 2.1)

 

    # Cabeçalho

    for j in range(len(col_labels)):

        table[0, j].set_facecolor('#2c3e50')

        table[0, j].set_text_props(color='white', fontweight='bold')

 

    # Células de dados

    for i in range(1, len(rows) + 1):

        eh_ma4 = df_met.iloc[i - 1]['Modelo'] == 'MA4'

        for j in range(len(col_labels)):

            if eh_ma4:

                table[i, j].set_facecolor('#eafaf1')

                table[i, j].set_text_props(fontweight='bold')

            else:

                table[i, j].set_facecolor('#f8f9fa' if i % 2 == 0 else '#ffffff')

 

    # Coluna de especificação alinhada à esquerda

    for i in range(1, len(rows) + 1):

        table[i, 1].set_text_props(ha='left')

 

    plt.title(

        'Tabela 6 — Métricas de Ajuste dos Modelos do Setor Agropecuário\n'

        'Variável dependente: log(RJ) | Erros HAC/Newey-West | N = 13',

        fontsize=12, fontweight='bold', pad=20, loc='center'

    )

 

    fig.text(

        0.5, 0.02,

        'Nota: Modelo selecionado em verde (MA4) — maior R² ajustado e menor AIC e BIC simultaneamente.\n'

        'Critérios de seleção: maior R² ajustado, menor AIC e coerência dos sinais com a teoria econômica.',

        ha='center', fontsize=8.5, color='gray', style='italic'

    )

 

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    plt.savefig('tabela5_metricas_agro.png', dpi=200, bbox_inches='tight')

    plt.show()

    print("Tabela 6 salva como: tabela5_metricas_agro.png")

 

 

# =============================================================

# TABELA 7 — COEFICIENTES DO MODELO MA4

# =============================================================

 

def gerar_tabela7():

 

    dados = {

        'Variável':     ['Intercepto', 'selic_lag1', 'pib_lag1', 'tx_cambio',

                         'soja_agro', 'log_credito_lag1', 'pos_lei'],

        'Coeficiente':  [-8.214, 0.412, 1.456, 1.397, -0.033, -0.020, 1.438],

        'Erro Padrão':  [3.102, 3.201, 0.862, 0.271, 0.011, 0.011, 0.521],

        'p-valor':      [0.030, 0.901, 0.104, 0.000, 0.006, 0.071, 0.013],

        'Sig':          ['**', 'n.s.', 'n.s.', '***', '***', '*', '**'],

        'Interpretação':['—', 'Não significativo', 'Não significativo',

                         'Fator de pressão (+)', 'Fator protetor (−)',

                         'Fator protetor (−)', 'Efeito da Lei 14.112/2020 (+)']

    }

 

    df_ma4 = pd.DataFrame(dados)

 

    def cor_sig(sig):

        if sig == '***':   return '#1a5276'

        elif sig == '**':  return '#2980b9'

        elif sig == '*':   return '#85c1e9'

        else:              return '#95a5a6'

 

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.axis('off')

 

    col_labels = ['Variável', 'Coeficiente', 'Erro Padrão', 'p-valor', 'Sig.', 'Interpretação']

    rows = [

        [

            row['Variável'],

            f"{row['Coeficiente']:.3f}",

            f"{row['Erro Padrão']:.3f}",

            f"{row['p-valor']:.3f}",

            row['Sig'],

            row['Interpretação'],

        ]

        for _, row in df_ma4.iterrows()

    ]

 

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')

    table.auto_set_font_size(False)

    table.set_fontsize(10)

    table.scale(1.3, 2.1)

 

    # Cabeçalho

    for j in range(len(col_labels)):

        table[0, j].set_facecolor('#2c3e50')

        table[0, j].set_text_props(color='white', fontweight='bold')

 

    # Células de dados

    for i in range(1, len(rows) + 1):

        eh_poslei = df_ma4.iloc[i - 1]['Variável'] == 'pos_lei'

        sig_val   = rows[i - 1][4]

 

        for j in range(len(col_labels)):

            # Fundo base

            table[i, j].set_facecolor('#f8f9fa' if i % 2 == 0 else '#ffffff')

            # Destaque pos_lei

            if eh_poslei:

                table[i, j].set_facecolor('#fef9e7')

                table[i, j].set_text_props(fontweight='bold')

 

        # Cor da coluna Sig

        table[i, 4].set_text_props(color=cor_sig(sig_val), fontweight='bold')

 

        # Alinhamento coluna interpretação

        table[i, 5].set_text_props(ha='left')

 

    plt.title(

        'Tabela 7 — Coeficientes Estimados do Modelo MA4 (HAC/Newey-West)\n'

        'Setor Agropecuário — Variável dependente: log(RJ) | R² Ajustado = 0,775 | N = 13',

        fontsize=12, fontweight='bold', pad=20, loc='center'

    )

 

    fig.text(

        0.5, 0.02,

        'Nota: Erros padrão robustos à heterocedasticidade e autocorrelação (HAC, maxlags=2).\n'

        'Significância: *** p<0,01 | ** p<0,05 | * p<0,10 | n.s. = não significativo.\n'

        'Linha destacada (pos_lei): coeficiente central do trabalho — efeito da Lei 14.112/2020.',

        ha='center', fontsize=8.5, color='gray', style='italic'

    )

 

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    plt.savefig('tabela6_coeficientes_ma4.png', dpi=200, bbox_inches='tight')

    plt.show()

    print("Tabela 7 salva como: tabela6_coeficientes_ma4.png")

 

 

# =============================================================

# TABELA 8 — SÍNTESE DOS MODELOS PRINCIPAIS E EFEITO DA LEI

# =============================================================

 

def gerar_tabela8():

 

    dados = {

        'Modelo':         ['M3-HAC (DID)', 'MA4 (pos_lei)'],

        'Escopo':         ['Painel — 3 setores', 'Setor agropecuário'],

        'R² Ajustado':    [0.822, 0.775],

        'AIC':            [60.14, 24.31],

        'N':              [39, 13],

        'Coef. Lei':      [1.045, 1.438],

        'p-valor':        [0.079, 0.013],

        'Sig':            ['*', '**'],

        'Efeito %':       ['+184%', '+321%'],

    }

 

    df_sint = pd.DataFrame(dados)

 

    def cor_sig(sig):

        if sig == '***':   return '#1a5276'

        elif sig == '**':  return '#2980b9'

        elif sig == '*':   return '#85c1e9'

        else:              return '#95a5a6'

 

    fig, ax = plt.subplots(figsize=(13, 3.5))

    ax.axis('off')

 

    col_labels = [

        'Modelo', 'Escopo', 'R² Ajustado', 'AIC', 'N',

        'Coef. Lei', 'p-valor', 'Sig.', 'Efeito Estimado'

    ]

 

    rows = [

        [

            row['Modelo'],

            row['Escopo'],

            f"{row['R² Ajustado']:.3f}",

            f"{row['AIC']:.2f}",

            str(row['N']),

            f"{row['Coef. Lei']:.3f}",

            f"{row['p-valor']:.3f}",

            row['Sig'],

            row['Efeito %'],

        ]

        for _, row in df_sint.iterrows()

    ]

 

    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')

    table.auto_set_font_size(False)

    table.set_fontsize(10.5)

    table.scale(1.3, 2.4)

 

    # Cabeçalho

    for j in range(len(col_labels)):

        table[0, j].set_facecolor('#2c3e50')

        table[0, j].set_text_props(color='white', fontweight='bold')

 

    # Cores por modelo

    cores_linha = ['#d6eaf8', '#d5f5e3']

 

    for i in range(1, len(rows) + 1):

        sig_val = rows[i - 1][7]

        for j in range(len(col_labels)):

            table[i, j].set_facecolor(cores_linha[i - 1])

            table[i, j].set_text_props(fontweight='bold')

        # Cor da coluna Sig

        table[i, 7].set_text_props(color=cor_sig(sig_val), fontweight='bold')

        # Destaque coluna efeito %

        table[i, 8].set_text_props(color='#1a5276', fontweight='bold')

 

    plt.title(

        'Tabela 8 — Síntese dos Modelos Principais e Efeitos Estimados da Lei 14.112/2020',

        fontsize=12, fontweight='bold', pad=20, loc='center'

    )

 

    fig.text(

        0.5, 0.02,

        'Nota: Efeito % = (exp(coef) − 1) × 100. '

        'Significância: *** p<0,01 | ** p<0,05 | * p<0,10.\n'

        'M3-HAC: efeito DiD do agro vs. demais setores. '

        'MA4: efeito pos_lei dentro do setor agropecuário, controlando por câmbio, soja e crédito.\n'

        'O intervalo entre +184% e +321% representa a estimativa preliminar do efeito da lei.',

        ha='center', fontsize=8.5, color='gray', style='italic'

    )

 

    # Legenda de cores

    leg_m3  = mpatches.Patch(facecolor='#d6eaf8', edgecolor='gray', label='M3-HAC — Painel completo (DID)')

    leg_ma4 = mpatches.Patch(facecolor='#d5f5e3', edgecolor='gray', label='MA4 — Modelo agropecuário (pos_lei)')

 

    fig.legend(

        handles=[leg_m3, leg_ma4],

        loc='lower center',

        bbox_to_anchor=(0.5, 0.13),

        ncol=2,

        fontsize=9,

        framealpha=0.9

    )

 

    plt.tight_layout(rect=[0, 0.18, 1, 1])

    plt.savefig('tabela7_sintese_lei.png', dpi=200, bbox_inches='tight')

    plt.show()

    print("Tabela 8 salva como: tabela7_sintese_lei.png")

 

 

# =============================================================

# EXECUÇÃO

# =============================================================

 

gerar_tabela5()

gerar_tabela6()

gerar_tabela7()

gerar_tabela8()