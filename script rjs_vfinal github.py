# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:07:29 2026

@author: PC
"""

# ==============================================================
# INSOLVÊNCIA NO AGRONEGÓCIO BRASILEIRO
# Transformação institucional, determinantes econômicos e limites
# de atribuição do efeito da Lei 14.112/2020
# ==============================================================
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# ETAPA 1 — CONFIGURAÇÃO
# ==============================================================
NOME_BASE = "database_rj setorial_v5.xlsx"
SETOR_TRATADO  = "agro"
SETORES        = ["agro", "industria", "comercio", "servicos"]
ANO_LEI        = 2021
ANO_BASE_EVENT = 2020
MAXLAGS_PAINEL = 2
MAXLAGS_AGRO   = 1
USE_CORRECTION = True
CREDITO_2012_COMO_AUSENTE = True
LIMIAR_ATENCAO, LIMIAR_SEVERO = 5.0, 10.0

sns.set_style("white")
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})
FIGURAS_NAO_GRAVADAS = []

COR_AGRO, COR_IND, COR_COM, COR_SERV = "#27AE60", "#8E44AD", "#D68910", "#16A085"
COR_LEI = "#C0392B"
CORES_SETOR = {"agro": COR_AGRO, "industria": COR_IND,
               "comercio": COR_COM, "servicos": COR_SERV}

try:
    _base = Path(__file__).parent
except NameError:
    _base = Path.cwd()
DIR_SAIDA = (_base / "saida").resolve()
try:
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    _t = DIR_SAIDA / "_teste.txt"
    _t.write_text("ok", encoding="utf-8")
    _t.unlink()
except OSError as e:
    DIR_SAIDA = (Path.home() / "saida_tcc").resolve()
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    print(f"Pasta padrão indisponível ({type(e).__name__}). Usando {DIR_SAIDA}")


def titulo(txt, char="="):
    print("\n" + char * 78)
    print(txt)
    print(char * 78)


def br(x, dec=3):
    """Formata número no padrão decimal brasileiro."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    if isinstance(x, float) and np.isinf(x):
        return "infinito"
    return f"{x:,.{dec}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def sig_stars(p):
    if pd.isna(p):
        return "—"
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return "n.s."


def efeito_pct(coef):
    """Variação percentual implicada por coeficiente em escala log."""
    return (np.exp(coef) - 1) * 100


def exportar(tabela, nome, indice=True):
    tabela.to_csv(DIR_SAIDA / f"{nome}.csv", sep=";", decimal=",",
                  index=indice, encoding="utf-8-sig")


def salvar_fig(nome, dpi=200):
    """Grava a figura corrente em caminho absoluto."""
    try:
        plt.savefig(DIR_SAIDA / f"{nome}.png", dpi=dpi, bbox_inches="tight")
    except Exception as e:
        FIGURAS_NAO_GRAVADAS.append((nome, type(e).__name__, str(e)))
    finally:
        plt.close()


def rotular_painel(ax, letra):
    """Identifica painel com letra maiuscula no canto superior esquerdo."""
    ax.text(0.01, 0.99, letra, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def localizar(nome_arquivo):
    """Procura o arquivo na pasta do script, na pasta atual e em Downloads."""
    candidatos = [_base, Path.cwd(), Path.home() / "Downloads"]
    for pasta in candidatos:
        alvo = pasta / nome_arquivo
        if alvo.exists():
            return alvo
    print(f"Diretório de trabalho: {Path.cwd()}")
    print("Arquivos .xlsx encontrados nele:")
    for f in sorted(Path.cwd().glob("*.xlsx")):
        print(f"  {f.name}")
    raise FileNotFoundError(f"'{nome_arquivo}' não localizado.")


def para_numero(serie):
    """Converte para float sem alterar escala."""
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")
    s = (serie.astype(str).str.strip()
              .str.replace("%", "", regex=False)
              .str.replace("\u00a0", "", regex=False)
              .str.replace(" ", "", regex=False))
    tem_virgula = s.str.contains(",", na=False)
    s = s.where(~tem_virgula,
                s.str.replace(".", "", regex=False)
                 .str.replace(",", ".", regex=False))
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    return pd.to_numeric(s, errors="coerce")


print(f"Diretório de saída: {DIR_SAIDA}")

# ==============================================================
# ETAPA 2 — CARREGAMENTO E LIMPEZA
# ==============================================================
titulo("ETAPA 2 — CARREGAMENTO E LIMPEZA")
ARQ = localizar(NOME_BASE)
print(f"Base: {ARQ}")
print(f"Abas: {pd.ExcelFile(ARQ).sheet_names}")

titulo("2.1 PAINEL ANUAL", "-")
COLS_ANUAL = ["ano", "setor", "rj_requeridas", "fal_requeridas", "selic",
              "tx_cambio", "var_pib", "preco_soja", "preco_milho",
              "credito_rural"]
anual = pd.read_excel(ARQ, sheet_name="painel_anual")
anual.columns = [str(c).strip() for c in anual.columns]
faltando = [c for c in COLS_ANUAL if c not in anual.columns]
if faltando:
    raise RuntimeError(f"painel_anual: colunas ausentes {faltando}")
anual = anual[COLS_ANUAL].copy()
anual["setor"] = anual["setor"].astype(str).str.strip().str.lower()
for c in COLS_ANUAL[2:]:
    anual[c] = para_numero(anual[c])
anual = anual.sort_values(["setor", "ano"]).reset_index(drop=True)
print(f"Dimensões: {anual.shape[0]} x {anual.shape[1]}")
print(f"Período  : {anual['ano'].min()} a {anual['ano'].max()}")
print(f"Setores  : {sorted(anual['setor'].unique())}")
print("\nEscala de cada variável:")
print(pd.DataFrame({
    "mínimo": anual[COLS_ANUAL[2:]].min(),
    "média":  anual[COLS_ANUAL[2:]].mean(),
    "máximo": anual[COLS_ANUAL[2:]].max(),
    "ausentes": anual[COLS_ANUAL[2:]].isna().sum()}).round(4).to_string())
if anual["selic"].max() > 1:
    raise RuntimeError(f"Selic anual com máximo {anual['selic'].max():.4f}: "
                       f"aparentemente percentual. Espera-se forma decimal.")
print(f"\nSelic em forma decimal confirmada: {br(anual['selic'].min(), 4)} a "
      f"{br(anual['selic'].max(), 4)}  ({br(anual['selic'].min()*100, 2)}% a "
      f"{br(anual['selic'].max()*100, 2)}% a.a.)")
print("\nSéries do setor tratado:")
print(anual[anual["setor"] == SETOR_TRATADO][
    ["ano", "rj_requeridas", "fal_requeridas", "selic", "tx_cambio",
     "var_pib", "preco_soja", "credito_rural"]].round(4).to_string(index=False))

titulo("2.2 PAINEL MENSAL", "-")
COLS_MENSAL = ["ano", "mes", "data", "setor", "fal_requeridas",
               "fal_decretadas", "rj_requeridas", "rj_deferidas",
               "rj_concedidas", "selic", "tx_cambio"]
mensal = pd.read_excel(ARQ, sheet_name="mensal_setor")
mensal.columns = [str(c).strip() for c in mensal.columns]
faltando = [c for c in COLS_MENSAL if c not in mensal.columns]
if faltando:
    raise RuntimeError(f"mensal_setor: colunas ausentes {faltando}")
mensal = mensal[COLS_MENSAL].copy()
mensal["setor"] = mensal["setor"].astype(str).str.strip().str.lower()
mensal["data"] = pd.to_datetime(mensal["data"])
for c in COLS_MENSAL[4:]:
    mensal[c] = para_numero(mensal[c])
mensal = mensal.sort_values(["setor", "data"]).reset_index(drop=True)
print(f"Dimensões: {mensal.shape[0]} x {mensal.shape[1]}")
print(f"Período  : {mensal['data'].min():%m/%Y} a {mensal['data'].max():%m/%Y}"
      f"  ({mensal['data'].nunique()} meses)")
print("\nEscala de cada variável:")
print(pd.DataFrame({
    "mínimo": mensal[COLS_MENSAL[4:]].min(),
    "média":  mensal[COLS_MENSAL[4:]].mean(),
    "máximo": mensal[COLS_MENSAL[4:]].max(),
    "ausentes": mensal[COLS_MENSAL[4:]].isna().sum()}).round(4).to_string())
if mensal["selic"].max() > 1:
    raise RuntimeError("Selic mensal aparentemente em formato percentual.")
print("\nMeses por ano:")
print(mensal.groupby("ano")["data"].nunique().to_string())
inc = mensal.groupby("ano")["data"].nunique()
inc = inc[inc < 12].index.tolist()
if inc:
    print(f"Anos incompletos, não usar em agregação anual: {inc}")

titulo("2.3 PESSOA FÍSICA — TRIMESTRAL", "-")
pf = pd.read_excel(ARQ, sheet_name="pf_trimestral")
pf.columns = [str(c).strip() for c in pf.columns]
pf = pf[["ano", "trimestre", "data", "pedidos_pf"]].copy()
pf["pedidos_pf"] = para_numero(pf["pedidos_pf"])
pf["trimestre_num"] = pf["trimestre"].astype(str).str.extract(r"(\d)").astype(int)
pf = pf.sort_values(["ano", "trimestre_num"]).reset_index(drop=True)
print(f"Trimestres: {len(pf)} | ausentes: {int(pf['pedidos_pf'].isna().sum())}")
print("\nSérie completa:")
print(pf[["ano", "trimestre", "pedidos_pf"]].to_string(index=False))
print("\nTotal por ano:")
print(pf.groupby("ano")["pedidos_pf"].agg(["sum", "count"]).to_string())

titulo("2.4 VERIFICAÇÕES DE INTEGRIDADE", "-")
erros, avisos = [], []
esperado = len(anual["ano"].unique()) * len(SETORES)
if len(anual) != esperado:
    erros.append(f"painel_anual tem {len(anual)} linhas, esperado {esperado}")
if set(anual["setor"].unique()) != set(SETORES):
    erros.append(f"setores do painel anual: {sorted(anual['setor'].unique())}")
if anual.duplicated(["ano", "setor"]).any():
    erros.append("painel_anual tem combinações ano/setor duplicadas")
for c in ["selic", "tx_cambio"]:
    if anual.groupby("ano")[c].nunique().max() > 1:
        erros.append(f"{c} varia entre setores no painel anual")
    if mensal.groupby("data")[c].nunique().max() > 1:
        erros.append(f"{c} varia entre setores no painel mensal")
for c in ["preco_soja", "preco_milho", "credito_rural"]:
    fora = anual[(anual["setor"] != SETOR_TRATADO) & anual[c].notna()]
    if len(fora):
        erros.append(f"{c} preenchida fora do agro em {len(fora)} linha(s)")
    dentro = anual[(anual["setor"] == SETOR_TRATADO) & anual[c].isna()]
    if len(dentro):
        anos_na = dentro["ano"].tolist()
        if c == "credito_rural" and anos_na == [2012]:
            print("credito_rural ausente em 2012 no agro — esperado, a série "
                  "do SICOR inicia em 2013")
        else:
            avisos.append(f"{c} ausente no agro em {anos_na}")
esperado_m = mensal["data"].nunique() * len(SETORES)
if len(mensal) != esperado_m:
    erros.append(f"mensal_setor tem {len(mensal)} linhas, esperado {esperado_m}")
if mensal.duplicated(["data", "setor"]).any():
    erros.append("mensal_setor tem combinações data/setor duplicadas")
n_inc = int((mensal["rj_concedidas"] > mensal["rj_deferidas"]).sum())
if n_inc:
    avisos.append(f"{n_inc} mês(es) com concedidas acima de deferidas — "
                  f"defasagem entre estágios processuais")
for nome_t, tab, cols in [
        ("painel_anual", anual, ["rj_requeridas", "fal_requeridas"]),
        ("mensal_setor", mensal, ["fal_requeridas", "fal_decretadas",
                                  "rj_requeridas", "rj_deferidas",
                                  "rj_concedidas"]),
        ("pf_trimestral", pf, ["pedidos_pf"])]:
    for c in cols:
        if (tab[c] < 0).any():
            erros.append(f"{nome_t}.{c} tem valor negativo")
if pf["pedidos_pf"].isna().any():
    erros.append("pf_trimestral tem trimestre sem valor")
print()
if erros:
    for e in erros:
        print(f"ERRO   {e}")
    raise RuntimeError(f"{len(erros)} inconsistência(s) estrutural(is).")
for a in avisos:
    print(f"AVISO  {a}")
print(f"\n{len(anual)} observações anuais, {len(mensal)} mensais, "
      f"{len(pf)} trimestrais — estrutura consistente")

# ==============================================================
# ETAPA 3 — ENGENHARIA DE VARIÁVEIS
# ==============================================================
titulo("ETAPA 3 — ENGENHARIA DE VARIÁVEIS")
pa = anual.copy()

pa["total_insolvencia"] = pa["rj_requeridas"] + pa["fal_requeridas"]
pa["log_rj"]   = np.log(pa["rj_requeridas"])
pa["log_fal"]  = np.log(pa["fal_requeridas"])
pa["razao_rj"] = pa["rj_requeridas"] / pa["total_insolvencia"]
pa["log_odds"] = pa["log_rj"] - pa["log_fal"]
print("Desfechos do setor tratado:")
print(pa[pa["setor"] == SETOR_TRATADO][
    ["ano", "rj_requeridas", "fal_requeridas", "total_insolvencia",
     "razao_rj", "log_odds"]].round(4).to_string(index=False))
print("\nAmplitude no painel:")
for c in ["log_rj", "log_fal", "razao_rj", "log_odds"]:
    print(f"  {c:<10} {pa[c].min():>8.4f} a {pa[c].max():>8.4f}")
sat = pa[(pa["setor"] == SETOR_TRATADO) & (pa["razao_rj"] > 0.95)]
if len(sat):
    print(f"\nAnos do agro com razão acima de 0,95: {sat['ano'].tolist()}")
    print("A razão é limitada superiormente; log_odds é o desfecho de modelagem.")

pa["agro"]    = (pa["setor"] == SETOR_TRATADO).astype(int)
pa["pos_lei"] = (pa["ano"] >= ANO_LEI).astype(int)
pa["did"]     = pa["agro"] * pa["pos_lei"]
if int(pa["agro"].sum()) != len(pa["ano"].unique()):
    raise RuntimeError("Indicador de grupo inconsistente.")
if not pa.loc[pa["agro"] == 1, "setor"].eq(SETOR_TRATADO).all():
    raise RuntimeError("Indicador de grupo marca setores incorretos.")
if not pa.loc[pa["agro"] == 0, "setor"].ne(SETOR_TRATADO).all():
    raise RuntimeError("Indicador de grupo deixa linhas do agro desmarcadas.")
print(f"\nIntegridade do grupo: {int(pa['agro'].sum())} linhas tratadas, "
      f"{len(pa) - int(pa['agro'].sum())} de controle")
print(f"Observações com did = 1: {int(pa['did'].sum())} "
      f"(anos {sorted(pa.loc[pa['did'] == 1, 'ano'].unique().tolist())})")

pa["t"]      = pa["ano"] - pa["ano"].min()
pa["t_agro"] = pa["t"] * pa["agro"]

pa["soja_agro"]   = pa["preco_soja"].fillna(0) * pa["agro"]
pa["milho_agro"]  = pa["preco_milho"].fillna(0) * pa["agro"]
pa["cred_agro"]   = pa["credito_rural"] * pa["agro"]
pa["log_credito"] = np.log(pa["cred_agro"].where(pa["agro"] == 1))

g = pa.groupby("setor", sort=False)
pa["selic_lag1"]       = g["selic"].shift(1)
pa["pib_lag1"]         = g["var_pib"].shift(1)
pa["log_credito_lag1"] = g["log_credito"].shift(1)
print("\nDefasagens no agro:")
print(pa[pa["agro"] == 1][["ano", "selic", "selic_lag1", "var_pib", "pib_lag1",
                           "log_credito", "log_credito_lag1"]]
      .round(4).to_string(index=False))

media_lc = pa.loc[(pa["agro"] == 1) & pa["log_credito_lag1"].notna(),
                  "log_credito_lag1"].mean()
pa["credito_c"] = np.where(pa["agro"] == 1,
                           pa["log_credito_lag1"] - media_lc, 0.0)
print(f"\nCrédito centrado na média do agro: {br(media_lc, 4)}")
_ac = pa[(pa["agro"] == 1) & pa["credito_c"].notna()]
print(f"  escala original: {br(_ac['log_credito_lag1'].min())} a "
      f"{br(_ac['log_credito_lag1'].max())} no agro")
print(f"  escala centrada: {br(_ac['credito_c'].min())} a "
      f"{br(_ac['credito_c'].max())} no agro, 0 nos controles")

COLS_NUCLEO = ["log_rj", "log_fal", "log_odds", "razao_rj",
               "selic_lag1", "pib_lag1", "tx_cambio"]
pa_est  = pa.dropna(subset=COLS_NUCLEO).reset_index(drop=True)
pa_cred = pa.dropna(subset=COLS_NUCLEO + ["credito_c"]).reset_index(drop=True)
print(f"\nAmostra principal            : {len(pa_est)} obs "
      f"({pa_est['ano'].min()}–{pa_est['ano'].max()})")
print(pa_est.groupby("setor")["ano"].agg(["count", "min", "max"]).to_string())
print(f"\nAmostra com crédito defasado : {len(pa_cred)} obs")
print(pa_cred.groupby("setor")["ano"].agg(["count", "min", "max"]).to_string())
print(f"\nCusto de incluir crédito: {len(pa_est) - len(pa_cred)} observação(ões)")
ag_est  = pa_est[pa_est["agro"] == 1].copy().reset_index(drop=True)
ag_cred = pa_cred[pa_cred["agro"] == 1].copy().reset_index(drop=True)
print(f"\nSetor tratado — principal: N = {len(ag_est)} "
      f"({ag_est['ano'].min()}–{ag_est['ano'].max()})")
print(f"Setor tratado — c/ crédito: N = {len(ag_cred)} "
      f"({ag_cred['ano'].min()}–{ag_cred['ano'].max()})")

print("\nEstrutura de variação:")
for c in ["selic_lag1", "pib_lag1", "tx_cambio", "soja_agro", "credito_c"]:
    varia = pa_est.groupby("ano")[c].nunique(dropna=False).max() > 1
    print(f"  {c:<18} {'varia entre setores' if varia else 'apenas temporal'}")

titulo("PARÂMETROS DA ANÁLISE — TRANSCREVER NA METODOLOGIA", "-")
print(f"Base                      : {NOME_BASE}")
print(f"Setores                   : {', '.join(SETORES)} (tratado: {SETOR_TRATADO})")
print(f"Desfechos                 : log das recuperações, log das falências,")
print(f"                            log da razão entre ambos")
print(f"Vigência da lei           : a partir de {ANO_LEI}")
print(f"Selic                     : forma decimal; efeito por ponto percentual")
print(f"PIB                       : variação anual do PIB setorial")
print(f"Selic e câmbio            : comuns aos setores; variação apenas temporal")
print(f"Crédito rural             : log, defasado, centrado na média do agro")
print(f"HAC maxlags painel / agro : {MAXLAGS_PAINEL} / {MAXLAGS_AGRO}")
print(f"Correção de amostra finita: {USE_CORRECTION}")
print(f"Painel anual              : {len(pa_est)} obs ({len(pa_cred)} com crédito)")
print(f"Observações tratadas      : {int(pa_est['did'].sum())}")

# ==============================================================
# ETAPA 4 — ANÁLISE EXPLORATÓRIA
# ==============================================================
titulo("ETAPA 4 — ANÁLISE EXPLORATÓRIA")
ANO_MIN, ANO_MAX = int(pa["ano"].min()), int(pa["ano"].max())
ROT_PRE = f"{ANO_MIN}–{ANO_LEI-1}"
ROT_POS = f"{ANO_LEI}–{ANO_MAX}"

titulo("4.1 EVOLUÇÃO POR SETOR", "-")
for var, rot in [("rj_requeridas", "Recuperações judiciais requeridas"),
                 ("fal_requeridas", "Falências requeridas"),
                 ("total_insolvencia", "Total de pedidos de insolvência")]:
    piv = pa.pivot_table(index="ano", columns="setor", values=var)
    print(f"\n{rot}:")
    print(piv.astype(int).to_string())
    exportar(piv.astype(int), f"tab_evolucao_{var}")
piv_razao = pa.pivot_table(index="ano", columns="setor", values="razao_rj")
print("\nParticipação da recuperação no total (%):")
print((piv_razao * 100).round(1).to_string())
exportar((piv_razao * 100).round(2), "tab_evolucao_razao")
piv_odds = pa.pivot_table(index="ano", columns="setor", values="log_odds")
print("\nLog da razão entre recuperações e falências:")
print(piv_odds.round(3).to_string())
exportar(piv_odds.round(4), "tab_evolucao_log_odds")

titulo("4.2 DECOMPOSIÇÃO DO MOVIMENTO DE COMPOSIÇÃO", "-")


def decompor(s, ini, fim):
    sub = pa[pa["setor"] == s].set_index("ano")
    d_rj  = sub.loc[fim, "log_rj"]  - sub.loc[ini, "log_rj"]
    d_fal = sub.loc[fim, "log_fal"] - sub.loc[ini, "log_fal"]
    d_odds = d_rj - d_fal
    return {"setor": s, "periodo": f"{ini}–{fim}",
            "rj_ini": int(sub.loc[ini, "rj_requeridas"]),
            "rj_fim": int(sub.loc[fim, "rj_requeridas"]),
            "fal_ini": int(sub.loc[ini, "fal_requeridas"]),
            "fal_fim": int(sub.loc[fim, "fal_requeridas"]),
            "delta_log_rj": d_rj, "delta_log_fal": d_fal,
            "delta_log_odds": d_odds,
            "contrib_rj": d_rj, "contrib_falencia": -d_fal,
            "share_falencia_pct": (-d_fal / d_odds * 100) if d_odds else np.nan}


dec = pd.DataFrame([decompor(s, ANO_MIN, ANO_LEI - 1) for s in SETORES] +
                   [decompor(s, ANO_LEI, ANO_MAX) for s in SETORES])
print(dec.round(4).to_string(index=False))
exportar(dec.round(4), "tab_decomposicao", indice=False)
print("\nParcela do deslocamento atribuível à queda das falências:")
for s in SETORES:
    a = dec[(dec.setor == s) & (dec.periodo == f"{ANO_MIN}–{ANO_LEI-1}")].iloc[0]
    b = dec[(dec.setor == s) & (dec.periodo == f"{ANO_LEI}–{ANO_MAX}")].iloc[0]
    print(f"  {s:<10} {ROT_PRE}: {br(a['share_falencia_pct'], 1):>7}%   "
          f"{ROT_POS}: {br(b['share_falencia_pct'], 1):>7}%")

titulo("4.3 INCLINAÇÃO MÉDIA ANUAL", "-")
linhas = []
for s in SETORES:
    sub = pa[pa["setor"] == s]
    pre, pos = sub[sub["ano"] < ANO_LEI], sub[sub["ano"] >= ANO_LEI]
    reg = {"setor": s}
    for var in ["log_rj", "log_fal", "log_odds"]:
        reg[f"{var}_pre"] = np.polyfit(pre["ano"], pre[var], 1)[0]
        reg[f"{var}_pos"] = np.polyfit(pos["ano"], pos[var], 1)[0]
    linhas.append(reg)
tab_incl = pd.DataFrame(linhas).set_index("setor")
print(tab_incl.round(4).to_string())
exportar(tab_incl.round(4), "tab_inclinacoes")
for var, rot in [("log_rj", "recuperações"), ("log_fal", "falências"),
                 ("log_odds", "composição")]:
    print(f"\n  {rot}, em log por ano:")
    for s in SETORES:
        print(f"    {s:<10} {ROT_PRE} {br(tab_incl.loc[s, var+'_pre']):>7}   "
              f"{ROT_POS} {br(tab_incl.loc[s, var+'_pos']):>7}")
    for per, suf in [(ROT_PRE, "_pre"), (ROT_POS, "_pos")]:
        ctrl = tab_incl.drop(SETOR_TRATADO)[var + suf].mean()
        if abs(ctrl) > 1e-6:
            print(f"    razão agro/controles em {per}: "
                  f"{br(tab_incl.loc[SETOR_TRATADO, var+suf] / ctrl, 1)} vezes")

titulo("4.4 MAIORES DESLOCAMENTOS ANUAIS", "-")
for s in SETORES:
    sub = pa[pa["setor"] == s].sort_values("ano").copy()
    sub["d_odds"] = sub["log_odds"].diff()
    print(f"\n{s} — três maiores variações anuais do log da razão:")
    print(sub.nlargest(3, "d_odds")[["ano", "d_odds", "rj_requeridas",
                                     "fal_requeridas"]]
          .round(3).to_string(index=False))

titulo("4.5 ESTATÍSTICAS DESCRITIVAS", "-")
VARS = ["log_rj", "log_fal", "log_odds", "razao_rj", "selic_lag1",
        "pib_lag1", "tx_cambio", "soja_agro", "credito_c", "did"]
print("Painel de estimação:")
print(pa_est[VARS].describe().T.round(4).to_string())
exportar(pa_est[VARS].describe().T.round(4), "tab_descritivas_painel")
VARS_AG = [v for v in VARS if v != "did"]
print("\nSetor tratado:")
print(ag_est[VARS_AG].describe().T.round(4).to_string())
exportar(ag_est[VARS_AG].describe().T.round(4), "tab_descritivas_agro")

titulo("4.6 CORRELAÇÕES", "-")
corr_p = pa_est[VARS].corr()
print("Painel de estimação:")
print(corr_p.round(3).to_string())
exportar(corr_p.round(3), "tab_corr_painel")
VARS_AGRO = ["log_rj", "log_fal", "log_odds", "selic_lag1", "pib_lag1",
             "tx_cambio", "soja_agro", "milho_agro", "credito_c", "pos_lei", "t"]
corr_a = ag_cred[VARS_AGRO].corr()
print("\nSetor tratado:")
print(corr_a.round(3).to_string())
exportar(corr_a.round(3), "tab_corr_agro")
regs = [v for v in VARS_AGRO if v not in ("log_rj", "log_fal", "log_odds")]
pares = [{"var_1": a, "var_2": b, "correlacao": corr_a.loc[a, b]}
         for i, a in enumerate(regs) for b in regs[i+1:]
         if abs(corr_a.loc[a, b]) > 0.70]
print("\nPares de regressores com correlação absoluta acima de 0,70 no agro:")
if pares:
    print(pd.DataFrame(pares).round(3).to_string(index=False))
    exportar(pd.DataFrame(pares).round(3), "tab_pares_correlacionados",
             indice=False)
else:
    print("Nenhum par acima do limiar.")
print("\nCorrelação de cada regressor com os desfechos, no agro:")
print(corr_a.loc[regs, ["log_rj", "log_fal", "log_odds"]].round(3).to_string())

titulo("4.7 FIGURAS", "-")


def marcar_lei(ax):
    ax.axvline(ANO_LEI - 0.5, color=COR_LEI, linestyle="--", linewidth=1.5)


fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharex=True)
for ax, (var, rot) in zip(axes, [("rj_requeridas", "Recuperações judiciais"),
                                 ("fal_requeridas", "Falências")]):
    for s in SETORES:
        d = pa[pa["setor"] == s]
        ax.plot(d["ano"], d[var], marker="o", color=CORES_SETOR[s],
                label=s.capitalize(), linewidth=2)
    marcar_lei(ax)
    rotular_painel(ax, "AB"[list(axes).index(ax)])
    ax.set_xlabel("Ano")
    ax.set_ylabel("CNPJs")
    ax.legend(fontsize=9)
plt.tight_layout()
salvar_fig("fig_niveis_setoriais")

fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharex=True)
for s in SETORES:
    d = pa[pa["setor"] == s]
    axes[0].plot(d["ano"], d["razao_rj"] * 100, marker="o",
                 color=CORES_SETOR[s], label=s.capitalize(), linewidth=2)
    axes[1].plot(d["ano"], d["log_odds"], marker="o",
                 color=CORES_SETOR[s], label=s.capitalize(), linewidth=2)
axes[0].axhline(100, color="black", linewidth=0.8, linestyle=":")
axes[0].set_ylabel("%")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("log(RJ / falências)")
rotular_painel(axes[0], "A")
rotular_painel(axes[1], "B")
for ax in axes:
    marcar_lei(ax)
    ax.set_xlabel("Ano")
    ax.legend(fontsize=9)
plt.tight_layout()
salvar_fig("fig_composicao")

sub = pa[pa["setor"] == SETOR_TRATADO].sort_values("ano")
base0 = sub.iloc[0]
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(sub["ano"], sub["log_rj"] - base0["log_rj"], marker="o",
        color="#2C6FAC", linewidth=2.5, label="Contribuição das recuperações")
ax.plot(sub["ano"], -(sub["log_fal"] - base0["log_fal"]), marker="s",
        color="#E07B39", linewidth=2.5,
        label="Contribuição da queda das falências")
ax.plot(sub["ano"], sub["log_odds"] - base0["log_odds"], marker="^",
        color="black", linewidth=2.5, linestyle="--",
        label="Deslocamento total da composição")
ax.axhline(0, color="gray", linewidth=0.8)
marcar_lei(ax)
ax.set_xlabel("Ano")
ax.set_ylabel("Variação acumulada (log)")
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
salvar_fig("fig_decomposicao_agro")

for rot, mat, arq in [("Painel de Estimação", corr_p, "fig_corr_painel"),
                      ("Setor Agropecuário", corr_a, "fig_corr_agro")]:
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5,
                square=True, vmin=-1, vmax=1, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})
    plt.tight_layout()
    salvar_fig(arq)
print("Figuras: fig_niveis_setoriais, fig_composicao, fig_decomposicao_agro, "
      "fig_corr_painel, fig_corr_agro")

# ==============================================================
# ETAPA 5 — DIAGNÓSTICO DE MULTICOLINEARIDADE
# ==============================================================
titulo("ETAPA 5 — DIAGNÓSTICO DE MULTICOLINEARIDADE")


def diagnostico_vif(v):
    if pd.isna(v):
        return "—"
    if v < LIMIAR_ATENCAO:
        return "aceitável"
    if v < LIMIAR_SEVERO:
        return "atenção"
    return "severo"


def montar_desenho(dados, regressores, dummies_setor=False):
    """Matriz de desenho efetivamente estimada."""
    X = dados[regressores].astype(float).copy()
    if dummies_setor:
        d = pd.get_dummies(dados["setor"], prefix="setor", drop_first=True)
        X = pd.concat([X, d.astype(float)], axis=1)
    return X.dropna()


def tabela_vif(dados, regressores, dummies_setor=False):
    X = montar_desenho(dados, regressores, dummies_setor)
    Xc = sm.add_constant(X)
    tab = pd.DataFrame({
        "variavel": Xc.columns,
        "vif": [variance_inflation_factor(Xc.values, i)
                for i in range(Xc.shape[1])]})
    tab = tab[tab["variavel"] != "const"].reset_index(drop=True)
    tab["diagnostico"] = tab["vif"].apply(diagnostico_vif)
    tab["r2_auxiliar"] = 1 - 1 / tab["vif"]
    return tab, len(X)


def numero_condicao(dados, regressores, dummies_setor=False):
    """Razão entre o maior e o menor valor singular da matriz padronizada."""
    X = montar_desenho(dados, regressores, dummies_setor)
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    return float(np.linalg.cond(sm.add_constant(Z).values))


titulo("5.1 PAINEL — ESPECIFICAÇÕES", "-")
ESPEC_PAINEL = {
    "P1 macro + did":
        (["selic_lag1", "pib_lag1", "tx_cambio", "did"], pa_est),
    "P2 macro + crédito + did":
        (["selic_lag1", "pib_lag1", "tx_cambio", "credito_c", "did"], pa_cred),
    "P3 macro + crédito + did + tendências":
        (["selic_lag1", "pib_lag1", "tx_cambio", "credito_c", "did",
          "t", "t_agro"], pa_cred),
}
resumo_p = []
for rot, (rgs, dados) in ESPEC_PAINEL.items():
    tab, n = tabela_vif(dados, rgs, dummies_setor=True)
    cond = numero_condicao(dados, rgs, dummies_setor=True)
    print(f"\n{rot}   (N = {n}, número de condição = {br(cond, 1)})")
    print(tab.round(3).to_string(index=False))
    exportar(tab.round(4), f"tab_vif_{rot.split()[0].lower()}", indice=False)
    v = tab.loc[tab["variavel"] == "did", "vif"]
    resumo_p.append({"especificacao": rot, "N": n, "k": len(tab),
                     "vif_maximo": tab["vif"].max(),
                     "variavel_critica": tab.loc[tab["vif"].idxmax(), "variavel"],
                     "vif_did": float(v.iloc[0]) if len(v) else np.nan,
                     "numero_condicao": cond})
tab_resumo_p = pd.DataFrame(resumo_p).set_index("especificacao")
print("\nResumo — painel:")
print(tab_resumo_p.round(3).to_string())
exportar(tab_resumo_p.round(4), "tab_vif_resumo_painel")
v_sem = tab_resumo_p.loc["P2 macro + crédito + did", "vif_did"]
v_com = tab_resumo_p.loc["P3 macro + crédito + did + tendências", "vif_did"]
print(f"\nVIF do termo de tratamento:")
print(f"  sem tendência setorial : {br(v_sem, 2)}")
print(f"  com tendência setorial : {br(v_com, 2)}")
print(f"  fator de inflação      : {br(v_com / v_sem, 1)} vez(es)")
print("O termo de tratamento e a tendência do grupo tratado disputam a mesma")
print("variação. A especificação com tendência é a correta quando as")
print("trajetórias anteriores divergem, ao custo de precisão.")

titulo("5.2 SETOR TRATADO — ESPECIFICAÇÕES", "-")
ESPEC_AGRO = {
    "A1 macro": ["selic_lag1", "pib_lag1", "tx_cambio"],
    "A2 macro + soja": ["selic_lag1", "pib_lag1", "tx_cambio", "soja_agro"],
    "A3 macro + crédito": ["selic_lag1", "pib_lag1", "tx_cambio", "credito_c"],
    "A4 saturada": ["selic_lag1", "pib_lag1", "tx_cambio", "soja_agro",
                    "credito_c", "pos_lei"],
    "A5 câmbio + soja + crédito + lei": ["tx_cambio", "soja_agro",
                                         "credito_c", "pos_lei"],
    "A6 câmbio + crédito + lei": ["tx_cambio", "credito_c", "pos_lei"],
    "A7 câmbio + lei": ["tx_cambio", "pos_lei"],
    "A8 câmbio + crédito + lei + tendência": ["tx_cambio", "credito_c",
                                              "pos_lei", "t"],
    "A9 completa com tendência": ["tx_cambio", "soja_agro", "credito_c",
                                  "pos_lei", "t"],
}
resumo_a, detalhes = [], {}
for rot, rgs in ESPEC_AGRO.items():
    base = ag_cred if "credito_c" in rgs else ag_est
    tab, n = tabela_vif(base, rgs)
    detalhes[rot] = (tab, n)
    cond = numero_condicao(base, rgs)
    v = tab.loc[tab["variavel"] == "pos_lei", "vif"]
    resumo_a.append({"especificacao": rot, "N": n, "k_regressores": len(rgs),
                     "gl_residuais": n - len(rgs) - 1,
                     "vif_maximo": tab["vif"].max(),
                     "variavel_critica": tab.loc[tab["vif"].idxmax(), "variavel"],
                     "vif_pos_lei": float(v.iloc[0]) if len(v) else np.nan,
                     "numero_condicao": cond,
                     "situacao": "estimável" if tab["vif"].max() < LIMIAR_SEVERO
                                 else "colinearidade severa"})
tab_resumo_a = pd.DataFrame(resumo_a).set_index("especificacao")
print(tab_resumo_a.round(3).to_string())
exportar(tab_resumo_a.round(4), "tab_vif_resumo_agro")
print("\nDetalhamento das especificações com colinearidade severa:")
for rot, (tab, n) in detalhes.items():
    if tab["vif"].max() >= LIMIAR_SEVERO:
        print(f"\n{rot}   (N = {n})")
        print(tab.round(3).to_string(index=False))
estimaveis = tab_resumo_a[tab_resumo_a["situacao"] == "estimável"].index.tolist()
print(f"\nEspecificações sem colinearidade severa: {len(estimaveis)} de "
      f"{len(ESPEC_AGRO)}")
for e in estimaveis:
    print(f"  {e}")

titulo("5.3 SEPARABILIDADE DO INDICADOR DE VIGÊNCIA", "-")


def r2_auxiliar(dados, alvo, outros):
    d = dados[[alvo] + outros].dropna()
    if d[alvo].nunique() < 2:
        return np.nan, len(d)
    m = sm.OLS(d[alvo], sm.add_constant(d[outros].astype(float))).fit()
    return m.rsquared, len(d)


print("Setor tratado — variação de pos_lei explicada por outros regressores:")
linhas = []
for rot, conj in [("tendência apenas", ["t"]),
                  ("câmbio apenas", ["tx_cambio"]),
                  ("crédito apenas", ["credito_c"]),
                  ("soja apenas", ["soja_agro"]),
                  ("câmbio + crédito", ["tx_cambio", "credito_c"]),
                  ("câmbio + crédito + tendência",
                   ["tx_cambio", "credito_c", "t"]),
                  ("todos os regressores",
                   ["selic_lag1", "pib_lag1", "tx_cambio", "soja_agro",
                    "credito_c", "t"])]:
    r2, n = r2_auxiliar(ag_cred, "pos_lei", conj)
    linhas.append({"conjunto": rot, "N": n, "r2": r2,
                   "vif_implicito": 1 / (1 - r2) if r2 < 1 else np.inf})
tab_sep = pd.DataFrame(linhas).set_index("conjunto")
print(tab_sep.round(4).to_string())
exportar(tab_sep.round(4), "tab_separabilidade_pos_lei")
r2_t = tab_sep.loc["tendência apenas", "r2"]
print(f"\nA tendência temporal reproduz {br(r2_t*100, 1)}% da variação do")
print(f"indicador de vigência. Com {len(ag_cred)} observações anuais e")
print(f"tratamento nos últimos {int(ag_cred['pos_lei'].sum())} anos, os dois")
print("termos descrevem em grande medida a mesma variação.")
print("\nPainel — variação de did explicada por outros regressores:")
linhas = []
for rot, conj in [("tendência do grupo tratado", ["t_agro"]),
                  ("tendência geral + do grupo", ["t", "t_agro"]),
                  ("macro + crédito", ["selic_lag1", "pib_lag1",
                                       "tx_cambio", "credito_c"]),
                  ("macro + crédito + tendências",
                   ["selic_lag1", "pib_lag1", "tx_cambio", "credito_c",
                    "t", "t_agro"])]:
    r2, n = r2_auxiliar(pa_cred, "did", conj)
    linhas.append({"conjunto": rot, "N": n, "r2": r2,
                   "vif_implicito": 1 / (1 - r2) if r2 < 1 else np.inf})
tab_sep_p = pd.DataFrame(linhas).set_index("conjunto")
print(tab_sep_p.round(4).to_string())
exportar(tab_sep_p.round(4), "tab_separabilidade_did")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
rot_p = [r.split()[0] for r in tab_resumo_p.index]
ax.bar(rot_p, tab_resumo_p["vif_maximo"], color="#2C6FAC", alpha=0.85,
       edgecolor="white", label="VIF máximo")
ax.bar(rot_p, tab_resumo_p["vif_did"], color="#E07B39", alpha=0.9, width=0.45,
       edgecolor="white", label="VIF do termo de tratamento")
ax.axhline(LIMIAR_ATENCAO, color="orange", linestyle="--", linewidth=1.2)
ax.axhline(LIMIAR_SEVERO, color=COR_LEI, linestyle="--", linewidth=1.2)
rotular_painel(ax, "A")
ax.set_ylabel("VIF")
ax.legend(fontsize=9)
ax = axes[1]
rot_a = [r.split()[0] for r in tab_resumo_a.index]
cores = ["#27AE60" if s == "estimável" else "#BDC3C7"
         for s in tab_resumo_a["situacao"]]
ax.bar(rot_a, tab_resumo_a["vif_maximo"], color=cores, alpha=0.9,
       edgecolor="white")
ax.axhline(LIMIAR_ATENCAO, color="orange", linestyle="--", linewidth=1.2,
           label=f"atenção ({LIMIAR_ATENCAO:.0f})")
ax.axhline(LIMIAR_SEVERO, color=COR_LEI, linestyle="--", linewidth=1.2,
           label=f"severo ({LIMIAR_SEVERO:.0f})")
rotular_painel(ax, "B")
ax.set_ylabel("VIF")
ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
salvar_fig("fig_vif_especificacoes")
print(f"\nFigura: fig_vif_especificacoes")
titulo("ETAPAS 1 A 5 CONCLUÍDAS")
print(f"Arquivos gerados em: {DIR_SAIDA}")

# ==============================================================
# ETAPA 6 — TRAJETÓRIAS PRÉ-TRATAMENTO
# ==============================================================
titulo("ETAPA 6 — TRAJETÓRIAS PRÉ-TRATAMENTO")
import statsmodels.formula.api as smf

DESFECHOS = {"log_rj": "log das recuperações",
             "log_fal": "log das falências",
             "log_odds": "log da razão entre ambos"}
HAC = {"cov_type": "HAC",
       "cov_kwds": {"maxlags": MAXLAGS_PAINEL, "use_correction": USE_CORRECTION}}
pre = pa[pa["ano"] < ANO_LEI].dropna(subset=list(DESFECHOS)).copy()
pre["t_pre"] = pre["ano"] - pre["ano"].min()
pre = pre.sort_values(["setor", "ano"]).reset_index(drop=True)
print(f"Amostra pré-tratamento: {len(pre)} observações, "
      f"{pre['ano'].min()}–{pre['ano'].max()}, {pre['setor'].nunique()} setores")
print(f"Anos por setor: {pre.groupby('setor')['ano'].count().unique().tolist()}")


def tabela_coef(mod, nome, rotulos=None):
    ci = mod.conf_int()
    tab = pd.DataFrame({"coeficiente": mod.params, "erro_padrao": mod.bse,
                        "estatistica": mod.tvalues, "p_valor": mod.pvalues,
                        "ic_95_inf": ci[0], "ic_95_sup": ci[1]})
    tab["significancia"] = tab["p_valor"].apply(sig_stars)
    if rotulos:
        tab.index = [rotulos.get(i, i) for i in tab.index]
    tab.index.name = "termo"
    exportar(tab.round(4), f"tab_{nome}")
    return tab.round(4)


titulo("6.1 DIFERENCIAL DE TENDÊNCIA DO GRUPO TRATADO", "-")
resumo_61 = []
for var, rot in DESFECHOS.items():
    mod = smf.ols(f"{var} ~ C(setor) + t_pre + t_pre:agro", data=pre).fit(**HAC)
    print(f"\n{rot}   (N = {int(mod.nobs)}, gl = {int(mod.df_resid)}, "
          f"R² aj. = {br(mod.rsquared_adj)})")
    print(tabela_coef(mod, f"pretend_{var}").to_string())
    c, p = mod.params["t_pre:agro"], mod.pvalues["t_pre:agro"]
    ci = mod.conf_int().loc["t_pre:agro"]
    resumo_61.append({"desfecho": var, "descricao": rot,
                      "tendencia_controles": mod.params["t_pre"],
                      "diferencial_agro": c, "erro_padrao": mod.bse["t_pre:agro"],
                      "p_valor": p, "significancia": sig_stars(p),
                      "ic_95_inf": ci[0], "ic_95_sup": ci[1]})
tab_61 = pd.DataFrame(resumo_61).set_index("desfecho")
print("\nSíntese — diferencial de tendência do grupo tratado:")
print(tab_61.round(4).to_string())
exportar(tab_61.round(4), "tab_pretendencia_sintese")
print("\nLeitura:")
for _, r in tab_61.iterrows():
    veredito = ("as trajetórias divergiam" if r["p_valor"] < 0.10
                else "não se detecta divergência")
    print(f"  {r['descricao']:<28} diferencial {br(r['diferencial_agro']):>7} "
          f"(p = {br(r['p_valor'], 4)})  →  {veredito}")

titulo("6.2 TENDÊNCIAS POR SETOR E COMPARAÇÕES PAREADAS", "-")
for var, rot in DESFECHOS.items():
    mod = smf.ols(f"{var} ~ C(setor) * t_pre", data=pre).fit(**HAC)
    print(f"\n{rot}   (N = {int(mod.nobs)}, gl = {int(mod.df_resid)})")
    b_trat = mod.params["t_pre"]
    linhas = [{"setor": SETOR_TRATADO, "tendencia": b_trat,
               "diferenca_vs_tratado": 0.0, "p_valor": np.nan,
               "significancia": "—"}]
    termos = []
    for s in SETORES:
        if s == SETOR_TRATADO:
            continue
        termo = f"C(setor)[T.{s}]:t_pre"
        if termo not in mod.params:
            continue
        termos.append(termo)
        linhas.append({"setor": s, "tendencia": b_trat + mod.params[termo],
                       "diferenca_vs_tratado": mod.params[termo],
                       "p_valor": mod.pvalues[termo],
                       "significancia": sig_stars(mod.pvalues[termo])})
    tab = pd.DataFrame(linhas).set_index("setor")
    print(tab.round(4).to_string())
    exportar(tab.round(4), f"tab_tendencias_setor_{var}")
    if termos:
        w = mod.f_test(", ".join(f"{t} = 0" for t in termos))
        est, pv = float(np.ravel(w.statistic)[0]), float(w.pvalue)
        print(f"  teste conjunto de igualdade das tendências: "
              f"estatística = {br(est, 3)}, p = {br(pv, 4)}")
        print(f"  {'rejeita-se' if pv < 0.10 else 'não se rejeita'} a hipótese de "
              f"trajetórias comparáveis")

titulo("6.3 GRUPO TRATADO CONTRA CADA CONTROLE", "-")
linhas = []
for var, rot in DESFECHOS.items():
    for s in SETORES:
        if s == SETOR_TRATADO:
            continue
        sub = pre[pre["setor"].isin([SETOR_TRATADO, s])].copy()
        mod = smf.ols(f"{var} ~ agro + t_pre + t_pre:agro", data=sub).fit(
            cov_type="HAC", cov_kwds={"maxlags": 1,
                                      "use_correction": USE_CORRECTION})
        c, p = mod.params["t_pre:agro"], mod.pvalues["t_pre:agro"]
        linhas.append({"desfecho": var, "controle": s, "N": int(mod.nobs),
                       "diferencial": c, "p_valor": p,
                       "significancia": sig_stars(p)})
tab_63 = pd.DataFrame(linhas)
print(tab_63.round(4).to_string(index=False))
exportar(tab_63.round(4), "tab_pretendencia_pareada", indice=False)
print("\nProporção de comparações com divergência detectada, por desfecho:")
for var, rot in DESFECHOS.items():
    sub = tab_63[tab_63["desfecho"] == var]
    print(f"  {rot:<28} {int((sub['p_valor'] < 0.10).sum())} de {len(sub)}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (var, rot) in zip(axes, DESFECHOS.items()):
    for s in SETORES:
        d = pre[pre["setor"] == s]
        ax.plot(d["ano"], d[var], marker="o", color=CORES_SETOR[s],
                label=s.capitalize(), linewidth=2, alpha=0.85)
        b = np.polyfit(d["ano"], d[var], 1)
        ax.plot(d["ano"], np.polyval(b, d["ano"]), color=CORES_SETOR[s],
                linestyle=":", linewidth=1.4)
    rotular_painel(ax, "ABC"[list(axes).index(ax)])
    ax.set_xlabel("Ano")
    ax.set_ylabel(var)
    ax.legend(fontsize=8)
plt.tight_layout()
salvar_fig("fig_pretendencias")
print("\nFigura: fig_pretendencias")

# ==============================================================
# ETAPA 7 — ESTUDO DE EVENTO
# ==============================================================
titulo("ETAPA 7 — ESTUDO DE EVENTO")
ev = pa.dropna(subset=list(DESFECHOS)).copy().reset_index(drop=True)
CONTROLES = [s for s in SETORES if s != SETOR_TRATADO]
SETOR_REF  = CONTROLES[0]
setores_fe = [s for s in SETORES if s != SETOR_REF]
anos_ev    = sorted(ev["ano"].unique())
ano_base_fe = anos_ev[0]
anos_fe     = [a for a in anos_ev if a != ano_base_fe]
anos_inter  = [a for a in anos_ev if a != ANO_BASE_EVENT]
X = pd.DataFrame(index=ev.index)
X["const"] = 1.0
for s in setores_fe:
    X[f"setor_{s}"] = (ev["setor"] == s).astype(float)
for a in anos_fe:
    X[f"ano_{a}"] = (ev["ano"] == a).astype(float)
for a in anos_inter:
    X[f"agro_x_{a}"] = ((ev["ano"] == a) & (ev["agro"] == 1)).astype(float)
rank = int(np.linalg.matrix_rank(X.values))
print(f"Setor de referência              : {SETOR_REF}")
print(f"Ano-base dos efeitos fixos       : {ano_base_fe}")
print(f"Ano de referência das interações : {ANO_BASE_EVENT}")
print(f"N = {len(X)} | parâmetros = {X.shape[1]} | rank = {rank} | "
      f"gl residuais = {len(X) - rank}")
if rank < X.shape[1]:
    nulas = [c for c in X.columns if X[c].abs().sum() == 0]
    print(f"ATENÇÃO: rank inferior ao número de colunas. Colunas nulas: {nulas}")
cols_inter = [c for c in X.columns if c.startswith("agro_x_")]
n_trat = {c: int(X[c].sum()) for c in cols_inter}
if min(n_trat.values()) == 0:
    print(f"ATENÇÃO: anos sem observação do grupo tratado: "
          f"{[c for c, v in n_trat.items() if v == 0]}")

titulo("7.1 COEFICIENTES ANO A ANO", "-")
evento = {}
for var, rot in DESFECHOS.items():
    mod = sm.OLS(ev[var], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": MAXLAGS_PAINEL,
                                  "use_correction": USE_CORRECTION})
    ci = mod.conf_int()
    tab = pd.DataFrame({
        "ano": [int(c.split("_")[-1]) for c in cols_inter],
        "coeficiente": [mod.params[c] for c in cols_inter],
        "erro_padrao": [mod.bse[c] for c in cols_inter],
        "p_valor": [mod.pvalues[c] for c in cols_inter],
        "ic_95_inf": [ci.loc[c, 0] for c in cols_inter],
        "ic_95_sup": [ci.loc[c, 1] for c in cols_inter]})
    tab["periodo"] = np.where(tab["ano"] < ANO_LEI, "pré", "pós")
    tab["significancia"] = tab["p_valor"].apply(sig_stars)
    tab = tab.sort_values("ano").reset_index(drop=True)
    evento[var] = (mod, tab)
    print(f"\n{rot}   (R² aj. = {br(mod.rsquared_adj)})")
    print(tab.round(3).to_string(index=False))
    exportar(tab.round(4), f"tab_evento_{var}", indice=False)
    pre_t, pos_t = tab[tab.periodo == "pré"], tab[tab.periodo == "pós"]
    print(f"  pré-vigência: {br(pre_t['coeficiente'].min())} a "
          f"{br(pre_t['coeficiente'].max())} | "
          f"{int((pre_t['p_valor'] < 0.10).sum())} de {len(pre_t)} com p < 0,10")
    print(f"  pós-vigência: {br(pos_t['coeficiente'].min())} a "
          f"{br(pos_t['coeficiente'].max())} | "
          f"{int((pos_t['p_valor'] < 0.10).sum())} de {len(pos_t)} com p < 0,10")

titulo("7.2 TESTE CONJUNTO DAS INTERAÇÕES PRÉ-VIGÊNCIA", "-")
cols_pre = [c for c in cols_inter if int(c.split("_")[-1]) < ANO_LEI]
cols_pos = [c for c in cols_inter if int(c.split("_")[-1]) >= ANO_LEI]
linhas = []
for var, rot in DESFECHOS.items():
    mod = evento[var][0]
    w_pre = mod.f_test(", ".join(f"{c} = 0" for c in cols_pre))
    w_pos = mod.f_test(", ".join(f"{c} = 0" for c in cols_pos))
    linhas.append({
        "desfecho": var, "descricao": rot,
        "n_restricoes_pre": len(cols_pre),
        "estatistica_pre": float(np.ravel(w_pre.statistic)[0]),
        "p_valor_pre": float(w_pre.pvalue),
        "n_restricoes_pos": len(cols_pos),
        "estatistica_pos": float(np.ravel(w_pos.statistic)[0]),
        "p_valor_pos": float(w_pos.pvalue)})
tab_wald = pd.DataFrame(linhas).set_index("desfecho")
print(tab_wald.round(4).to_string())
exportar(tab_wald.round(4), "tab_wald_evento")
print("\nLeitura:")
for _, r in tab_wald.iterrows():
    v_pre = ("rejeita-se a nulidade" if r["p_valor_pre"] < 0.10
             else "não se rejeita a nulidade")
    print(f"  {r['descricao']:<28} pré: {v_pre} (p = {br(r['p_valor_pre'], 4)})")

titulo("7.3 VERSÃO AGREGADA EM BLOCOS", "-")
BLOCOS = {"2012-2015": (2012, 2015), "2016-2020": (2016, 2020),
          "2021-2022": (2021, 2022), "2023-2025": (2023, 2025)}
BLOCO_REF = "2016-2020"


def rotular(a):
    for rot, (i, f) in BLOCOS.items():
        if i <= a <= f:
            return rot
    return None


ev["bloco"] = ev["ano"].apply(rotular)
if ev["bloco"].isna().any():
    raise RuntimeError("Há anos fora dos blocos definidos.")
Xb = pd.DataFrame(index=ev.index)
Xb["const"] = 1.0
for s in setores_fe:
    Xb[f"setor_{s}"] = (ev["setor"] == s).astype(float)
for b in list(BLOCOS)[1:]:
    Xb[f"bloco_{b}"] = (ev["bloco"] == b).astype(float)
blocos_inter = [b for b in BLOCOS if b != BLOCO_REF]
for b in blocos_inter:
    Xb[f"agro_x_{b}"] = ((ev["bloco"] == b) & (ev["agro"] == 1)).astype(float)
print(f"Bloco de referência: {BLOCO_REF}")
print(f"N = {len(Xb)} | parâmetros = {Xb.shape[1]} | "
      f"rank = {int(np.linalg.matrix_rank(Xb.values))}")
for var, rot in DESFECHOS.items():
    mod = sm.OLS(ev[var], Xb).fit(
        cov_type="HAC", cov_kwds={"maxlags": MAXLAGS_PAINEL,
                                  "use_correction": USE_CORRECTION})
    ci = mod.conf_int()
    cols = [c for c in Xb.columns if c.startswith("agro_x_")]
    tab = pd.DataFrame({
        "bloco": [c.replace("agro_x_", "") for c in cols],
        "coeficiente": [mod.params[c] for c in cols],
        "erro_padrao": [mod.bse[c] for c in cols],
        "p_valor": [mod.pvalues[c] for c in cols],
        "ic_95_inf": [ci.loc[c, 0] for c in cols],
        "ic_95_sup": [ci.loc[c, 1] for c in cols]})
    tab["significancia"] = tab["p_valor"].apply(sig_stars)
    print(f"\n{rot}   (R² aj. = {br(mod.rsquared_adj)})")
    print(tab.round(4).to_string(index=False))
    exportar(tab.round(4), f"tab_evento_blocos_{var}", indice=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (var, rot) in zip(axes, DESFECHOS.items()):
    tab = evento[var][1]
    cores = ["#2C6FAC" if p == "pré" else COR_AGRO for p in tab["periodo"]]
    ax.errorbar(tab["ano"], tab["coeficiente"],
                yerr=[tab["coeficiente"] - tab["ic_95_inf"],
                      tab["ic_95_sup"] - tab["coeficiente"]],
                fmt="none", ecolor="gray", elinewidth=1.2, capsize=3, zorder=2)
    ax.scatter(tab["ano"], tab["coeficiente"], c=cores, s=55, zorder=3,
               edgecolor="white", linewidth=0.8)
    ax.scatter([ANO_BASE_EVENT], [0], marker="D", s=70, color="black", zorder=4)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(ANO_LEI - 0.5, color=COR_LEI, linestyle="--", linewidth=1.5)
    rotular_painel(ax, "ABC"[list(axes).index(ax)])
    ax.set_xlabel("Ano")
    ax.set_ylabel("Coeficiente da interação")

axes[0].plot([], [], "o", color="#2C6FAC", label="anterior à vigência")
axes[0].plot([], [], "o", color=COR_AGRO, label="posterior à vigência")
axes[0].plot([], [], "D", color="black", label=f"referência ({ANO_BASE_EVENT})")
axes[0].legend(fontsize=8, loc="upper left")
plt.tight_layout()
salvar_fig("fig_evento")
print("\nFigura: fig_evento")

# ==============================================================
# ETAPA 8 — MODELOS DE PAINEL
# ==============================================================
titulo("ETAPA 8 — MODELOS DE PAINEL")
CLUSTER = {"cov_type": "cluster", "cov_kwds": {"groups": pa_est["setor"]}}
ESPEC = {
    "A1 macro + did":
        "C(setor) + selic_lag1 + pib_lag1 + tx_cambio + did",
    "A2 macro + did + tendências":
        "C(setor) + selic_lag1 + pib_lag1 + tx_cambio + did + t + t_agro",
    "B1 efeitos fixos de ano + did":
        "C(setor) + C(ano) + did",
    "B2 efeitos fixos de ano + did + tendência do tratado":
        "C(setor) + C(ano) + did + t_agro",
}
ROT_PAINEL = {"Intercept": "intercepto",
              "C(setor)[T.comercio]": "comercio",
              "C(setor)[T.industria]": "industria",
              "C(setor)[T.servicos]": "servicos",
              "did": "did (tratamento)"}


def ajustar(formula, dados, robusto="hac"):
    if robusto == "hac":
        return smf.ols(formula, data=dados).fit(**HAC)
    if robusto == "cluster":
        return smf.ols(formula, data=dados).fit(
            cov_type="cluster", cov_kwds={"groups": dados["setor"]})
    if robusto == "hc3":
        return smf.ols(formula, data=dados).fit(cov_type="HC3")
    return smf.ols(formula, data=dados).fit()


titulo("8.1 COEFICIENTE DE TRATAMENTO", "-")
linhas = []
modelos = {}
for var, rot_d in DESFECHOS.items():
    for rot_e, form in ESPEC.items():
        for robusto in ["hac", "cluster"]:
            mod = ajustar(f"{var} ~ {form}", pa_est, robusto)
            modelos[(var, rot_e, robusto)] = mod
            c, p = mod.params["did"], mod.pvalues["did"]
            ci = mod.conf_int().loc["did"]
            linhas.append({
                "desfecho": var, "especificacao": rot_e,
                "erros": "HAC" if robusto == "hac" else "cluster por setor",
                "N": int(mod.nobs), "gl": int(mod.df_resid),
                "coef_did": c, "erro_padrao": mod.bse["did"],
                "p_valor": p, "significancia": sig_stars(p),
                "ic_95_inf": ci[0], "ic_95_sup": ci[1],
                "efeito_pct": efeito_pct(c),
                "r2_ajustado": mod.rsquared_adj, "aic": mod.aic})
tab_did = pd.DataFrame(linhas)
exportar(tab_did.round(4), "tab_painel_did", indice=False)
for var, rot_d in DESFECHOS.items():
    print(f"\n{rot_d}:")
    sub = tab_did[tab_did["desfecho"] == var].drop(columns=["desfecho"])
    print(sub.round(4).to_string(index=False))

titulo("8.2 COEFICIENTES COMPLETOS", "-")
for var, rot_d in DESFECHOS.items():
    for rot_e in ["A2 macro + did + tendências",
                  "B2 efeitos fixos de ano + did + tendência do tratado"]:
        mod = modelos[(var, rot_e, "hac")]
        print(f"\n{rot_d} — {rot_e}")
        print(f"  N = {int(mod.nobs)}, gl = {int(mod.df_resid)}, "
              f"R² aj. = {br(mod.rsquared_adj)}, AIC = {br(mod.aic, 2)}")
        tab = tabela_coef(mod, f"painel_{var}_{rot_e.split()[0]}", ROT_PAINEL)
        mostrar = [i for i in tab.index if not str(i).startswith("C(ano)")]
        print(tab.loc[mostrar].to_string())

titulo("8.3 COEFICIENTES MACROECONÔMICOS", "-")
linhas = []
for var, rot_d in DESFECHOS.items():
    for rot_e in ["A1 macro + did", "A2 macro + did + tendências"]:
        mod = modelos[(var, rot_e, "hac")]
        for v in ["selic_lag1", "pib_lag1", "tx_cambio"]:
            c, p = mod.params[v], mod.pvalues[v]
            ci = mod.conf_int().loc[v]
            linhas.append({"desfecho": var, "especificacao": rot_e.split()[0],
                           "variavel": v, "coeficiente": c, "p_valor": p,
                           "significancia": sig_stars(p),
                           "ic_95_inf": ci[0], "ic_95_sup": ci[1]})
tab_macro = pd.DataFrame(linhas)
print(tab_macro.round(4).to_string(index=False))
exportar(tab_macro.round(4), "tab_painel_macro", indice=False)
print("\nInterpretação por ponto percentual de Selic (desfecho log):")
for var in DESFECHOS:
    for e in ["A1", "A2"]:
        r = tab_macro[(tab_macro.desfecho == var) &
                      (tab_macro.especificacao == e) &
                      (tab_macro.variavel == "selic_lag1")]
        if len(r):
            c = r.iloc[0]["coeficiente"]
            print(f"  {var:<10} {e}: {br(c/100*100, 3)} log por p.p. "
                  f"→ {br(efeito_pct(c/100), 2)}% (p = "
                  f"{br(r.iloc[0]['p_valor'], 4)})")

titulo("8.4 TENDÊNCIA DO GRUPO TRATADO", "-")
linhas = []
for var, rot_d in DESFECHOS.items():
    for rot_e in ["A2 macro + did + tendências",
                  "B2 efeitos fixos de ano + did + tendência do tratado"]:
        mod = modelos[(var, rot_e, "hac")]
        if "t_agro" not in mod.params:
            continue
        c, p = mod.params["t_agro"], mod.pvalues["t_agro"]
        ci = mod.conf_int().loc["t_agro"]
        linhas.append({"desfecho": var, "especificacao": rot_e.split()[0],
                       "t_agro": c, "p_valor": p,
                       "significancia": sig_stars(p),
                       "ic_95_inf": ci[0], "ic_95_sup": ci[1],
                       "pretendencia_etapa6": float(
                           tab_61.loc[var, "diferencial_agro"])})
tab_tend = pd.DataFrame(linhas)
tab_tend["dentro_do_ic"] = ((tab_tend["pretendencia_etapa6"] >= tab_tend["ic_95_inf"]) &
                            (tab_tend["pretendencia_etapa6"] <= tab_tend["ic_95_sup"]))
print(tab_tend.round(4).to_string(index=False))
exportar(tab_tend.round(4), "tab_painel_tendencia", indice=False)
print("\nA tendência medida na amostra completa contém a medida no período")
print("anterior à vigência dentro do intervalo de 95%? "
      f"{int(tab_tend['dentro_do_ic'].sum())} de {len(tab_tend)} casos")

titulo("8.5 MÉTRICAS DE AJUSTE", "-")
linhas = []
for var in DESFECHOS:
    for rot_e in ESPEC:
        mod = modelos[(var, rot_e, "hac")]
        linhas.append({"desfecho": var, "especificacao": rot_e,
                       "k": int(mod.df_model + 1), "gl": int(mod.df_resid),
                       "r2": mod.rsquared, "r2_ajustado": mod.rsquared_adj,
                       "aic": mod.aic, "bic": mod.bic})
tab_ajuste = pd.DataFrame(linhas)
print(tab_ajuste.round(4).to_string(index=False))
exportar(tab_ajuste.round(4), "tab_painel_ajuste", indice=False)
print("\nEspecificação de melhor ajuste por desfecho, pelo critério AIC:")
for var, rot_d in DESFECHOS.items():
    sub = tab_ajuste[tab_ajuste["desfecho"] == var]
    melhor = sub.loc[sub["aic"].idxmin()]
    print(f"  {rot_d:<28} {melhor['especificacao']} "
          f"(AIC = {br(melhor['aic'], 2)}, R² aj. = {br(melhor['r2_ajustado'])})")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
for ax, (var, rot_d) in zip(axes, DESFECHOS.items()):
    sub = tab_did[(tab_did["desfecho"] == var) &
                  (tab_did["erros"] == "HAC")].reset_index(drop=True)
    y = np.arange(len(sub))
    ax.errorbar(sub["coef_did"], y,
                xerr=[sub["coef_did"] - sub["ic_95_inf"],
                      sub["ic_95_sup"] - sub["coef_did"]],
                fmt="o", color=COR_AGRO, ecolor="gray", elinewidth=1.4,
                capsize=4, markersize=8)
    ax.axvline(0, color=COR_LEI, linestyle="--", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([e.split()[0] for e in sub["especificacao"]])
    ax.invert_yaxis()
    rotular_painel(ax, "ABC"[list(axes).index(ax)])
    ax.set_xlabel("Coeficiente de tratamento")
plt.tight_layout()
salvar_fig("fig_painel_did")
print("\nFigura: fig_painel_did")

# ==============================================================
# ETAPA 9 — ESCADA DE ESPECIFICAÇÕES DO SETOR TRATADO
# ==============================================================
titulo("ETAPA 9 — ESCADA DE ESPECIFICAÇÕES DO SETOR TRATADO")
HAC_AGRO = {"cov_type": "HAC",
            "cov_kwds": {"maxlags": MAXLAGS_AGRO,
                         "use_correction": USE_CORRECTION}}

titulo("9.1 COEFICIENTE DO INDICADOR DE VIGÊNCIA", "-")
linhas, mod_agro = [], {}
for var, rot_d in DESFECHOS.items():
    for rot_e, regs in ESPEC_AGRO.items():
        base = ag_cred if "credito_c" in regs else ag_est
        form = f"{var} ~ " + " + ".join(regs)
        mod = smf.ols(form, data=base).fit(**HAC_AGRO)
        mod_agro[(var, rot_e)] = mod
        sit = tab_resumo_a.loc[rot_e, "situacao"]
        vif_max = tab_resumo_a.loc[rot_e, "vif_maximo"]
        reg = {"desfecho": var, "especificacao": rot_e,
               "N": int(mod.nobs), "k": int(mod.df_model + 1),
               "gl": int(mod.df_resid),
               "tem_tendencia": "sim" if "t" in regs else "não",
               "tem_indicador": "sim" if "pos_lei" in regs else "não",
               "vif_maximo": vif_max, "situacao": sit,
               "r2_ajustado": mod.rsquared_adj, "aic": mod.aic}
        if "pos_lei" in mod.params:
            c, p = mod.params["pos_lei"], mod.pvalues["pos_lei"]
            ci = mod.conf_int().loc["pos_lei"]
            reg.update({"coef_pos_lei": c, "erro_padrao": mod.bse["pos_lei"],
                        "p_valor": p, "significancia": sig_stars(p),
                        "ic_95_inf": ci[0], "ic_95_sup": ci[1],
                        "efeito_pct": efeito_pct(c)})
        else:
            reg.update({"coef_pos_lei": np.nan, "erro_padrao": np.nan,
                        "p_valor": np.nan, "significancia": "—",
                        "ic_95_inf": np.nan, "ic_95_sup": np.nan,
                        "efeito_pct": np.nan})
        linhas.append(reg)
tab_agro = pd.DataFrame(linhas)
exportar(tab_agro.round(4), "tab_agro_escada", indice=False)
COLS_MOSTRA = ["especificacao", "N", "k", "gl", "tem_tendencia", "vif_maximo",
               "situacao", "coef_pos_lei", "p_valor", "significancia",
               "ic_95_inf", "ic_95_sup", "r2_ajustado", "aic"]
for var, rot_d in DESFECHOS.items():
    print(f"\n{rot_d}:")
    sub = tab_agro[(tab_agro["desfecho"] == var) &
                   (tab_agro["tem_indicador"] == "sim")]
    print(sub[COLS_MOSTRA].round(4).to_string(index=False))

titulo("9.2 AMPLITUDE DO COEFICIENTE POR DESFECHO", "-")
linhas = []
for var, rot_d in DESFECHOS.items():
    sub = tab_agro[(tab_agro["desfecho"] == var) &
                   tab_agro["coef_pos_lei"].notna()]
    est = sub[sub["situacao"] == "estimável"]
    linhas.append({
        "desfecho": var, "descricao": rot_d,
        "n_especificacoes": len(sub),
        "minimo": sub["coef_pos_lei"].min(),
        "maximo": sub["coef_pos_lei"].max(),
        "amplitude": sub["coef_pos_lei"].max() - sub["coef_pos_lei"].min(),
        "n_positivos": int((sub["coef_pos_lei"] > 0).sum()),
        "n_significativos": int((sub["p_valor"] < 0.10).sum()),
        "min_estimaveis": est["coef_pos_lei"].min() if len(est) else np.nan,
        "max_estimaveis": est["coef_pos_lei"].max() if len(est) else np.nan})
tab_ampl = pd.DataFrame(linhas).set_index("desfecho")
print(tab_ampl.round(4).to_string())
exportar(tab_ampl.round(4), "tab_agro_amplitude")

titulo("9.3 EFEITO DA INCLUSÃO DO TERMO DE TENDÊNCIA", "-")
linhas = []
for var, rot_d in DESFECHOS.items():
    sub = tab_agro[(tab_agro["desfecho"] == var) &
                   tab_agro["coef_pos_lei"].notna()]
    for tem in ["não", "sim"]:
        g = sub[sub["tem_tendencia"] == tem]
        if not len(g):
            continue
        linhas.append({
            "desfecho": var, "descricao": rot_d,
            "termo_de_tendencia": tem, "n_modelos": len(g),
            "coef_medio": g["coef_pos_lei"].mean(),
            "coef_min": g["coef_pos_lei"].min(),
            "coef_max": g["coef_pos_lei"].max(),
            "n_positivos": int((g["coef_pos_lei"] > 0).sum()),
            "n_significativos": int((g["p_valor"] < 0.10).sum()),
            "n_com_colinearidade_severa":
                int((g["situacao"] != "estimável").sum())})
tab_contraste = pd.DataFrame(linhas)
print(tab_contraste.round(4).to_string(index=False))
exportar(tab_contraste.round(4), "tab_agro_contraste", indice=False)
print("\nLeitura por desfecho:")
for var, rot_d in DESFECHOS.items():
    sem = tab_contraste[(tab_contraste.desfecho == var) &
                        (tab_contraste.termo_de_tendencia == "não")]
    com = tab_contraste[(tab_contraste.desfecho == var) &
                        (tab_contraste.termo_de_tendencia == "sim")]
    if len(sem) and len(com):
        print(f"  {rot_d}")
        print(f"    sem tendência: média {br(sem.iloc[0]['coef_medio']):>7}  "
              f"({sem.iloc[0]['n_positivos']} de {sem.iloc[0]['n_modelos']} positivos)")
        print(f"    com tendência: média {br(com.iloc[0]['coef_medio']):>7}  "
              f"({com.iloc[0]['n_positivos']} de {com.iloc[0]['n_modelos']} positivos, "
              f"{com.iloc[0]['n_com_colinearidade_severa']} com colinearidade severa)")

titulo("9.4 ESPECIFICAÇÕES ESTIMÁVEIS COM INDICADOR DE VIGÊNCIA", "-")
estim_com_ind = tab_agro[(tab_agro["situacao"] == "estimável") &
                         (tab_agro["tem_indicador"] == "sim")
                         ]["especificacao"].unique().tolist()
print(f"Especificações estimáveis que contêm o indicador: {estim_com_ind}")
for var, rot_d in DESFECHOS.items():
    for rot_e in estim_com_ind:
        mod = mod_agro[(var, rot_e)]
        print(f"\n{rot_d} — {rot_e}")
        print(f"  N = {int(mod.nobs)}, gl = {int(mod.df_resid)}, "
              f"R² aj. = {br(mod.rsquared_adj)}, AIC = {br(mod.aic, 2)}")
        print(tabela_coef(mod, f"agro_{var}_{rot_e.split()[0]}").to_string())

titulo("9.5 CONFRONTO ENTRE SÉRIE ISOLADA E PAINEL", "-")
linhas = []
for var, rot_d in DESFECHOS.items():
    p_a2 = modelos[(var, "A2 macro + did + tendências", "hac")]
    ci_p = p_a2.conf_int().loc["did"]
    sub = tab_agro[(tab_agro["desfecho"] == var) &
                   (tab_agro["situacao"] == "estimável") &
                   tab_agro["coef_pos_lei"].notna()]
    linhas.append({
        "desfecho": var, "descricao": rot_d,
        "painel_N": int(p_a2.nobs), "painel_coef": p_a2.params["did"],
        "painel_p": p_a2.pvalues["did"],
        "painel_ic_inf": ci_p[0], "painel_ic_sup": ci_p[1],
        "agro_N": int(sub["N"].max()) if len(sub) else np.nan,
        "agro_coef_min": sub["coef_pos_lei"].min() if len(sub) else np.nan,
        "agro_coef_max": sub["coef_pos_lei"].max() if len(sub) else np.nan})
tab_conf = pd.DataFrame(linhas).set_index("desfecho")
print(tab_conf.round(4).to_string())
exportar(tab_conf.round(4), "tab_agro_vs_painel")
print("\nO painel controla tendência e estima o termo de tratamento na mesma")
print("especificação; a série isolada não. Por isso o painel é a referência")
print("de inferência e a série isolada entra como análise de sensibilidade.")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (var, rot_d) in zip(axes, DESFECHOS.items()):
    sub = tab_agro[(tab_agro["desfecho"] == var) &
                   tab_agro["coef_pos_lei"].notna()].reset_index(drop=True)
    y = np.arange(len(sub))
    cores = ["#27AE60" if s == "estimável" else "#BDC3C7"
             for s in sub["situacao"]]
    ax.errorbar(sub["coef_pos_lei"], y,
                xerr=[sub["coef_pos_lei"] - sub["ic_95_inf"],
                      sub["ic_95_sup"] - sub["coef_pos_lei"]],
                fmt="none", ecolor="gray", elinewidth=1.3, capsize=3)
    ax.scatter(sub["coef_pos_lei"], y, c=cores, s=80, zorder=3,
               edgecolor="black", linewidth=0.6)
    ax.axvline(0, color=COR_LEI, linestyle="--", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e.split()[0]} ({t})" for e, t in
                        zip(sub["especificacao"], sub["tem_tendencia"])],
                       fontsize=8)
    ax.invert_yaxis()
    rotular_painel(ax, "ABC"[list(axes).index(ax)])
    ax.set_xlabel("Coeficiente do indicador de vigência")
plt.tight_layout()
salvar_fig("fig_agro_escada")
print("\nFigura: fig_agro_escada")

# ==============================================================
# ETAPA 10 — ANÁLISES DE ROBUSTEZ
# ==============================================================
titulo("ETAPA 10 — ANÁLISES DE ROBUSTEZ")
FORM_A1 = "C(setor) + selic_lag1 + pib_lag1 + tx_cambio + did"
FORM_A2 = FORM_A1 + " + t + t_agro"
FORM_B1 = "C(setor) + C(ano) + did"

titulo("10.1 MODELOS DE CONTAGEM", "-")
CONTAGENS = {"rj_requeridas": "recuperações requeridas",
             "fal_requeridas": "falências requeridas"}
linhas = []
for var, rot in CONTAGENS.items():
    for rot_e, form in [("A1 sem tendência", FORM_A1),
                        ("A2 com tendência", FORM_A2)]:
        try:
            mp = smf.glm(f"{var} ~ {form}", data=pa_est,
                         family=sm.families.Poisson()).fit(
                cov_type="HAC", cov_kwds={"maxlags": MAXLAGS_PAINEL})
            disp = mp.pearson_chi2 / mp.df_resid
            ci = mp.conf_int().loc["did"]
            linhas.append({"desfecho": var, "descricao": rot,
                           "modelo": "Poisson", "especificacao": rot_e,
                           "coef_did": mp.params["did"],
                           "p_valor": mp.pvalues["did"],
                           "ic_95_inf": ci[0], "ic_95_sup": ci[1],
                           "dispersao": disp})
        except Exception as e:
            print(f"  Poisson {var} {rot_e}: {type(e).__name__}")
        try:
            mn = smf.negativebinomial(f"{var} ~ {form}", data=pa_est).fit(disp=0)
            ci = mn.conf_int().loc["did"]
            linhas.append({"desfecho": var, "descricao": rot,
                           "modelo": "Binomial negativa", "especificacao": rot_e,
                           "coef_did": mn.params["did"],
                           "p_valor": mn.pvalues["did"],
                           "ic_95_inf": ci[0], "ic_95_sup": ci[1],
                           "dispersao": np.nan})
        except Exception as e:
            print(f"  Binomial negativa {var} {rot_e}: {type(e).__name__}")
tab_cont = pd.DataFrame(linhas)
tab_cont["significancia"] = tab_cont["p_valor"].apply(sig_stars)
print(tab_cont.round(4).to_string(index=False))
exportar(tab_cont.round(4), "tab_robustez_contagem", indice=False)
alta = tab_cont[(tab_cont["modelo"] == "Poisson") & (tab_cont["dispersao"] > 2)]
if len(alta):
    print(f"\nSuperdispersão acima de 2 em {len(alta)} ajuste(s) de Poisson.")
    print("A inferência de Poisson é inconsistente nesses casos; a binomial")
    print("negativa é a referência entre os modelos de contagem.")

titulo("10.2 ANOS DE TRATAMENTO FICTÍCIOS", "-")
ANOS_FICTICIOS = [2015, 2016, 2017, 2018, 2019]
placebo = pa[pa["ano"] < ANO_LEI].dropna(
    subset=["log_rj", "log_fal", "log_odds", "selic_lag1",
            "pib_lag1", "tx_cambio"]).copy()
placebo["t"] = placebo["ano"] - placebo["ano"].min()
placebo["t_agro"] = placebo["t"] * placebo["agro"]
print(f"Amostra restrita ao período anterior à vigência: {len(placebo)} obs "
      f"({placebo['ano'].min()}–{placebo['ano'].max()})")
linhas = []
for var, rot in DESFECHOS.items():
    for ano_f in ANOS_FICTICIOS:
        d = placebo.copy()
        d["did"] = d["agro"] * (d["ano"] >= ano_f).astype(int)
        if d["did"].nunique() < 2:
            continue
        for rot_e, form in [("A1 sem tendência", FORM_A1),
                            ("A2 com tendência", FORM_A2)]:
            mod = smf.ols(f"{var} ~ {form}", data=d).fit(**HAC)
            c, p = mod.params["did"], mod.pvalues["did"]
            ci = mod.conf_int().loc["did"]
            linhas.append({"desfecho": var, "ano_ficticio": ano_f,
                           "especificacao": rot_e, "N": int(mod.nobs),
                           "coef_did": c, "p_valor": p,
                           "significancia": sig_stars(p),
                           "ic_95_inf": ci[0], "ic_95_sup": ci[1]})
tab_pl = pd.DataFrame(linhas)
exportar(tab_pl.round(4), "tab_robustez_placebo", indice=False)
for var, rot in DESFECHOS.items():
    print(f"\n{rot}:")
    print(tab_pl[tab_pl["desfecho"] == var].drop(columns=["desfecho"])
          .round(4).to_string(index=False))
print("\nSíntese — coeficientes fictícios significativos a 10%:")
for var, rot in DESFECHOS.items():
    for rot_e in ["A1 sem tendência", "A2 com tendência"]:
        sub = tab_pl[(tab_pl["desfecho"] == var) &
                     (tab_pl["especificacao"] == rot_e)]
        n_sig = int((sub["p_valor"] < 0.10).sum())
        print(f"  {rot:<28} {rot_e:<18} {n_sig} de {len(sub)}   "
              f"média {br(sub['coef_did'].mean())}")
print("\nCoeficientes fictícios elevados na especificação sem tendência, e")
print("próximos de zero na especificação com tendência, indicam que o termo")
print("de tratamento capta trajetória preexistente quando não há controle.")

titulo("10.3 EXCLUSÃO SUCESSIVA DE CONTROLES", "-")
linhas = []
for var, rot in DESFECHOS.items():
    for fora in [None] + CONTROLES:
        d = pa_est if fora is None else pa_est[pa_est["setor"] != fora]
        mod = smf.ols(f"{var} ~ {FORM_A2}", data=d).fit(**HAC)
        c, p = mod.params["did"], mod.pvalues["did"]
        ci = mod.conf_int().loc["did"]
        linhas.append({"desfecho": var, "controle_excluido": fora or "nenhum",
                       "N": int(mod.nobs), "coef_did": c, "p_valor": p,
                       "significancia": sig_stars(p),
                       "ic_95_inf": ci[0], "ic_95_sup": ci[1],
                       "t_agro": mod.params.get("t_agro", np.nan)})
tab_loo = pd.DataFrame(linhas)
print(tab_loo.round(4).to_string(index=False))
exportar(tab_loo.round(4), "tab_robustez_controles", indice=False)
print("\nAmplitude do coeficiente conforme o controle excluído:")
for var, rot in DESFECHOS.items():
    sub = tab_loo[tab_loo["desfecho"] == var]
    print(f"  {rot:<28} {br(sub['coef_did'].min())} a "
          f"{br(sub['coef_did'].max())}  "
          f"({int((sub['p_valor'] < 0.10).sum())} de {len(sub)} significativos)")

titulo("10.4 SENSIBILIDADE À JANELA TEMPORAL", "-")
JANELAS = {
    "completa": lambda d: d,
    "sem 2020 (pandemia)": lambda d: d[d["ano"] != 2020],
    "sem 2024 e 2025": lambda d: d[d["ano"] < 2024],
    "até 2023": lambda d: d[d["ano"] <= 2023],
    "desde 2015": lambda d: d[d["ano"] >= 2015],
}
linhas = []
for var, rot in DESFECHOS.items():
    for rot_j, filtro in JANELAS.items():
        d = filtro(pa_est)
        if d["did"].nunique() < 2 or len(d) < 20:
            continue
        mod = smf.ols(f"{var} ~ {FORM_A2}", data=d).fit(**HAC)
        c, p = mod.params["did"], mod.pvalues["did"]
        ci = mod.conf_int().loc["did"]
        linhas.append({"desfecho": var, "janela": rot_j, "N": int(mod.nobs),
                       "coef_did": c, "p_valor": p,
                       "significancia": sig_stars(p),
                       "ic_95_inf": ci[0], "ic_95_sup": ci[1],
                       "t_agro": mod.params.get("t_agro", np.nan)})
tab_jan = pd.DataFrame(linhas)
print(tab_jan.round(4).to_string(index=False))
exportar(tab_jan.round(4), "tab_robustez_janelas", indice=False)
print("\nAmplitude do coeficiente conforme a janela:")
for var, rot in DESFECHOS.items():
    sub = tab_jan[tab_jan["desfecho"] == var]
    print(f"  {rot:<28} {br(sub['coef_did'].min())} a "
          f"{br(sub['coef_did'].max())}  "
          f"({int((sub['coef_did'] > 0).sum())} de {len(sub)} positivos)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (var, rot) in zip(axes, DESFECHOS.items()):
    for rot_e, cor, marca in [("A1 sem tendência", "#E07B39", "o"),
                              ("A2 com tendência", "#2C6FAC", "s")]:
        sub = tab_pl[(tab_pl["desfecho"] == var) &
                     (tab_pl["especificacao"] == rot_e)]
        ax.errorbar(sub["ano_ficticio"], sub["coef_did"],
                    yerr=[sub["coef_did"] - sub["ic_95_inf"],
                          sub["ic_95_sup"] - sub["coef_did"]],
                    fmt=marca, color=cor, ecolor=cor, alpha=0.8,
                    elinewidth=1.3, capsize=3, markersize=7, label=rot_e)
    ax.axhline(0, color="black", linewidth=1.0)
    rotular_painel(ax, "ABC"[list(axes).index(ax)])
    ax.set_xlabel("Ano de tratamento fictício")
    ax.set_ylabel("Coeficiente")
    ax.legend(fontsize=8)
plt.tight_layout()
salvar_fig("fig_placebo")
print("\nFigura: fig_placebo")

# ==============================================================
# ETAPA 11 — TENDÊNCIA NÃO LINEAR E DIAGNÓSTICO DE RESÍDUOS
# ==============================================================
titulo("ETAPA 11 — TENDÊNCIA NÃO LINEAR E DIAGNÓSTICO")
pa_est = pa_est.copy()
pa_est["t2"] = pa_est["t"] ** 2
pa_est["t2_agro"] = pa_est["t2"] * pa_est["agro"]
FORM_A3 = (FORM_A1 + " + t + t_agro + t2 + t2_agro")

titulo("11.1 TENDÊNCIA LINEAR CONTRA QUADRÁTICA", "-")
linhas = []
for var, rot in DESFECHOS.items():
    for rot_e, form in [("A2 tendência linear", FORM_A2),
                        ("A3 tendência quadrática", FORM_A3)]:
        mod = smf.ols(f"{var} ~ {form}", data=pa_est).fit(**HAC)
        c, p = mod.params["did"], mod.pvalues["did"]
        ci = mod.conf_int().loc["did"]
        reg = {"desfecho": var, "especificacao": rot_e,
               "k": int(mod.df_model + 1), "gl": int(mod.df_resid),
               "coef_did": c, "p_valor": p, "significancia": sig_stars(p),
               "ic_95_inf": ci[0], "ic_95_sup": ci[1],
               "r2_ajustado": mod.rsquared_adj, "aic": mod.aic}
        for termo in ["t_agro", "t2_agro"]:
            reg[termo] = mod.params.get(termo, np.nan)
            reg[f"p_{termo}"] = mod.pvalues.get(termo, np.nan)
        linhas.append(reg)
        if rot_e.startswith("A3"):
            print(f"\n{rot} — {rot_e}")
            print(f"  N = {int(mod.nobs)}, gl = {int(mod.df_resid)}, "
                  f"R² aj. = {br(mod.rsquared_adj)}, AIC = {br(mod.aic, 2)}")
            tab = tabela_coef(mod, f"painel_quad_{var}", ROT_PAINEL)
            print(tab.to_string())
tab_quad = pd.DataFrame(linhas)
print("\nComparação entre tendência linear e quadrática:")
print(tab_quad.round(4).to_string(index=False))
exportar(tab_quad.round(4), "tab_tendencia_quadratica", indice=False)
print("\nCurvatura da trajetória do grupo tratado:")
for var, rot in DESFECHOS.items():
    r = tab_quad[(tab_quad.desfecho == var) &
                 (tab_quad.especificacao == "A3 tendência quadrática")].iloc[0]
    sig = "significativa" if r["p_t2_agro"] < 0.10 else "não significativa"
    print(f"  {rot:<28} t2_agro = {br(r['t2_agro'], 4):>8} "
          f"(p = {br(r['p_t2_agro'], 4)}) — {sig}")

titulo("11.2 ANOS FICTÍCIOS COM TENDÊNCIA QUADRÁTICA", "-")
pl2 = placebo.copy()
pl2["t2"] = pl2["t"] ** 2
pl2["t2_agro"] = pl2["t2"] * pl2["agro"]
linhas = []
for var, rot in DESFECHOS.items():
    for ano_f in ANOS_FICTICIOS:
        d = pl2.copy()
        d["did"] = d["agro"] * (d["ano"] >= ano_f).astype(int)
        if d["did"].nunique() < 2:
            continue
        mod = smf.ols(f"{var} ~ {FORM_A3}", data=d).fit(**HAC)
        c, p = mod.params["did"], mod.pvalues["did"]
        ci = mod.conf_int().loc["did"]
        linhas.append({"desfecho": var, "ano_ficticio": ano_f,
                       "N": int(mod.nobs), "gl": int(mod.df_resid),
                       "coef_did": c, "p_valor": p,
                       "significancia": sig_stars(p),
                       "ic_95_inf": ci[0], "ic_95_sup": ci[1]})
tab_pl2 = pd.DataFrame(linhas)
print(tab_pl2.round(4).to_string(index=False))
exportar(tab_pl2.round(4), "tab_placebo_quadratica", indice=False)
print("\nComparação da proporção de coeficientes fictícios significativos:")
for var, rot in DESFECHOS.items():
    a1 = tab_pl[(tab_pl.desfecho == var) &
                (tab_pl.especificacao == "A1 sem tendência")]
    a2 = tab_pl[(tab_pl.desfecho == var) &
                (tab_pl.especificacao == "A2 com tendência")]
    a3 = tab_pl2[tab_pl2.desfecho == var]
    print(f"  {rot}")
    print(f"    sem tendência        {int((a1['p_valor'] < 0.10).sum())} de {len(a1)}"
          f"   média {br(a1['coef_did'].mean())}")
    print(f"    tendência linear     {int((a2['p_valor'] < 0.10).sum())} de {len(a2)}"
          f"   média {br(a2['coef_did'].mean())}")
    print(f"    tendência quadrática {int((a3['p_valor'] < 0.10).sum())} de {len(a3)}"
          f"   média {br(a3['coef_did'].mean())}")

titulo("11.3 DIAGNÓSTICO DOS RESÍDUOS", "-")
MOD_DIAG = {}
for var in DESFECHOS:
    MOD_DIAG[(var, "A2")] = smf.ols(f"{var} ~ {FORM_A2}", data=pa_est).fit(**HAC)
    MOD_DIAG[(var, "A3")] = smf.ols(f"{var} ~ {FORM_A3}", data=pa_est).fit(**HAC)
print("Durbin-Watson calculado dentro de cada setor, evitando que a última")
print("observação de um setor seja tratada como defasagem do setor seguinte.\n")
linhas = []
for (var, esp), mod in MOD_DIAG.items():
    res = pd.Series(mod.resid.values, index=pa_est.index)
    reg = {"desfecho": var, "especificacao": esp,
           "dw_empilhado": durbin_watson(res.values)}
    for s in SETORES:
        idx = pa_est.index[pa_est["setor"] == s]
        reg[f"dw_{s}"] = durbin_watson(res.loc[idx].values)
    W, p = stats.shapiro(mod.resid)
    reg["shapiro_W"] = W
    reg["shapiro_p"] = p
    reg["normalidade"] = "não rejeitada" if p > 0.05 else "rejeitada"
    linhas.append(reg)
tab_diag = pd.DataFrame(linhas)
print(tab_diag.round(4).to_string(index=False))
exportar(tab_diag.round(4), "tab_diagnostico_residuos", indice=False)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for col, (var, rot) in enumerate(DESFECHOS.items()):
    mod = MOD_DIAG[(var, "A2")]
    res = mod.resid

    ax = axes[0, col]
    for s in SETORES:
        m = pa_est["setor"] == s
        ax.scatter(pa_est.loc[m, "ano"], res[m.values], color=CORES_SETOR[s],
                   s=45, label=s.capitalize(), alpha=0.85)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(ANO_LEI - 0.5, color=COR_LEI, linestyle="--", linewidth=1.3)
    rotular_painel(ax, "ABC"[col])
    ax.set_xlabel("Ano")
    ax.set_ylabel("Resíduo")
    if col == 0:
        ax.legend(fontsize=8)

    ax = axes[1, col]
    (osm, osr), (sl, inter, r) = stats.probplot(res, dist="norm")
    ax.scatter(osm, osr, color="#2C6FAC", alpha=0.85, s=35)
    lx = np.array([osm.min(), osm.max()])
    ax.plot(lx, sl * lx + inter, color=COR_LEI, linestyle="--", linewidth=1.3)
    ax.annotate(f"R = {r:.3f}", xy=(0.05, 0.90), xycoords="axes fraction",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="lightyellow"))
    rotular_painel(ax, "DEF"[col])
    ax.set_xlabel("Quantis teóricos")
    ax.set_ylabel("Quantis observados")

plt.tight_layout()
salvar_fig("fig_diagnostico_residuos")
print("\nFigura: fig_diagnostico_residuos")


# ==============================================================
# ETAPA 12 — CONTRAFACTUAIS
#
# Três construções, com hipóteses distintas e explícitas:
#   C1 — crescimento observado dos controles aplicado ao nível do
#        grupo tratado no último ano anterior à vigência.
#   C2 — tendência linear do próprio grupo tratado, ajustada ao
#        período anterior à vigência e prolongada.
#   C3 — idem, com termo quadrático.
# ==============================================================
titulo("ETAPA 12 — CONTRAFACTUAIS")
VARS_CF = {"rj_requeridas": "recuperações requeridas",
           "fal_requeridas": "falências requeridas"}
ANO_CORTE = ANO_LEI - 1
serie = pa.pivot_table(index="ano", columns="setor", values=list(VARS_CF))
anos_pre = [a for a in sorted(pa["ano"].unique()) if a <= ANO_CORTE]
anos_pos = [a for a in sorted(pa["ano"].unique()) if a > ANO_CORTE]
print(f"Período de ajuste : {anos_pre[0]}–{anos_pre[-1]} ({len(anos_pre)} anos)")
print(f"Período projetado : {anos_pos[0]}–{anos_pos[-1]} ({len(anos_pos)} anos)")
resultados = {}
for var, rot in VARS_CF.items():
    obs = {a: serie.loc[a, (var, SETOR_TRATADO)] for a in anos_pre + anos_pos}
    ln = {a: np.log(obs[a]) for a in obs}
    t_ = {a: a - anos_pre[0] for a in obs}
    c1 = {}
    for a in anos_pos:
        fatores = [np.log(serie.loc[a, (var, s)]) -
                   np.log(serie.loc[ANO_CORTE, (var, s)]) for s in CONTROLES]
        c1[a] = np.exp(ln[ANO_CORTE] + float(np.mean(fatores)))
    xs = np.array([t_[a] for a in anos_pre])
    ys = np.array([ln[a] for a in anos_pre])
    b1 = np.polyfit(xs, ys, 1)
    b2 = np.polyfit(xs, ys, 2)
    c2 = {a: float(np.exp(np.polyval(b1, t_[a]))) for a in anos_pos}
    c3 = {a: float(np.exp(np.polyval(b2, t_[a]))) for a in anos_pos}
    tab = pd.DataFrame({
        "ano": anos_pos,
        "observado": [obs[a] for a in anos_pos],
        "c1_controles": [c1[a] for a in anos_pos],
        "c2_tendencia_linear": [c2[a] for a in anos_pos],
        "c3_tendencia_quadratica": [c3[a] for a in anos_pos]})
    for c in ["c1_controles", "c2_tendencia_linear", "c3_tendencia_quadratica"]:
        tab[f"dif_{c}"] = tab["observado"] - tab[c]
    resultados[var] = tab
    print(f"\n{rot} — projeção anual:")
    print(tab.round(1).to_string(index=False))
    exportar(tab.round(2), f"tab_contrafactual_{var}", indice=False)
    print(f"\n{rot} — acumulado {anos_pos[0]}–{anos_pos[-1]}:")
    print(f"  observado                        {br(tab['observado'].sum(), 0):>10}")
    for c, rot_c in [("c1_controles", "crescimento dos controles"),
                     ("c2_tendencia_linear", "tendência linear própria"),
                     ("c3_tendencia_quadratica", "tendência quadrática própria")]:
        s = tab[c].sum()
        d = (tab["observado"].sum() / s - 1) * 100
        print(f"  {rot_c:<32} {br(s, 0):>10}   observado {br(d, 1):>7}%")

titulo("12.2 CONCLUSÃO IMPLICADA POR CADA CONTRAFACTUAL", "-")
linhas = []
for var, rot in VARS_CF.items():
    tab = resultados[var]
    for c, rot_c, hip in [
            ("c1_controles", "C1 crescimento dos controles",
             "trajetórias paralelas entre grupos"),
            ("c2_tendencia_linear", "C2 tendência linear própria",
             "continuidade da trajetória do grupo tratado"),
            ("c3_tendencia_quadratica", "C3 tendência quadrática própria",
             "continuidade com curvatura")]:
        linhas.append({
            "desfecho": var, "contrafactual": rot_c, "hipotese": hip,
            "observado_acum": tab["observado"].sum(),
            "projetado_acum": tab[c].sum(),
            "diferenca_pct": (tab["observado"].sum() / tab[c].sum() - 1) * 100,
            "anos_acima": int((tab["observado"] > tab[c]).sum()),
            "de_n_anos": len(tab)})
tab_sint = pd.DataFrame(linhas)
print(tab_sint.round(2).to_string(index=False))
exportar(tab_sint.round(4), "tab_contrafactual_sintese", indice=False)
print("\nO teste de trajetórias pré-tratamento rejeitou a hipótese exigida por")
print("C1. Os contrafactuais C2 e C3 não dependem dessa hipótese e apontam a")
print("mesma direção, o que os torna as referências preferíveis. Ainda assim,")
print("prolongar por cinco anos uma tendência ajustada a nove é procedimento")
print("agressivo, e a magnitude deve ser lida como ordem de grandeza.")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (var, rot) in zip(axes, VARS_CF.items()):
    hist = pa[(pa["setor"] == SETOR_TRATADO)].sort_values("ano")
    ax.plot(hist["ano"], hist[var], marker="o", color=COR_AGRO,
            linewidth=2.5, label="Observado", zorder=5)
    tab = resultados[var]
    for c, rot_c, cor, est in [
            ("c1_controles", "C1 controles", "#8E44AD", "--"),
            ("c2_tendencia_linear", "C2 tendência linear", "gray", ":"),
            ("c3_tendencia_quadratica", "C3 tendência quadrática",
             "#D68910", "-.")]:
        ax.plot(tab["ano"], tab[c], marker="s", markersize=5, color=cor,
                linestyle=est, linewidth=2, alpha=0.9, label=rot_c)
    ax.axvline(ANO_LEI - 0.5, color=COR_LEI, linestyle=":", linewidth=1.8)
    rotular_painel(ax, "AB"[list(axes).index(ax)])
    ax.set_xlabel("Ano")
    ax.set_ylabel("Número de CNPJs")
    ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
salvar_fig("fig_contrafactuais")
print("\nFigura: fig_contrafactuais")

# ==============================================================
# ETAPA 13 — PAINEL MENSAL E FUNIL PROCESSUAL
# ==============================================================
titulo("ETAPA 13 — PAINEL MENSAL E FUNIL PROCESSUAL")
ms = mensal.copy()
ms["log_rj_req"] = np.log(ms["rj_requeridas"].replace(0, np.nan))
ms["log_fal_req"] = np.log1p(ms["fal_requeridas"])
ms["total_insolv"] = ms["rj_requeridas"] + ms["fal_requeridas"]
ms["razao_rj"] = ms["rj_requeridas"] / ms["total_insolv"]
ms["mes_do_ano"] = ms["data"].dt.month
ms["t_mes"] = (ms["data"].dt.year - ms["data"].dt.year.min()) * 12 + \
              ms["data"].dt.month - 1
print(f"Cobertura: {ms['data'].min():%m/%Y} a {ms['data'].max():%m/%Y}, "
      f"{ms['data'].nunique()} meses, {ms['setor'].nunique()} setores")
print(f"Meses com zero falências requeridas: "
      f"{int((ms['fal_requeridas'] == 0).sum())} de {len(ms)}")

titulo("13.1 PANORAMA POR SETOR", "-")
ESTAGIOS = ["fal_requeridas", "fal_decretadas", "rj_requeridas",
            "rj_deferidas", "rj_concedidas"]
pan = ms.groupby("setor")[ESTAGIOS].agg(["sum", "mean", "min", "max"])
print(pan.round(1).to_string())
exportar(pan.round(2), "tab_mensal_panorama")
print("\nMédia mensal de recuperações requeridas por setor e ano:")
piv = ms.pivot_table(index="ano", columns="setor", values="rj_requeridas",
                     aggfunc="mean")
print(piv.round(1).to_string())
exportar(piv.round(2), "tab_mensal_media_rj")
print("\nComposição mensal média — recuperações sobre total de pedidos (%):")
piv_c = ms.pivot_table(index="ano", columns="setor", values="razao_rj",
                       aggfunc="mean") * 100
print(piv_c.round(1).to_string())
exportar(piv_c.round(2), "tab_mensal_composicao")

titulo("13.2 FUNIL PROCESSUAL", "-")
ms = ms.sort_values(["setor", "data"]).reset_index(drop=True)
for c in ESTAGIOS:
    ms[f"m12_{c}"] = ms.groupby("setor")[c].transform(
        lambda s: s.rolling(12, min_periods=12).sum())
ms["taxa_deferimento"] = ms["m12_rj_deferidas"] / ms["m12_rj_requeridas"]
ms["taxa_concessao"] = ms["m12_rj_concedidas"] / ms["m12_rj_deferidas"]
ms["taxa_decretacao"] = ms["m12_fal_decretadas"] / ms["m12_fal_requeridas"]
funil = ms.dropna(subset=["taxa_deferimento"]).groupby("setor").agg(
    meses=("taxa_deferimento", "count"),
    deferimento_medio=("taxa_deferimento", "mean"),
    deferimento_min=("taxa_deferimento", "min"),
    deferimento_max=("taxa_deferimento", "max"),
    concessao_media=("taxa_concessao", "mean"),
    decretacao_media=("taxa_decretacao", "mean"))
print("Taxas acumuladas em doze meses:")
print(funil.round(3).to_string())
exportar(funil.round(4), "tab_mensal_funil")
print("\nLeitura das taxas:")
for s in SETORES:
    if s not in funil.index:
        continue
    r = funil.loc[s]
    print(f"  {s:<10} de cada 100 recuperações requeridas, "
          f"{br(r['deferimento_medio']*100, 0)} são deferidas; "
          f"de cada 100 deferidas, {br(r['concessao_media']*100, 0)} concedidas")
print("\nDe cada 100 falências requeridas, quantas são decretadas:")
for s in SETORES:
    if s in funil.index:
        print(f"  {s:<10} {br(funil.loc[s, 'decretacao_media']*100, 0)}")

titulo("13.3 DETERMINANTES MACRO EM FREQUÊNCIA MENSAL", "-")
for k in [1, 3, 6]:
    for v in ["selic", "tx_cambio"]:
        ms[f"{v}_l{k}"] = ms.groupby("setor")[v].shift(k)
HAC_MES = {"cov_type": "HAC",
           "cov_kwds": {"maxlags": 6, "use_correction": USE_CORRECTION}}
linhas = []
for k in [1, 3, 6]:
    form = (f"C(setor) + selic_l{k} + tx_cambio_l{k} + t_mes")
    d = ms.dropna(subset=["log_rj_req", f"selic_l{k}", f"tx_cambio_l{k}"])
    mod = smf.ols(f"log_rj_req ~ {form}", data=d).fit(**HAC_MES)
    for v in [f"selic_l{k}", f"tx_cambio_l{k}"]:
        ci = mod.conf_int().loc[v]
        linhas.append({"defasagem_meses": k, "variavel": v.split("_l")[0],
                       "N": int(mod.nobs), "coeficiente": mod.params[v],
                       "p_valor": mod.pvalues[v],
                       "significancia": sig_stars(mod.pvalues[v]),
                       "ic_95_inf": ci[0], "ic_95_sup": ci[1],
                       "r2_ajustado": mod.rsquared_adj})
tab_mes = pd.DataFrame(linhas)
print(tab_mes.round(4).to_string(index=False))
exportar(tab_mes.round(4), "tab_mensal_determinantes", indice=False)

mod_saz = smf.ols("log_rj_req ~ C(setor) + C(mes_do_ano) + t_mes",
                  data=ms.dropna(subset=["log_rj_req"])).fit(**HAC_MES)
termos_saz = [t for t in mod_saz.params.index if t.startswith("C(mes_do_ano)")]
w = mod_saz.f_test(", ".join(f"{t} = 0" for t in termos_saz))
print(f"\nTeste conjunto de sazonalidade mensal: "
      f"estatística = {br(float(np.ravel(w.statistic)[0]), 3)}, "
      f"p = {br(float(w.pvalue), 4)}")
print(f"  {'há' if float(w.pvalue) < 0.10 else 'não há'} padrão sazonal "
      f"detectável nos pedidos de recuperação")

titulo("13.4 ANATOMIA DO PERÍODO DE ACELERAÇÃO", "-")
ag_m = ms[ms["setor"] == SETOR_TRATADO].sort_values("data")
print("Setor tratado — série mensal completa:")
print(ag_m[["ano", "mes", "rj_requeridas", "rj_deferidas", "rj_concedidas",
            "fal_requeridas", "fal_decretadas"]].to_string(index=False))
exportar(ag_m[["ano", "mes", "rj_requeridas", "rj_deferidas", "rj_concedidas",
               "fal_requeridas", "fal_decretadas"]],
         "tab_mensal_agro", indice=False)
print("\nMédia mensal de recuperações requeridas por semestre:")
ag_m2 = ag_m.copy()
ag_m2["semestre"] = ag_m2["ano"].astype(str) + "-S" + \
                    ((ag_m2["mes"] - 1) // 6 + 1).astype(str)
sem = ag_m2.groupby("semestre")["rj_requeridas"].agg(["mean", "sum", "count"])
print(sem.round(1).to_string())
exportar(sem.round(2), "tab_mensal_semestres")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
ax = axes[0, 0]
for s in SETORES:
    d = ms[ms["setor"] == s]
    ax.plot(d["data"], d["rj_requeridas"], marker="o", markersize=3.5,
            color=CORES_SETOR[s], label=s.capitalize(), linewidth=1.6)
rotular_painel(ax, "A")
ax.set_ylabel("Processos")
ax.legend(fontsize=8)

ax = axes[0, 1]
for s in SETORES:
    d = ms[ms["setor"] == s]
    ax.plot(d["data"], d["fal_requeridas"], marker="o", markersize=3.5,
            color=CORES_SETOR[s], label=s.capitalize(), linewidth=1.6)
rotular_painel(ax, "B")
ax.set_ylabel("Processos")
ax.legend(fontsize=8)

ax = axes[1, 0]
for s in SETORES:
    d = ms[ms["setor"] == s]
    ax.plot(d["data"], d["razao_rj"] * 100, marker="o", markersize=3.5,
            color=CORES_SETOR[s], label=s.capitalize(), linewidth=1.6)
ax.axhline(100, color="black", linewidth=0.8, linestyle=":")
rotular_painel(ax, "C")
ax.set_ylabel("%")
ax.legend(fontsize=8)

ax = axes[1, 1]
for s in SETORES:
    d = ms[ms["setor"] == s].dropna(subset=["taxa_deferimento"])
    ax.plot(d["data"], d["taxa_deferimento"] * 100, marker="o", markersize=3.5,
            color=CORES_SETOR[s], label=s.capitalize(), linewidth=1.6)
rotular_painel(ax, "D")
ax.set_ylabel("%")
ax.legend(fontsize=8)
plt.tight_layout()
salvar_fig("fig_mensal")
print("\nFigura: fig_mensal")

# ==============================================================
# ETAPA 14 — PESSOA FÍSICA
# ==============================================================
titulo("ETAPA 14 — PESSOA FÍSICA")
pf_a = pf.groupby("ano")["pedidos_pf"].agg(
    pf_total="sum", trimestres="count").reset_index()
cnpj_a = (pa[pa["setor"] == SETOR_TRATADO][["ano", "rj_requeridas"]]
          .rename(columns={"rj_requeridas": "cnpj_total"}))
comp = pf_a.merge(cnpj_a, on="ano", how="left")
comp["total_pf_mais_cnpj"] = comp["pf_total"] + comp["cnpj_total"]
comp["share_pf_pct"] = comp["pf_total"] / comp["total_pf_mais_cnpj"] * 100
comp["razao_pf_cnpj"] = comp["pf_total"] / comp["cnpj_total"]
print("Pessoa física e pessoa jurídica no setor tratado:")
print(comp.round(2).to_string(index=False))
exportar(comp.round(4), "tab_pf_vs_cnpj", indice=False)

titulo("14.1 RITMO DE CRESCIMENTO", "-")
base = comp.iloc[0]
comp["indice_pf"] = comp["pf_total"] / base["pf_total"] * 100
comp["indice_cnpj"] = comp["cnpj_total"] / base["cnpj_total"] * 100
print(f"Índice com base em {int(base['ano'])} = 100:")
print(comp[["ano", "indice_pf", "indice_cnpj"]].round(1).to_string(index=False))
ult = comp.iloc[-1]
print(f"\nDe {int(base['ano'])} a {int(ult['ano'])}:")
print(f"  pessoa física  : {br(base['pf_total'], 0)} → {br(ult['pf_total'], 0)}"
      f"   multiplicou por {br(ult['pf_total'] / base['pf_total'], 1)}")
print(f"  pessoa jurídica: {br(base['cnpj_total'], 0)} → "
      f"{br(ult['cnpj_total'], 0)}   multiplicou por "
      f"{br(ult['cnpj_total'] / base['cnpj_total'], 1)}")
for c, rot in [("pf_total", "pessoa física"), ("cnpj_total", "pessoa jurídica")]:
    b = np.polyfit(comp["ano"], np.log(comp[c].replace(0, np.nan).dropna()), 1)[0]
    print(f"  inclinação de {rot:<16} {br(b)} log por ano")

titulo("14.2 PARTICIPAÇÃO DA PESSOA FÍSICA", "-")
for _, r in comp.iterrows():
    print(f"  {int(r['ano'])}: pessoa física responde por "
          f"{br(r['share_pf_pct'], 1)}% do total de pedidos do setor "
          f"({br(r['pf_total'], 0)} de {br(r['total_pf_mais_cnpj'], 0)})")
print(f"\nImplicação para a variável dependente principal: ao considerar apenas")
print(f"pessoas jurídicas, a análise cobre {br(100 - comp['share_pf_pct'].iloc[-1], 1)}% "
      f"dos pedidos do setor no último ano da série.")

titulo("14.3 SÉRIE TRIMESTRAL", "-")
pf_t = pf.copy()
pf_t["rotulo"] = pf_t["trimestre"] + "/" + pf_t["ano"].astype(str)
pf_t["variacao_pct"] = pf_t["pedidos_pf"].pct_change() * 100
print(pf_t[["rotulo", "pedidos_pf", "variacao_pct"]].round(1).to_string(index=False))
exportar(pf_t[["ano", "trimestre", "pedidos_pf", "variacao_pct"]].round(2),
         "tab_pf_trimestral", indice=False)
print("\nMédia trimestral por ano:")
print(pf.groupby("ano")["pedidos_pf"].agg(["mean", "min", "max"]).round(1).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
ax = axes[0]
ax.bar(comp["ano"] - 0.2, comp["cnpj_total"], width=0.4,
       color=COR_AGRO, label="Pessoa jurídica", alpha=0.9)
ax.bar(comp["ano"] + 0.2, comp["pf_total"], width=0.4,
       color="#8E44AD", label="Pessoa física", alpha=0.9)
ax.set_xlabel("Ano")
ax.set_ylabel("Pedidos")
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(comp["ano"], comp["share_pf_pct"], marker="o", color="#8E44AD",
        linewidth=2.5)
ax.axhline(50, color="gray", linestyle=":", linewidth=1.2)
ax.set_xlabel("Ano")
ax.set_ylabel("%")

ax = axes[2]
ax.bar(range(len(pf_t)), pf_t["pedidos_pf"], color="#8E44AD", alpha=0.9)
ax.set_xticks(range(0, len(pf_t), 2))
ax.set_xticklabels(pf_t["rotulo"].iloc[::2], rotation=45, fontsize=8)
ax.set_ylabel("Pedidos")

rotular_painel(axes[0], "A")
rotular_painel(axes[1], "B")
rotular_painel(axes[2], "C")
plt.tight_layout()
salvar_fig("fig_pessoa_fisica")
print("\nFigura: fig_pessoa_fisica")

# ==============================================================
# ETAPA 15 — SÍNTESE E BLOCO DE TRANSCRIÇÃO
# ==============================================================
titulo("ETAPA 15 — SÍNTESE")


def val(expr, default="—"):
    """Avalia com segurança um resultado de etapa anterior."""
    try:
        r = eval(expr, globals())
        return r if r is not None else default
    except Exception:
        return default


titulo("15.1 QUADRO-RESUMO", "-")
resumo = []
for var, rot in DESFECHOS.items():
    r = {"desfecho": rot}
    r["tendencia_pre"] = val(f"float(tab_61.loc['{var}', 'diferencial_agro'])", np.nan)
    r["p_tendencia_pre"] = val(f"float(tab_61.loc['{var}', 'p_valor'])", np.nan)
    sub = val(f"tab_did[(tab_did.desfecho=='{var}') & "
              f"(tab_did.especificacao.str.startswith('A1')) & "
              f"(tab_did.erros=='HAC')]['coef_did'].iloc[0]", np.nan)
    r["did_sem_tendencia"] = sub
    sub2 = val(f"tab_did[(tab_did.desfecho=='{var}') & "
               f"(tab_did.especificacao.str.startswith('A2')) & "
               f"(tab_did.erros=='HAC')]['coef_did'].iloc[0]", np.nan)
    r["did_com_tendencia"] = sub2
    r["wald_pre_evento"] = val(f"float(tab_wald.loc['{var}', 'estatistica_pre'])", np.nan)
    r["placebos_significativos_sem_tend"] = val(
        f"int((tab_pl[(tab_pl.desfecho=='{var}') & "
        f"(tab_pl.especificacao=='A1 sem tendência')]['p_valor'] < 0.10).sum())", np.nan)
    r["placebos_significativos_com_tend"] = val(
        f"int((tab_pl[(tab_pl.desfecho=='{var}') & "
        f"(tab_pl.especificacao=='A2 com tendência')]['p_valor'] < 0.10).sum())", np.nan)
    resumo.append(r)
tab_resumo = pd.DataFrame(resumo).set_index("desfecho")
print(tab_resumo.round(4).to_string())
exportar(tab_resumo.round(4), "tab_sintese_geral")

titulo("BLOCO PARA TRANSCRIÇÃO NO TEXTO", "=")
print("\n[ AMOSTRA E DESENHO ]")
print(f"Painel anual: {len(pa_est)} observações, {pa_est['setor'].nunique()} "
      f"setores, {pa_est['ano'].min()}–{pa_est['ano'].max()}")
print(f"Observações tratadas: {int(pa_est['did'].sum())} "
      f"(setor {SETOR_TRATADO}, {ANO_LEI}–{int(pa_est['ano'].max())})")
print(f"Painel mensal: {len(mensal)} observações, "
      f"{mensal['data'].min():%m/%Y}–{mensal['data'].max():%m/%Y}")
print(f"Série de pessoa física: {len(pf)} trimestres, "
      f"{pf['ano'].min()}–{pf['ano'].max()}")

print("\n[ TRANSFORMAÇÃO INSTITUCIONAL — ETAPA 4 ]")
for s in SETORES:
    a = val(f"dec[(dec.setor=='{s}') & (dec.periodo=='{ANO_MIN}–{ANO_LEI-1}')].iloc[0]")
    b = val(f"dec[(dec.setor=='{s}') & (dec.periodo=='{ANO_LEI}–{ANO_MAX}')].iloc[0]")
    if isinstance(a, pd.Series):
        print(f"  {s:<10} parcela da queda de falências no deslocamento: "
              f"{br(a['share_falencia_pct'], 1)}% antes, "
              f"{br(b['share_falencia_pct'], 1)}% depois")

print("\n[ TRAJETÓRIAS PRÉ-TRATAMENTO — ETAPA 6 ]")
for var, rot in DESFECHOS.items():
    d = val(f"float(tab_61.loc['{var}', 'diferencial_agro'])", np.nan)
    p = val(f"float(tab_61.loc['{var}', 'p_valor'])", np.nan)
    li = val(f"float(tab_61.loc['{var}', 'ic_95_inf'])", np.nan)
    ls = val(f"float(tab_61.loc['{var}', 'ic_95_sup'])", np.nan)
    print(f"  {rot:<28} diferencial {br(d)} (IC {br(li)} a {br(ls)}; "
          f"p = {br(p, 4)})")
print(f"  comparações pareadas com divergência detectada: "
      f"{val('int((tab_63.p_valor < 0.10).sum())')} de {val('len(tab_63)')}")

print("\n[ ESTUDO DE EVENTO — ETAPA 7 ]")
for var, rot in DESFECHOS.items():
    e = val(f"float(tab_wald.loc['{var}', 'estatistica_pre'])", np.nan)
    p = val(f"float(tab_wald.loc['{var}', 'p_valor_pre'])", np.nan)
    print(f"  {rot:<28} teste conjunto pré-vigência: F = {br(e, 1)}, "
          f"p = {br(p, 4)}")

print("\n[ INVERSÃO DE SINAL — ETAPA 8 ]")
for var, rot in DESFECHOS.items():
    a1 = val(f"tab_did[(tab_did.desfecho=='{var}') & "
             f"(tab_did.especificacao.str.startswith('A1')) & "
             f"(tab_did.erros=='HAC')].iloc[0]")
    a2 = val(f"tab_did[(tab_did.desfecho=='{var}') & "
             f"(tab_did.especificacao.str.startswith('A2')) & "
             f"(tab_did.erros=='HAC')].iloc[0]")
    if isinstance(a1, pd.Series):
        print(f"  {rot}")
        print(f"    sem tendência: {br(a1['coef_did'])} "
              f"(p = {br(a1['p_valor'], 4)}) → {br(a1['efeito_pct'], 1)}%")
        print(f"    com tendência: {br(a2['coef_did'])} "
              f"(p = {br(a2['p_valor'], 4)}) → {br(a2['efeito_pct'], 1)}%")

print("\n[ COERÊNCIA DA TENDÊNCIA — ETAPA 8 ]")
print(f"  casos em que a tendência da amostra completa contém a do")
print(f"  período anterior no intervalo de 95%: "
      f"{val('int(tab_tend.dentro_do_ic.sum())')} de {val('len(tab_tend)')}")

print("\n[ DETERMINANTES MACRO — ETAPA 8 ]")
for var in DESFECHOS:
    for v in ["selic_lag1", "tx_cambio", "pib_lag1"]:
        r = val(f"tab_macro[(tab_macro.desfecho=='{var}') & "
                f"(tab_macro.especificacao=='A2') & "
                f"(tab_macro.variavel=='{v}')].iloc[0]")
        if isinstance(r, pd.Series):
            print(f"  {var:<10} {v:<12} {br(r['coeficiente'])} "
                  f"(p = {br(r['p_valor'], 4)}) {r['significancia']}")

print("\n[ SEPARABILIDADE — ETAPA 5 ]")
r2_tend_only = val('float(tab_sep.loc["tendência apenas", "r2"])', np.nan)
r2_todos = val('float(tab_sep.loc["todos os regressores", "r2"])', np.nan)
print(f"  variação do indicador reproduzida pela tendência, no setor tratado: "
      f"{br(r2_tend_only * 100, 1)}%")
print(f"  reproduzida por todos os regressores: {br(r2_todos * 100, 1)}%")
print(f"  especificações do setor tratado sem colinearidade severa: "
      f"{val('len(estimaveis)')} de {val('len(ESPEC_AGRO)')}")

print("\n[ DIAGNÓSTICO DOS RESÍDUOS — ETAPA 11 ]")
for var, rot in DESFECHOS.items():
    r = val(f"tab_diag[(tab_diag.desfecho=='{var}') & "
            f"(tab_diag.especificacao=='A2')].iloc[0]")
    if isinstance(r, pd.Series):
        dws = " | ".join(f"{s} {br(r[f'dw_{s}'], 2)}" for s in SETORES)
        print(f"  {rot}")
        print(f"    Durbin-Watson por setor: {dws}")
        print(f"    Shapiro-Wilk: W = {br(r['shapiro_W'], 3)}, "
              f"p = {br(r['shapiro_p'], 4)} — normalidade {r['normalidade']}")

print("\n[ ANOS FICTÍCIOS — ETAPA 10 ]")
for var, rot in DESFECHOS.items():
    n1 = val(f"int((tab_pl[(tab_pl.desfecho=='{var}') & "
             f"(tab_pl.especificacao=='A1 sem tendência')]['p_valor'] < 0.10).sum())")
    m1 = val(f"float(tab_pl[(tab_pl.desfecho=='{var}') & "
             f"(tab_pl.especificacao=='A1 sem tendência')]['coef_did'].mean())", np.nan)
    n2 = val(f"int((tab_pl[(tab_pl.desfecho=='{var}') & "
             f"(tab_pl.especificacao=='A2 com tendência')]['p_valor'] < 0.10).sum())")
    m2 = val(f"float(tab_pl[(tab_pl.desfecho=='{var}') & "
             f"(tab_pl.especificacao=='A2 com tendência')]['coef_did'].mean())", np.nan)
    print(f"  {rot:<28} sem tendência {n1} de 5 (média {br(m1)}) | "
          f"com tendência {n2} de 5 (média {br(m2)})")

print("\n[ CONTRAFACTUAIS — ETAPA 12 ]")
for _, r in pd.DataFrame(val("tab_sint", pd.DataFrame())).iterrows():
    print(f"  {r['desfecho']:<15} {r['contrafactual']:<32} "
          f"observado {br(r['diferenca_pct'], 1)}% "
          f"({int(r['anos_acima'])} de {int(r['de_n_anos'])} anos acima)")

print("\n[ FUNIL PROCESSUAL — ETAPA 13 ]")
for s in SETORES:
    r = val(f"funil.loc['{s}']")
    if isinstance(r, pd.Series):
        print(f"  {s:<10} deferimento {br(r['deferimento_medio']*100, 1)}% | "
              f"concessão {br(r['concessao_media']*100, 1)}% | "
              f"decretadas por requerida {br(r['decretacao_media'], 2)}")

print("\n[ PESSOA FÍSICA — ETAPA 14 ]")
for _, r in pd.DataFrame(val("comp", pd.DataFrame())).iterrows():
    print(f"  {int(r['ano'])}: pessoa física {br(r['pf_total'], 0):>6} | "
          f"jurídica {br(r['cnpj_total'], 0):>6} | "
          f"participação da física {br(r['share_pf_pct'], 1)}%")

titulo("15.3 ARQUIVOS GERADOS", "-")
csvs = sorted(DIR_SAIDA.glob("*.csv"))
pngs = sorted(DIR_SAIDA.glob("*.png"))
print(f"Tabelas: {len(csvs)}")
for f in csvs:
    print(f"  {f.name}")
print(f"\nFiguras: {len(pngs)}")
for f in pngs:
    print(f"  {f.name}")

titulo("FIGURAS NÃO GRAVADAS", "-")
if FIGURAS_NAO_GRAVADAS:
    for nome, erro, msg in FIGURAS_NAO_GRAVADAS:
        print(f"  {nome:<28} {erro}: {msg}")
else:
    print("Todas as figuras foram gravadas.")

titulo("ANÁLISE CONCLUÍDA", "=")
print(f"Saída em: {DIR_SAIDA}")
