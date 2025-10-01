import os
import glob
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logit, expit as sigmoid
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# Normalização de nomes Tennis-data → Jeff
# =====================================================
import re
def normalize_tennisdata_names(df, jeff_names, verbose=True):
    manual_map = {
        "Auger-Aliassime F.": "Felix Auger-Aliassime",
        "Mpetshi G.": "Giovanni Mpetshi Perricard",
        "Dedura D.": "Daniel Dedura-Palomero",
        "Dedura-Palomero D.": "Daniel Dedura-Palomero",
        "Bu Y.": "Yu Bu",
    }

    mapping = {}
    for name in jeff_names:
        parts = name.split()
        if len(parts) < 2:
            continue
        surname = parts[-1].lower()
        initial = parts[0][0].lower()
        mapping[(surname, initial)] = name

    unresolved = set()

    def expand(name):
        if pd.isna(name):
            return name
        if name in manual_map:
            return manual_map[name]
        tokens = re.split(r"\s+", name.strip())
        if len(tokens) == 2:
            surname = tokens[0].lower()
            initial = tokens[1][0].lower()
        elif len(tokens) == 2 and "." in tokens[0]:
            surname = tokens[1].lower()
            initial = tokens[0][0].lower()
        else:
            return name
        full = mapping.get((surname, initial))
        if full is None:
            unresolved.add(name)
            return name
        return full

    for col in ["winner_name", "loser_name"]:
        df[col] = df[col].apply(expand)

    if verbose and unresolved:
        print("⚠️ Nomes não resolvidos:", sorted(unresolved))

    return df

# =====================================================
# Funções de dados
# =====================================================
def download_data(repo="atp"):
    url = f"https://github.com/JeffSackmann/tennis_{repo}/archive/refs/heads/master.zip"
    print(f"🔽 Baixando dataset {repo.upper()} do Jeff Sackmann...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(".")
    return f"./tennis_{repo}-master/"

def load_matches(path, repo="atp", start_year=2015):
    files = glob.glob(os.path.join(path, f"{repo}_matches_*.csv"))
    if not files:
        raise FileNotFoundError("CSV do Jeff não encontrado.")
    dfs = [pd.read_csv(f) for f in files]
    matches = pd.concat(dfs, ignore_index=True)

    matches['tourney_date'] = pd.to_datetime(matches['tourney_date'], format='%Y%m%d', errors='coerce')
    cols = ['tourney_name','tourney_date','surface','round',
            'winner_id','winner_name','winner_rank',
            'loser_id','loser_name','loser_rank']
    matches = matches[cols]

    ban = ['Davis Cup','Laver Cup','United Cup','Olympics','Atp Cup','Next Gen','Billie Jean King']
    mask = matches['tourney_name'].fillna('').str.contains('|'.join(ban), case=False, regex=True)
    m = matches.dropna(subset=['winner_rank','loser_rank'])
    m = m[~mask].copy()
    m = m[m['tourney_date'] >= f"{start_year}-01-01"]

    m['gap'] = (m['winner_rank'] - m['loser_rank']).abs()
    m['better_ranked_won'] = m['winner_rank'] < m['loser_rank']
    m['best_rank'] = m[['winner_rank','loser_rank']].min(axis=1)
    return m

def load_tennisdata_2025(repo="atp"):
    url = "http://tennis-data.co.uk/2025/2025.xlsx" if repo=="atp" else "http://tennis-data.co.uk/2025w/2025.xlsx"
    print(f"🔽 Baixando dataset 2025 {repo.upper()} do Tennis-data...")
    df = pd.read_excel(url)
    df = df.rename(columns={
        "Tournament": "tourney_name",
        "Date": "tourney_date",
        "Surface": "surface",
        "Round": "round",
        "Winner": "winner_name",
        "Loser": "loser_name",
        "WRank": "winner_rank",
        "LRank": "loser_rank"
    })
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
    cols = ["tourney_name","tourney_date","surface","round",
            "winner_name","winner_rank","loser_name","loser_rank"]
    return df[cols]

# =====================================================
# p_gap
# =====================================================
gap_bins   = [0,5,10,20,50,100,200,500, np.inf]
gap_labels = ["0-5","6-10","11-20","21-50","51-100","101-200","201-500","500+"]

best_bins   = [0,10,20,50,100,200,500,1000, np.inf]
best_labels = ["Top10","11-20","21-50","51-100","101-200","201-500","501-1000","1000+"]

def build_p_gap_table(m):
    m['gap_bin']  = pd.cut(m['gap'], bins=gap_bins, labels=gap_labels, right=True)
    m['best_bin'] = pd.cut(m['best_rank'], bins=best_bins, labels=best_labels, right=True)
    agg = m.groupby(['best_bin','gap_bin'])['better_ranked_won'].agg(['sum','count']).reset_index()
    agg['p_gap'] = (agg['sum'] + 2) / (agg['count'] + 4)
    return agg.pivot(index='best_bin', columns='gap_bin', values='p_gap')\
              .reindex(index=best_labels, columns=gap_labels)

def get_p_gap(rankA, rankB, p_gap_table):
    gap = abs(rankA - rankB)
    best = min(rankA, rankB)
    gap_lab  = pd.cut([gap], bins=gap_bins, labels=gap_labels)[0]
    best_lab = pd.cut([best], bins=best_bins, labels=best_labels)[0]
    p = p_gap_table.loc[best_lab, gap_lab]
    return float(np.clip(p, 1e-4, 1-1e-4)), float(np.log(p/(1-p)))

# =====================================================
# Forma recente
# =====================================================
def beta_posterior_mean(wins, n, alpha=2, beta=2):
    return (alpha + wins) / (alpha + beta + n)

def player_form_series(df_matches, player_name, N=20, alpha=2, beta=2):
    hist = df_matches[(df_matches['winner_name'].str.contains(player_name, case=False)) |
                      (df_matches['loser_name'].str.contains(player_name, case=False))]
    hist = hist.sort_values('tourney_date')
    if hist.empty:
        return pd.DataFrame([], columns=['date','form'])

    records = []
    for i, row in hist.iterrows():
        cutoff = row['tourney_date']
        sub = hist[hist['tourney_date'] < cutoff].tail(N)
        wins = (sub['winner_name'].str.contains(player_name, case=False)).sum()
        n = len(sub)
        p_post = beta_posterior_mean(wins, n, alpha, beta) if n>0 else beta_posterior_mean(0,0,alpha,beta)
        records.append((cutoff, p_post))
    return pd.DataFrame(records, columns=['date','form'])

def get_last_form(df_matches, player_name, N=20, alpha=2, beta=2):
    """Retorna o logit da forma mais recente do jogador."""
    series = player_form_series(df_matches, player_name, N=N, alpha=alpha, beta=beta)
    if series.empty:
        # prior
        p_post = beta_posterior_mean(0, 0, alpha, beta)
        return logit(np.clip(p_post, 1e-4, 1-1e-4))
    last_p = series['form'].iloc[-1]
    return logit(np.clip(last_p, 1e-4, 1-1e-4))

# =====================================================
# Head2Head
# =====================================================
def head_to_head(df_matches, playerA, playerB, gamma=0.95, alpha=1, beta=1, min_matches=4):
    """
    Estatística H2H com pesos por recência de jogo.
    Último confronto = peso 1, anteriores decaem por fator gamma.
    """
    mask = (
        ((df_matches['winner_name'].str.contains(playerA, case=False)) &
         (df_matches['loser_name'].str.contains(playerB, case=False)))
        |
        ((df_matches['winner_name'].str.contains(playerB, case=False)) &
         (df_matches['loser_name'].str.contains(playerA, case=False)))
    )
    h2h = df_matches[mask].copy()
    if h2h.empty:
        return {"msg": f"⚠️ Nenhum confronto entre {playerA} e {playerB}"}

    wins_A = ((h2h['winner_name'].str.contains(playerA, case=False))).sum()
    wins_B = ((h2h['winner_name'].str.contains(playerB, case=False))).sum()
    n_total = len(h2h)

    # sempre retorna retrospecto
    msg = f"Head-to-head: {n_total} jogos ({playerA} {wins_A} x {wins_B} {playerB})"

    if n_total < min_matches:
        return {"msg": msg}

    # Ordenar por data e atribuir pesos decrescentes
    h2h = h2h.sort_values("tourney_date", ascending=False).reset_index(drop=True)
    h2h["weight"] = [gamma**i for i in range(len(h2h))]

    h2h["A_win"] = h2h["winner_name"].str.contains(playerA, case=False).astype(int)

    weighted_wins = (h2h["A_win"] * h2h["weight"]).sum()
    weighted_total = h2h["weight"].sum()

    p_h2h = (alpha + weighted_wins) / (alpha + beta + weighted_total)

    return {
        "msg": msg,
        "n_total": n_total,
        "wins_A": wins_A,
        "wins_B": wins_B,
        "p_h2h": p_h2h,
        "logit_h2h": logit(np.clip(p_h2h, 1e-4, 1-1e-4))
    }




# =====================================================
# Predição
# =====================================================
def predict_future_match(df_matches, p_gap_table, playerA_name, playerB_name, 
                         match_date=None, lambda_h2h=1.0, gamma=0.95):
    if match_date is None:
        match_date = pd.Timestamp.today().normalize()

    # Último rank
    def get_last_info(name):
        hist = df_matches[(df_matches['winner_name'].str.contains(name, case=False)) |
                          (df_matches['loser_name'].str.contains(name, case=False))]
        hist = hist.sort_values('tourney_date')
        if hist.empty:
            raise ValueError(f"Sem histórico para {name}")
        last = hist.iloc[-1]
        if name.lower() in str(last['winner_name']).lower():
            return last['winner_id'], last['winner_rank']
        else:
            return last['loser_id'], last['loser_rank']

    idA, rankA = get_last_info(playerA_name)
    idB, rankB = get_last_info(playerB_name)

    # baseline p_gap
    p_gap, logit_gap = get_p_gap(rankA, rankB, p_gap_table)

    # forma (último ponto da série)
    deltaA = get_last_form(df_matches, playerA_name)
    deltaB = get_last_form(df_matches, playerB_name)


    # head-to-head
    h2h = head_to_head(df_matches, playerA_name, playerB_name, gamma=gamma, min_matches=4)
    if "logit_h2h" in h2h:
        logit_h2h = h2h["logit_h2h"]
    else:
        logit_h2h = 0.0  # sem influência se poucos jogos

    # logit final
    if rankA < rankB:
        logit_p = logit_gap + (deltaA - deltaB) + lambda_h2h * logit_h2h
        pA = sigmoid(logit_p); pB = 1 - pA
        favorite = playerA_name
    else:
        logit_p = logit_gap + (deltaB - deltaA) - lambda_h2h * logit_h2h
        pB = sigmoid(logit_p); pA = 1 - pB
        favorite = playerB_name

    print("\n🔎 Fatores do cálculo:")
    print(f" - Baseline p_gap: {p_gap:.3f} | logit_gap: {logit_gap:.3f}")
    print(f" - Forma {playerA_name}: {deltaA:.3f}")
    print(f" - Forma {playerB_name}: {deltaB:.3f}")
    if "msg" in h2h:
        print(f" - Head-to-head: {h2h['msg']}")
    print(f" - logit_h2h: {logit_h2h:.3f} (λ={lambda_h2h})")
    print(f" - Logit final: {logit_p:.3f}\n")

    print(f"{playerA_name} (rank {rankA}) - Prob vitória: {pA:.3f}")
    print(f"{playerB_name} (rank {rankB}) - Prob vitória: {pB:.3f}")
    print(f"➡ Favorito pelo modelo: {favorite}")

    # gráfico da forma
    formA = player_form_series(df_matches, playerA_name)
    formB = player_form_series(df_matches, playerB_name)

    plt.figure(figsize=(10,4))
    plt.plot(formA['date'], formA['form'], label=f"{playerA_name}", lw=2)
    plt.plot(formB['date'], formB['form'], label=f"{playerB_name}", lw=2)
    plt.axhline(0.5, color='gray', ls='--', lw=1)
    plt.title("Evolução da Forma Recente (probabilidade posterior)")
    plt.xlabel("Ano"); plt.ylabel("Forma")
    plt.legend(); plt.grid(alpha=0.3)
    plt.show()


# =====================================================
# Programa principal
# =====================================================
def main():
    repo_choice = input("Escolha o circuito [1] ATP (masculino) | [2] WTA (feminino): ").strip()
    repo = "atp" if repo_choice == "1" else "wta"

    # Jeff
    path = f"./tennis_{repo}-master/"
    if not os.path.exists(path):
        print("⚠️ Base do Jeff não encontrada localmente. Baixando...")
        path = download_data(repo=repo)
    jeff = load_matches(path, repo=repo)

    # Tennis-data 2025
    td2025 = load_tennisdata_2025(repo=repo)
    jeff_names = pd.concat([jeff['winner_name'], jeff['loser_name']]).unique()
    td2025 = normalize_tennisdata_names(td2025, jeff_names)

    # Dataset combinado
    matches = pd.concat([jeff, td2025], ignore_index=True).dropna(subset=["winner_rank","loser_rank"])

    # p_gap apenas com Jeff
    p_gap_table = build_p_gap_table(jeff)

    while True:
        playerA = input("\nDigite o nome do Jogador(a) A: ").strip()
        playerB = input("Digite o nome do Jogador(a) B: ").strip()
        try:
            lambda_h2h = float(input("Digite λ (head-to-head weight, ex: 0.0 para ignorar, 1.0 padrão): ").strip())
        except ValueError:
            lambda_h2h = 1.0
            print("⚠️ Valor inválido, usando λ=1.0")

        try:
            predict_future_match(matches, p_gap_table, playerA, playerB,
                                 lambda_h2h=lambda_h2h, gamma=0.95)
        except Exception as e:
            print("Erro:", e)

        again = input("\nDeseja avaliar outro jogo? (s/n): ").strip().lower()
        if again != 's':
            print("✅ Finalizado.")
            break


if __name__ == "__main__":
    main()
