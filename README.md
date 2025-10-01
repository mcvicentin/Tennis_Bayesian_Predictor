# Tenis data & analysis

> Pipeline de análise e predição de partidas de tênis (ATP/WTA) combinando histórico do Jeff Sackmann (baseline) com partidas de 2025 do Tennis-data.co.uk, e um modelinho bayesiano simples de “força recente” + *head-to-head* ponderado por recência.

---

## Sumário

- [Objetivo](#objetivo)  
- [Fontes de dados](#fontes-de-dados)  
- [Pré-processamento](#pré-processamento)  
- [Modelo (visão geral)](#modelo-visão-geral)  
- [Componentes do modelo](#componentes-do-modelo)  
  - [1) Baseline por ranking: \(p_{\text{gap}}\)](#1-baseline-por-ranking-p_textgap)  
  - [2) Forma recente (Beta–Binomial)](#2-forma-recente-beta–binomial)  
  - [3) Head-to-Head (H2H) com decaimento por recência](#3-head-to-head-h2h-com-decaimento-por-recência)  
  - [4) Combinação final](#4-combinação-final)  
- [Hiperparâmetros e escolhas de projeto](#hiperparâmetros-e-escolhas-de-projeto)  
- [Uso (script `run_predictor.py`)](#uso-script-run_predictorpy)  
- [Limitações conhecidas](#limitações-conhecidas)  
- [Roadmap / ideias de melhoria](#roadmap--ideias-de-melhoria)

---

## Objetivo

1. Estimar a probabilidade de vitória de um jogador **A** contra **B** em um confronto futuro usando:
   - um **baseline** empírico de “melhor ranqueado vence” em função do **gap** de ranking e do **nível do melhor rank**,  
   - a **forma recente** dos dois atletas (vitórias/derrotas recentes via Beta–Binomial),  
   - o retrospecto **head-to-head** com **peso maior para confrontos mais recentes**.

2. Fornecer saídas interpretáveis (fatores do cálculo + gráfico de forma), e permitir ajustar a influência do H2H.

---

## Fontes de dados

- **Jeff Sackmann** (`tennis_atp`, `tennis_wta`): histórico amplo (anos anteriores), consistente, com `winner_id/loser_id`, datas, rankings dos jogadores no dia do jogo.  
  - Usado para: **calibrar o baseline** \(p_{\text{gap}}\) e compor histórico para **forma** e **H2H**.
- **Tennis-data.co.uk (2025)**: partidas correntes (ATP/WTA) de 2025.  
  - Usado para: **atualizar** o histórico recente (forma/H2H) e **pegar o ranking do último jogo**.

**Observação:** dados especiais (Davis/Laver/United Cup, Olimpíadas, Next Gen, BJK Cup) são **excluídos** por dinâmica diferente de eventos padrão do tour.

---

## Pré-processamento

1. **Filtro temporal** padrão: `start_year = 2015` (ajustável).  
2. **Limpeza**: remoção de torneios especiais (citados acima).  
3. **Campos derivados**:
   - `gap = |winner_rank - loser_rank|`
   - `better_ranked_won = 1` se `winner_rank < loser_rank`, senão `0`
   - `best_rank = min(winner_rank, loser_rank)`
4. **Normalização de nomes** entre Jeff (nome completo) e Tennis-data (abreviado, ex.: “Nadal R.”):  
   - Heurística `{sobrenome, inicial} → "Nome Completo"` usando dicionário construído a partir do Jeff;  
   - **Mapa manual** para casos ambíguos/conhecidos (ex.:  
     `"Auger-Aliassime F." → "Felix Auger-Aliassime"`,  
     `"Mpetshi G." → "Giovanni Mpetshi Perricard"`,  
     `"Dedura D."/"Dedura-Palomero D." → "Daniel Dedura-Palomero"`,  
     `"Bu Y." → "Yu Bu"`).  
   - Nomes não resolvidos são listados em *log* para ajustes futuros.

5. **Ranking do último jogo (para predição):** para cada jogador consultado, usamos o **ranking que consta no último jogo encontrado** (se estiver faltando/zero, retrocedemos até achar um válido).

---

## Modelo (visão geral)

A probabilidade de A vencer é modelada via um logit que agrega três termos:

\[
\underbrace{\text{logit}\,P(A \text{ vence})}_{\text{logit}(p_A)}
\;=\;
\underbrace{s\cdot \text{logit}(p_{\text{gap}})}_{\text{baseline por ranking}}
\;+\;
\underbrace{\big[\Delta_A - \Delta_B\big]}_{\text{forma recente}}
\;+\;
\underbrace{\lambda\cdot \text{logit}(p_{\text{H2H}})}_{\text{head-to-head (se n}\ge 4)}
\]

- \(s=+1\) se **A** é melhor ranqueado (menor número), \(s=-1\) se **B** é melhor ranqueado.
- \(\Delta_X = \text{logit}(p_{\text{form},X})\) é a força recente do jogador \(X\) (vide abaixo).
- \(\lambda\) é o **peso** do fator H2H, ajustável pelo usuário.
- A saída final é \(p_A = \sigma\big(\text{logit}(p_A)\big)\), onde \(\sigma\) é a sigmoid.

---

## Componentes do modelo

### 1) Baseline por ranking: \(p_{\text{gap}}\)

Construímos uma **tabela empírica** a partir do Jeff:

- *Bins* de **gap** de ranking (diferença absoluta):  
  \([0,5], (5,10], (10,20], (20,50], (50,100], (100,200], (200,500], (500, \infty)\).
- *Bins* do **melhor ranking** em quadra:  
  \([1,10], [11,20], [21,50], [51,100], [101,200], [201,500], [501,1000], (1000,\infty)\).

Para cada célula (best\_bin, gap\_bin) calculamos a fração de vezes que o **melhor ranqueado venceu**, com **suavização de Laplace** (equivalente a prior Beta(2,2)):

\[
\hat{p}_{\text{gap}} \;=\; \frac{\text{wins} + 2}{\text{jogos} + 4}.
\]

Guardamos também \(\text{logit}(p_{\text{gap}})=\ln\frac{p_{\text{gap}}}{1-p_{\text{gap}}}\).

---

### 2) Forma recente (Beta–Binomial)

Para cada jogador \(X\), percorremos seu histórico **ordenado por data** e calculamos, em cada jogo, a **probabilidade posterior** de vitória considerando as **últimas \(N\)** partidas anteriores (janela deslizante):

- Prior: \(\text{Beta}(\alpha,\beta)\) — padrão \(\alpha=\beta=2\) (suave e simétrica).
- Se, antes do jogo \(t\), \(X\) tem \(n\) partidas na janela e \(w\) vitórias, então:
  \[
  p_{\text{form},X}(t) \;=\; \mathbb{E}[p\mid w,n] \;=\; 
  \frac{\alpha + w}{\alpha + \beta + n}.
  \]
- Para predição **futura**, usamos **o último ponto disponível** \(p_{\text{form},X}^{\text{(últ)}}\) e definimos:
  \[
  \Delta_X \;=\; \text{logit}\Big(\text{clip}\big(p_{\text{form},X}^{\text{(últ)}}, 10^{-4}, 1-10^{-4}\big)\Big).
  \]

---

### 3) Head-to-Head (H2H) com decaimento por recência

- Selecionamos **apenas jogos A×B** (qualquer ordem).  
- Se o total de confrontos \(n_{\text{H2H}} < 4\): **não aplicamos** o fator (informamos apenas “A x B = a–b”).  
- Caso contrário, ponderamos as partidas por **recência**:  
  - Ordene os jogos da **mais recente** para a **mais antiga**;  
  - Dê peso \(w_i = \gamma^{\,i}\) para o \(i\)-ésimo jogo (com \(i=0\) no mais recente);  
  - \(\gamma\in(0,1)\) — padrão **0,95** (memória longa, mas com decaimento).

Definindo \(y_i=1\) se **A venceu** o jogo \(i\), e \(0\) caso contrário, usamos um posterior Beta ponderado:

\[
p_{\text{H2H}} \;=\;
\frac{\alpha + \sum_i w_i\,y_i}{\alpha + \beta + \sum_i w_i}
\quad\text{e}\quad
\text{logit}_\text{H2H}=\ln\frac{p_{\text{H2H}}}{1-p_{\text{H2H}}}.
\]

> Hiperparâmetros padrão para H2H: \(\alpha=\beta=1\) (uniforme) e \(\gamma=0{,}95\).

---

### 4) Combinação final

Definimos o sinal do baseline de acordo com quem é **melhor ranqueado**:

\[
s \;=\; 
\begin{cases}
+1, & \text{se } \text{rank}_A < \text{rank}_B \\
-1, & \text{se } \text{rank}_B < \text{rank}_A
\end{cases}
\]

e combinamos:

\[
\text{logit}(p_A)
\;=\;
s\cdot \text{logit}(p_{\text{gap}})
\;+\;
\big[\Delta_A - \Delta_B\big]
\;+\;
\lambda\cdot \text{logit}(p_{\text{H2H}}),
\]

com \(\lambda\ge 0\) ajustável pelo usuário (ex.: 0 = ignora H2H; 1 = influência padrão).  
Por fim, \(p_A=\sigma(\text{logit}(p_A))\) e \(p_B=1-p_A\).

---

## Hiperparâmetros e escolhas de projeto

- **Filtros/Data**: `start_year=2015` (ajustável).  
- **Bins**:
  - `gap_bins = [0,5,10,20,50,100,200,500, ∞)`  
  - `best_bins = [0,10,20,50,100,200,500,1000, ∞)`  
- **Suavização do baseline**: \((\text{wins}+2)/(\text{count}+4)\) ≡ Beta(2,2).  
- **Forma recente**: janela **N=20** jogos (ajustável); prior Beta(2,2).  
- **H2H**: \(\gamma=0{,}95\), prior Beta(1,1), mínimo 4 jogos.  
- **Peso H2H**: \(\lambda\) informado pelo usuário a cada consulta.  
- **Normalização de nomes**: heurística `{sobrenome, inicial} → nome completo` + mapa manual.  
- **Ranking do último jogo**: usa o **último registro**; se faltante, **retrocede** até achar valor válido.

---

## Uso (script `run_predictor.py`)

### Requisitos

```bash
python >= 3.9
pip install pandas numpy matplotlib scipy requests openpyxl


