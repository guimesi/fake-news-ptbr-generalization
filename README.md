# Detecção de fake news em português brasileiro: generalização, atalhos e robustez

Artefato de reprodutibilidade da qualificação de mestrado do autor e da
dissertação de mestrado em andamento sobre detecção de fake news em português
brasileiro com o dataset
[FakeRecogna 2.0](https://huggingface.co/datasets/recogna-nlp/fakerecogna2-abstrativa).
Parte dos resultados foi sistematizada em manuscrito submetido ao ENIAC 2026,
atualmente em avaliação. O repositório implementa, de ponta a ponta, o
protocolo experimental completo: carregamento e auditoria de dados,
pré-processamento, extração de features, treinamento de 17 configurações de
modelo, avaliação estatística robusta, explicabilidade (XAI), robustez a
perturbações, avaliação cross-dataset e, como eixo auxiliar, métricas de
custo computacional (latência e memória), com geração automática de tabelas,
figuras e relatório consolidado.

Este README é o guia principal para entender, instalar, executar e auditar o
experimento.

## Sumário

1. [Objetivo científico](#1-objetivo-científico)
2. [Relação com o mestrado e o artigo](#2-relação-com-o-mestrado-e-o-artigo)
3. [A tarefa de classificação](#3-a-tarefa-de-classificação)
4. [Dados utilizados](#4-dados-utilizados)
5. [Estrutura do repositório](#5-estrutura-do-repositório)
6. [Principais scripts e o notebook](#6-principais-scripts-e-o-notebook)
7. [Instalação do ambiente](#7-instalação-do-ambiente)
8. [Dependências principais](#8-dependências-principais)
9. [Ordem recomendada de execução](#9-ordem-recomendada-de-execução)
10. [Como reproduzir os experimentos](#10-como-reproduzir-os-experimentos)
11. [Onde encontrar os resultados e o que cada artefato significa](#11-onde-encontrar-os-resultados-e-o-que-cada-artefato-significa)
12. [Protocolo experimental (seeds, splits, hiperparâmetros)](#12-protocolo-experimental-seeds-splits-hiperparâmetros)
13. [Checklist de reprodutibilidade](#13-checklist-de-reprodutibilidade)
14. [Limitações e ameaças à validade](#14-limitações-e-ameaças-à-validade)
15. [Troubleshooting](#15-troubleshooting)
16. [Testes](#16-testes)
17. [Citação acadêmica](#17-citação-acadêmica)
18. [Licença e uso acadêmico](#18-licença-e-uso-acadêmico)
19. [Autoria e contato](#19-autoria-e-contato)

## 1. Objetivo científico

O objetivo é avaliar, de forma rigorosa e reprodutível, o desempenho e a
**generalização** de modelos de detecção de fake news em português brasileiro.
Mais do que reportar acurácia em um único split, o projeto investiga uma questão
central: até que ponto o alto desempenho dentro da distribuição (IID) reflete
aprendizado do fenômeno "desinformação" e não **atalhos** (shortcut learning),
ou seja, correlações espúrias com a fonte, o estilo ou o metatexto de
fact-checking presente nas notícias falsas.

Para isso, o experimento combina três frentes:

1. Medição cuidadosa do desempenho IID, com intervalos de confiança e testes de
   significância pareados, para 17 configurações de modelo.
2. Diagnóstico de robustez e de generalização: splits por fonte, temporais e
   anti-viés (categoria por período), avaliação cross-dataset com outro corpus,
   e perturbações adversariais.
3. Interpretabilidade: explicações locais (LIME, Integrated Gradients, Attention
   Rollout) cruzadas com a análise de erros, para verificar se os modelos se
   apoiam em pistas de conteúdo ou em marcas de origem e metatexto.

## 2. Relação com o mestrado e o artigo

Este código é o artefato de reprodutibilidade da qualificação de mestrado do
autor, concluída em agosto de 2026, e parte da dissertação de mestrado em
andamento. Parte dos resultados foi sistematizada em manuscrito submetido ao
ENIAC 2026, atualmente em avaliação, sobre lacunas de generalização e
aprendizado por atalho em detecção de fake news em português brasileiro. O
repositório foi organizado para que um avaliador, parecerista ou pesquisador
externo consiga reproduzir todos os experimentos relatados, inspecionar as
decisões metodológicas e auditar os artefatos gerados.

Nota: as métricas e figuras versionadas em `outputs/` correspondem à versão
dos experimentos utilizada na qualificação de mestrado e no manuscrito
submetido ao ENIAC 2026.

## 3. A tarefa de classificação

Classificação binária de notícias em **fake** ou **real**. O FakeRecogna 2.0
usa rótulos inteiros nativos (card oficial: 0 para notícias reais e 1 para
notícias falsas). A função `encode_labels` aplica `LabelEncoder` sobre os
rótulos convertidos para string, o que preserva esses inteiros. Portanto:

- `real` = classe 0
- `fake` = classe 1

A classe de interesse do problema (a que se deseja detectar) é a `fake`
(classe 1).

As métricas principais são reportadas em macro (precision, recall e F1 macro,
mais accuracy), de modo a não favorecer a classe majoritária.

## 4. Dados utilizados

**Dataset principal: FakeRecogna 2.0 (variante abstrativa).**

- Origem: Hugging Face Hub, id `recogna-nlp/fakerecogna2-abstrativa` (template
  `recogna-nlp/fakerecogna2-{variant}` em `configs/config.yaml`). Baixado
  automaticamente na primeira execução.
- A variante `abstrativa` é usada como dataset principal; a `extrativa` é
  carregada opcionalmente como controle (ablação C).
- Carregamento (`src/fakerecogna2/data/loading.py`): concatena todos os splits
  do Hub, normaliza o schema para as colunas canônicas (`text`, `title`,
  `category`, `author`, `date`, `url`, `label`), deriva `source` a partir do
  domínio da URL e faz o parsing de datas no formato brasileiro (dia primeiro).
- Deduplicação em duas etapas: exata (hash SHA-1 do texto normalizado) e
  near-duplicate via MinHashLSH (`datasketch`), com shingles de 5 caracteres,
  `num_perm=128` e limiar de Jaccard 0.85.

**Dataset cross-dataset: Fake.br-Corpus (opcional).**

- Usado apenas na avaliação cross-dataset. Fonte:
  [github.com/roneysco/Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus).
- Caminho local esperado: `data/external/Fake.br-Corpus-master`
  (chave `data.fakebr_local_path` em `configs/config.yaml`).
- Loader em `src/fakerecogna2/data/fakebr_loader.py`. Por padrão usa a versão
  `size_normalized_texts` (textos reais truncados ao tamanho da fake pareada,
  evitando viés de comprimento).

Os diretórios `data/raw`, `data/processed`, `data/interim` e `data/external` são
caches locais e estão no `.gitignore`. O repositório não distribui os dados; o
FakeRecogna 2.0 é baixado do Hub e o Fake.br-Corpus precisa ser obtido à parte.

## 5. Estrutura do repositório

```
.
├── README.md                       documentação principal (este arquivo)
├── pyproject.toml, requirements.txt, .gitignore
├── configs/config.yaml             parâmetros centralizados (seeds, splits, hparams, paths)
├── data/{raw,processed,interim}    caches locais de dados (gitignored)
│   └── external/                   datasets externos (Fake.br-Corpus)
├── docs/experiment_overview.md     visão geral textual do pipeline
├── notebooks/demo.ipynb            tour curto da API (cerca de 5 min, sem treino pesado)
├── outputs/                        artefatos de resultados versionados e saídas regeneráveis
│   ├── metrics/                    CSVs centrais usados na qualificação + novas métricas do pipeline
│   ├── figures/                    figuras centrais usadas na qualificação + novas figuras do pipeline
│   ├── logs/                       logs locais de execução (não versionados)
│   ├── models/                     modelos treinados (não versionados)
│   ├── explanations/              reservado para artefatos de XAI
│   └── .cache/                     results.json e results_multiseed.json (PersistentDict)
├── scripts/                        16 executáveis numerados + 2 helpers
│   ├── 00_validate_environment.py  valida imports, GPU e modelo spaCy
│   ├── 01_prepare_data.py          carregamento, integridade, preprocessing, splits
│   ├── 02_preprocess_text.py       EDA lexical + embeddings BERTimbau + TF-IDF
│   ├── 03_train_baselines.py       6 baselines clássicos
│   ├── 04_train_deep_models.py     CNN/LSTM/ConvLSTM, ensembles e BERTimbau FT
│   ├── 05_evaluate_models.py       bootstrap, McNemar, calibração, CV, NER masking
│   ├── 06_statistical_analysis.py  log-odds, Chi quadrado e McNemar
│   ├── 07_explainability.py        LIME, Integrated Gradients, Attention Rollout
│   ├── 08_cross_dataset.py         cross-dataset OOD (FakeRecogna para Fake.br)
│   ├── 09_adversarial_robustness.py  typos, deleção, swap, back-translation
│   ├── 10_plm_finetune.py          BERTimbau-large, XLM-R, mDeBERTa-v3
│   ├── 11_paraphrasing_equalizer.py  equalização de estilo e de metatexto
│   ├── 12_deployment_metrics.py    latência, VRAM, disco, fronteira de Pareto
│   ├── 13_ablations.py             ablações A, B, D, E e quartil curto
│   ├── 14_generate_report.py       relatório consolidado (Markdown + JSON)
│   ├── 15_xai_error_integration.py relatório standalone XAI versus erros
│   ├── run_all.py                  pipeline completo em um único processo
│   ├── _pipeline.py                helpers de setup (não executável)
│   └── _training.py                helpers de treino (não executável)
├── src/fakerecogna2/               pacote Python instalável
│   ├── __init__.py                 expõe ExperimentContext e __version__
│   ├── _context.py                 dataclass ExperimentContext (estado compartilhado)
│   ├── config.py                   constantes lidas de config.yaml (SEED, MAX_LEN, paths, ...)
│   ├── data/                       loading, integrity_checks, splits, anti_bias_splits,
│   │                               fakebr_loader, pipeline
│   ├── preprocessing/              text_cleaning, paraphrasing, equalizer, pipeline
│   ├── features/                   embeddings (BERTimbau), dataloaders, vectorization, streaming
│   ├── models/                     baselines, neural_models, training, ensembles,
│   │                               bertimbau_finetune, plm_finetune
│   ├── evaluation/                 metrics, bootstrap, calibration, cross_validation,
│   │                               stress_tests, error_analysis, cross_dataset,
│   │                               confusion_matrices, xai_error_integration, ablations, plots
│   ├── statistics/                 log_odds (prior de Dirichlet), significance_tests (McNemar + Holm)
│   ├── explainability/             lime_explainer, integrated_gradients, attention_rollout,
│   │                               comparison, stability
│   ├── adversarial/                perturbations, back_translation, runner
│   ├── deployment/                 benchmarks (latência, VRAM, disco, Pareto, auditoria de parâmetros)
│   ├── reports/                    consolidated (tabela final, relatório MD e JSON)
│   └── utils/                      seed, logging_utils, io_utils (PersistentDict), memory
└── tests/                          pytest (40 testes, cerca de 6 s, offline e sem GPU)
```

### Arquitetura em uma frase

O pacote `fakerecogna2` expõe funções e classes nomeadas; o estado dinâmico do
experimento (DataFrame, tokenizer, embeddings, modelos, predições, resultados)
circula por um objeto único, o `ExperimentContext` (`src/fakerecogna2/_context.py`).
Cada subpacote tem submódulos temáticos e, quando faz sentido, um `pipeline.py`
com `run(ctx, **flags)`. Os scripts em `scripts/` são invólucros finos sobre
esses pipelines, com `argparse` para as flags úteis.

```python
from fakerecogna2 import ExperimentContext
from fakerecogna2.data import load_and_prepare
from fakerecogna2.utils import set_global_seeds

ctx = ExperimentContext()
set_global_seeds(ctx.seed)
ctx.df, encoder, class_names = load_and_prepare("abstrativa")
# as etapas seguintes mutam ou enriquecem ctx
```

## 6. Principais scripts e o notebook

Os scripts numerados são auto-contidos: cada um refaz o setup de que precisa
(carregar dados, extrair embeddings, treinar modelos), o que é mais lento porém
consistente. Para reproduzir o experimento inteiro sem retrabalho, use o
`run_all.py`, que executa tudo em um único processo compartilhando o
`ExperimentContext`.

| Script | O que faz | Pré-requisitos | Principais saídas |
|---|---|---|---|
| `00_validate_environment.py` | Verifica imports, GPU e modelo spaCy | nenhum | apenas diagnóstico no terminal |
| `01_prepare_data.py` | Carrega, deduplica, audita integridade e gera splits | nenhum (baixa o dataset) | figuras `03_*` de integridade |
| `02_preprocess_text.py` | Preprocessing, embeddings BERTimbau, TF-IDF, análise lexical | nenhum | `04_logodds_*`, `04_chi2_top_abstrativa` |
| `03_train_baselines.py` | Treina os 6 baselines clássicos | nenhum | métricas em `RESULTS` (cache) |
| `04_train_deep_models.py` | CNN/LSTM/ConvLSTM, ensembles e BERTimbau FT | nenhum | métricas em `RESULTS` |
| `05_evaluate_models.py` | Bootstrap, McNemar, calibração, CV, NER masking, tabela final | nenhum | `13_*`, `14_ner_masking`, `16_final_results`, `16_final_comparison.png` |
| `06_statistical_analysis.py` | Log-odds de Dirichlet, Chi quadrado e McNemar | nenhum | `04_logodds_*`, `04_chi2_*`, `13_mcnemar_pairwise_holm` |
| `07_explainability.py` | LIME, Integrated Gradients, Attention Rollout, estabilidade | nenhum | `15_lime_*`, `I_lime_stability`, figuras em `figures/lime/` |
| `08_cross_dataset.py` | Cross-dataset OOD direto com salvaguarda de polaridade | Fake.br-Corpus local | `17_cross_dataset_ood`, `17_polarity_audit_log` |
| `09_adversarial_robustness.py` | Typos, deleção, swap e back-translation | nenhum (baixa MarianMT) | `19_adversarial_robustness` |
| `10_plm_finetune.py` | Fine-tuning de BERTimbau-large, XLM-R, mDeBERTa-v3 | nenhum (baixa vários GB) | `20_larger_models`, matrizes por PLM |
| `11_paraphrasing_equalizer.py` | Equalização de estilo (TextRank) e de metatexto | nenhum | `21_paraphrasing_equalizer` |
| `12_deployment_metrics.py` | Latência, VRAM, disco e fronteira de Pareto | nenhum (ver aviso abaixo) | `22_deployment_metrics`, `22_pareto_f1_vs_latency.png` |
| `13_ablations.py` | Ablações A, B, D, E e quartil curto | nenhum | `preprocessing_ablation`, `seqlen_ablation`, `learning_curve`, etc. |
| `14_generate_report.py` | Relatório consolidado lendo os CSVs já gerados | rodar etapas anteriores | `outputs/relatorio_final.md`, `outputs/resultados.json` |
| `15_xai_error_integration.py` | Relatório standalone cruzando XAI e erros | CSVs de XAI e de erros já gerados | `xai_err_consolidated_standalone.md` |
| `run_all.py` | Pipeline completo em um processo | nenhum | todas as saídas acima |

Diferenças importantes entre os scripts isolados e o `run_all.py` estão
documentadas nas docstrings de cada script. As mais relevantes:

- `05_evaluate_models.py` roda apenas o stress test de NER masking; os splits por
  fonte, temporal e anti-viés rodam somente no `run_all.py` (etapas 8b a 8d).
- `08_cross_dataset.py` faz apenas a direção direta (FakeRecogna para Fake.br); a
  direção inversa fica no `run_all.py` (etapa 10b).
- `10_plm_finetune.py` usa uma seed quando rodado isolado; o `run_all.py` roda
  multi-seed via `--trans-seeds`.
- `12_deployment_metrics.py`, rodado isolado, não treina o baseline TF-IDF + MLP,
  então o ponto TF-IDF + MLP no Pareto cai em um fallback aleatório (com aviso no
  log). Para o resultado correto, use o `run_all.py`.

**Notebook.** [`notebooks/demo.ipynb`](notebooks/demo.ipynb) oferece um tour de
cerca de 5 minutos pela API, sem treinar modelos pesados. É opcional e não faz
parte do fluxo de reprodução.

## 7. Instalação do ambiente

Requer Python 3.10 ou superior. GPU NVIDIA é recomendada (ver
[Recursos pesados](#recursos-pesados)).

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m spacy download pt_core_news_sm
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download pt_core_news_sm
```

O `pip install -e .` instala o pacote `fakerecogna2` em modo editável (a partir
de `src/`) e todas as dependências declaradas em `pyproject.toml`. O modelo
spaCy `pt_core_news_sm` precisa ser baixado à parte (não é uma dependência pip
comum). Ferramentas de desenvolvimento (pytest e ruff) podem ser instaladas
com `pip install -e .[dev]`. Em seguida, valide o ambiente:

```powershell
python scripts/00_validate_environment.py
```

## 8. Dependências principais

Declaradas em `pyproject.toml` e espelhadas em `requirements.txt`:

- Numérico e científico: `numpy>=1.24,<2.0`, `pandas>=2.0`, `scipy>=1.10`,
  `scikit-learn>=1.3`, `statsmodels>=0.14`, `PyYAML>=6.0`
- Deep learning: `torch>=2.0`, `transformers>=4.35`
- NLP e texto: `nltk>=3.8`, `spacy>=3.6` (mais o modelo `pt_core_news_sm`),
  `datasets>=2.14`, `datasketch>=1.6` (deduplicação MinHashLSH),
  `sumy>=0.11` (TextRank), `sentencepiece>=0.1.99` e `protobuf>=4.21`
  (tokenizadores de XLM-R e mDeBERTa)
- Explicabilidade: `lime>=0.2.0.1`, `captum>=0.7`
- Visualização e IO: `matplotlib>=3.7`, `seaborn>=0.12`, `tabulate>=0.9`,
  `tqdm>=4.65`, `openpyxl>=3.1`
- Desenvolvimento (opcional, extra `dev`): `pytest>=7.4`, `ruff>=0.1`

<a name="recursos-pesados"></a>
**Recursos pesados.** Uma GPU com 8 GB ou mais de VRAM (linha RTX 30/40/50) é
recomendada. Sem GPU o pipeline completo ainda roda, mas leva muitas horas. Na
primeira execução, os modelos são baixados do Hugging Face e ficam em cache
(BERTimbau-base cerca de 440 MB, BERTimbau-large cerca de 1.3 GB, XLM-R cerca de
1.1 GB, mDeBERTa cerca de 700 MB, MarianMT cerca de 600 MB por direção), algo em
torno de 5 GB no total.

## 9. Ordem recomendada de execução

Há dois caminhos. O recomendado para reproduzir o artigo é o caminho único.

**Caminho único (recomendado).**

```powershell
python scripts/00_validate_environment.py
python scripts/run_all.py
```

O `run_all.py` executa todas as etapas na ordem correta, compartilhando o
contexto, e ao final escreve o relatório consolidado.

**Caminho granular (etapas isoladas).** Útil para depurar uma etapa específica.
Como cada script refaz o setup, a ordem entre `01` e `13` é em grande parte
independente; a única dependência real é que `14` e `15` leem CSVs produzidos
pelas etapas anteriores. Sequência sugerida:

```
00 (validar) -> 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15
```

Atenção: rodar `14_generate_report.py` ou `15_xai_error_integration.py` sem ter
gerado os CSVs de entrada produz relatórios parciais (campos ausentes são
marcados como tal, sem erro). Para um relatório completo, prefira o `run_all.py`.

## 10. Como reproduzir os experimentos

```powershell
# Tudo (pode levar de 3 a 4 horas em GPU)
python scripts/run_all.py

# Variações úteis
python scripts/run_all.py --quick          # pula cross-dataset, PLM, adversarial, ablações e paraphrasing
python scripts/run_all.py --skip-bt        # pula back-translation (lento)
python scripts/run_all.py --skip-xai       # pula LIME, IG e Attention Rollout
python scripts/run_all.py --skip-cv        # pula a validação cruzada de 5 folds (cara)
python scripts/run_all.py --trans-seeds 42 # usa 1 seed nos Transformers (mais rápido)
```

Flags do `run_all.py`: `--quick`, `--skip-plm`, `--skip-bt`, `--skip-cross`,
`--skip-adv`, `--skip-ablations`, `--skip-xai`, `--skip-paraphr`, `--skip-cv`,
`--epochs`, `--trans-seeds`.

Para gerar uma etapa específica (métricas, matrizes de confusão, calibração,
XAI, etc.), use o script numerado correspondente; todos aceitam `--help`:

```powershell
python scripts/05_evaluate_models.py --help
python scripts/04_train_deep_models.py --epochs 5 --no-bert   # treino rápido para depuração
python scripts/15_xai_error_integration.py                    # relatório XAI versus erros a partir dos CSVs
```

Para um recomeço limpo do cache de resultados, apague `outputs/.cache/results.json`
e `outputs/.cache/results_multiseed.json` ou defina a variável de ambiente
`FAKERECOGNA_RESULTS_FRESH=1` antes de rodar.

## 11. Onde encontrar os resultados e o que cada artefato significa

Os artefatos são roteados por três funções utilitárias
(`src/fakerecogna2/utils/io_utils.py`): `save_table` grava CSV em
`outputs/metrics/`, `save_plot` grava PNG em `outputs/figures/`, e `save_json`
grava JSON em `outputs/metrics/`. As métricas centrais da qualificação
(`outputs/metrics/`, 110 CSVs) e as figuras (`outputs/figures/`) estão
versionadas neste repositório. Modelos treinados, logs pesados, caches e
dados brutos continuam fora do Git; novos outputs são regenerados ao rodar
o pipeline.

Tabelas e relatórios principais (`outputs/metrics/`):

| Artefato | Conteúdo |
|---|---|
| `16_final_results.csv` | Tabela final com Accuracy, Precision, Recall, F1 macro e latência das 17 configurações |
| `13_bootstrap_ci.csv` | Intervalos de confiança (percentil 2.5 a 97.5) de F1 e Accuracy por modelo |
| `13_mcnemar_pairwise_holm.csv` | Teste de McNemar pareado entre modelos, com correção de Holm |
| `13_calibration.csv` | ECE e Brier por modelo |
| `13_cross_validation.csv` | Acurácia e F1 por fold (5-fold estratificado) |
| `14_ner_masking.csv` | Queda de F1 ao mascarar entidades nomeadas (stress test) |
| `14_source_split.csv`, `14_temporal_split.csv`, `14_anti_bias_split.csv` | Desempenho nos splits OOD por fonte, temporal e anti-viés |
| `cm_<contexto>_<modelo>_cm.csv` e `_per_class.csv` | Matrizes de confusão e métricas por classe (IID, fonte, temporal, anti-viés, OOD) |
| `error_distrib_<modelo>_by_{fonte,categoria,year,confidence}.csv` | Distribuição de erros estratificada |
| `17_cross_dataset_ood.csv`, `17_cross_dataset_delta.csv`, `17_cross_dataset_inverse_ood.csv` | Cross-dataset direto, delta e inverso |
| `17_polarity_audit_log.csv` | Auditoria da salvaguarda de polaridade no cross-dataset |
| `19_adversarial_robustness.csv` | F1 sob typos, deleção, swap e back-translation |
| `20_larger_models.csv`, `plms_multiseed.csv` | PLMs maiores (BERTimbau-large, XLM-R, mDeBERTa) |
| `22_deployment_metrics.csv`, `J_disk_sizes_corrected.csv`, `J_parameters_audit.csv` | Latência, VRAM, disco e contagem de parâmetros |
| `21_paraphrasing_equalizer.csv` | Efeito da equalização de estilo e de metatexto |
| `abstrativa_vs_extrativa.csv` | Comparação abstrativa versus extrativa (ablação C) |
| `04_logodds_*`, `04_chi2_top_abstrativa.csv` | Análise lexical por classe (log-odds e Chi quadrado) |
| `baselines_multiseed.csv`, `bertimbau_ft_multiseed.csv`, `iid_multiseed.csv` | Agregados multi-seed (média e desvio) |
| `xai_err_*` | Integração entre XAI e análise de erros |

Figuras (`outputs/figures/`): heatmaps de matrizes de confusão (`*_cm.png`),
`13_reliability.png`, `16_final_comparison.png`, `22_pareto_f1_vs_latency.png`,
`D_learning_curve.png`, `E_performance_by_length.png`, `03_length_per_class_*.png`,
`03_temporal_per_class.png`, e as explicações em `outputs/figures/lime/`
(LIME, IG e Attention Rollout por TP/TN/FP/FN).

Relatórios consolidados:

- `outputs/relatorio_final.md`: narrativa em Markdown reunindo os principais
  resultados.
- `outputs/resultados.json`: os mesmos números em formato estruturado.
- `outputs/manifest.txt`: manifesto da execução.

Outros:

- `outputs/models/`: vazio por padrão. Os modelos não são persistidos entre
  execuções; eles vivem em memória durante o `run_all.py`.
- `outputs/.cache/results.json` e `results_multiseed.json`: dicionários
  auto-persistentes (`PersistentDict`) com as métricas agregadas. São lidos no
  import e regravados a cada escrita.
- `outputs/logs/exp_<timestamp>.log`: log estruturado de cada execução.

## 12. Protocolo experimental (seeds, splits, hiperparâmetros)

Todos os parâmetros estão em [`configs/config.yaml`](configs/config.yaml) e são
expostos como constantes por [`src/fakerecogna2/config.py`](src/fakerecogna2/config.py).
O YAML é a fonte de verdade; os valores `default=` em `config.py` são apenas
fallbacks (mantidos idênticos ao YAML) usados caso o arquivo esteja ausente.
**Edite o YAML, não o código.**

Parâmetros críticos (não alterar a menos que esteja conduzindo um experimento
novo):

- Determinismo: `SEED=42`; multi-seed `SEEDS_MULTI=[42, 7, 2024]` (intervalos de
  confiança dos modelos neurais e baselines). O fine-tuning dos Transformers usa
  por padrão as sementes `[42, 7]` (`--trans-seeds`).
- Splits: estratificado 70/10/20 (`test_size=0.20`, `val_size=0.10`), feito
  sobre a coluna pré-processada `text_proc` com estratificação por rótulo. Esse
  é o split usado no protocolo principal. Os splits temporal, por fonte
  (GroupShuffleSplit, sem sobreposição de fonte entre treino e teste) e
  anti-viés (estratificado por rótulo, categoria e período) são análises
  adicionais de robustez.
- Tamanho de batch: as redes neurais profundas (CNN/LSTM/ConvLSTM) usam
  `BATCH_SIZE=32` (lido do config). O BERTimbau fine-tuned e os PLMs maiores
  usam batch 16, que é o default interno de `train_bert_classifier`,
  `train_bertimbau_finetune` e `plm_finetune`.
- BERTimbau fine-tuned: `neuralmind/bert-base-portuguese-cased`, `MAX_LEN=200`,
  batch 16, `LEARNING_RATE=2e-5`, `NUM_EPOCHS=10`, `PATIENCE=3`.
- PLMs maiores (mesmos hiperparâmetros de fine-tuning, batch 16):
  `neuralmind/bert-large-portuguese-cased`, `xlm-roberta-base`,
  `microsoft/mdeberta-v3-base`.
- Avaliação: bootstrap `BOOTSTRAP_ITERS=10000`, ECE `CALIBRATION_BINS=15`,
  reliability `RELIABILITY_BINS=10`, validação cruzada `CV_FOLDS=5`.
- Adversarial: `typo_rate=0.05`, `deletion_rate=0.10`, `swap_rate=0.05`,
  amostra de perturbação 500 e amostra de back-translation 300; modelos MarianMT
  `Helsinki-NLP/opus-mt-roa-en` e `Helsinki-NLP/opus-mt-en-roa`.

As 17 configurações de modelo:

- 6 baselines: `TFIDF+LogReg`, `TFIDF+SVM`, `TFIDF+MLP`, `BERT[CLS]+LogReg`,
  `BERT[CLS]+SVM`, `BERT[CLS]+MLP`
- 3 redes neurais sobre embeddings token-level do BERTimbau: `CNN`, `LSTM`,
  `ConvLSTM`
- 4 ensembles: `Ens2 (CNN+LSTM)`, `Ens3 (CNN+LSTM+ConvLSTM)`, `WEns2` e `WEns3`
  (pesos por grid search na validação)
- 1 BERTimbau fine-tuned: `BERTimbau FT`
- 3 PLMs maiores: `PLM: bert-large-portuguese-cased`, `PLM: xlm-roberta-base`,
  `PLM: mdeberta-v3-base`

## 13. Checklist de reprodutibilidade

- [x] Seeds fixas e centralizadas (`SEED=42`, `SEEDS_MULTI`), aplicadas via
  `set_global_seeds` (numpy, random, torch, cudnn determinístico, PYTHONHASHSEED).
- [x] Hiperparâmetros e caminhos centralizados em `configs/config.yaml`.
- [x] Splits determinísticos e estratificados, com semente fixa.
- [x] Protocolo de avaliação explícito (bootstrap, McNemar com correção de Holm,
  calibração, validação cruzada).
- [x] Multi-seed com média e desvio para os modelos sensíveis a inicialização.
- [x] Versões de dependências fixadas em `pyproject.toml` e `requirements.txt`.
- [x] Pipeline de ponta a ponta em um comando (`run_all.py`).
- [x] Geração automática de tabelas, figuras e relatório consolidado.
- [x] Testes automatizados (offline, sem GPU) cobrindo importação, config, API
  de dados, métricas e persistência.
- [ ] Pesos dos modelos treinados: não são versionados nem persistidos entre
  execuções (reproduzíveis ao rodar o pipeline).
- [ ] Dados: não são versionados (FakeRecogna 2.0 vem do Hugging Face; o
  Fake.br-Corpus precisa ser baixado à parte).
- [ ] Determinismo absoluto em GPU: pequenas variações numéricas entre hardwares
  e versões de CUDA/cuDNN são possíveis mesmo com seeds fixas.

## 14. Limitações e ameaças à validade

- **Generalização frágil.** O foco do estudo é justamente mostrar que o alto
  desempenho IID não se traduz necessariamente em generalização. Resultados
  cross-dataset e nos splits OOD (fonte, temporal, anti-viés) tendem a ser
  bem mais baixos que os IID; leia esses números em conjunto.
- **Aprendizado por atalho.** Há indícios de que parte do sinal vem da fonte e
  de metatexto de fact-checking, não do conteúdo. O equalizador linguístico e a
  remoção de metatexto (etapa 11), os splits anti-viés e a análise XAI versus
  erros existem para tornar esse efeito mensurável.
- **Determinismo em GPU.** Mesmo com seeds fixas, operações em GPU podem variar
  ligeiramente entre versões de hardware, CUDA e cuDNN.
- **Salvaguarda de polaridade no cross-dataset.** O auto-flip de rótulos quando
  a acurácia invertida supera a direta por mais de 0.05 é registrado em log de
  auditoria (`17_polarity_audit_log.csv`); o limiar é uma escolha de projeto e
  deve ser considerado ao interpretar os resultados inversos.
- **Recursos manuais.** A lista de metatexto de fact-checking (equalizador) e o
  mapa de teclado das perturbações de typo (layout QWERTY) são heurísticas
  construídas manualmente.
- **Custo computacional.** A reprodução completa exige GPU e várias horas; as
  flags `--quick` e `--skip-*` permitem subconjuntos.

## 15. Troubleshooting

- **`ModuleNotFoundError: fakerecogna2`.** Rode `pip install -e .` na raiz do
  projeto. Os testes de persistência (que abrem subprocessos) só passam com o
  pacote instalado; alternativamente, defina `PYTHONPATH` apontando para `src/`.
- **`spaCy pt_core_news_sm AUSENTE`.** Rode
  `python -m spacy download pt_core_news_sm`.
- **Caracteres quebrados no terminal Windows.** O pacote reconfigura a saída
  para UTF-8 ao importar. Para reforçar, defina
  `$env:PYTHONIOENCODING = "utf-8"` antes de rodar.
- **Out of memory em cross-dataset ou adversarial.** Use as fábricas de predição
  em lote `make_predict_ens3(ctx, batch_size=32)` e
  `make_predict_bert_ft(ctx, batch_size=32)` (`scripts/_training.py`), que já
  fazem o batching interno.
- **mDeBERTa quebra com erro de dtype.** O checkpoint vem em FP16 e o head é
  criado em FP32; o fix (carregar com `torch_dtype=torch.float32`) já está em
  `src/fakerecogna2/models/plm_finetune.py`.
- **Cross-dataset é pulado.** O Fake.br-Corpus não foi encontrado em
  `data.fakebr_local_path`. Baixe e extraia o corpus para
  `data/external/Fake.br-Corpus-master` ou ajuste o caminho no YAML.
- **Primeira execução muito lenta.** São baixados cerca de 5 GB de modelos do
  Hugging Face; execuções seguintes usam o cache.
- **Resultados antigos persistem.** O cache `outputs/.cache/results.json`
  preserva valores entre execuções. Para um recomeço limpo, apague-o ou use
  `FAKERECOGNA_RESULTS_FRESH=1`.
- **Ponto TF-IDF + MLP sem sentido no Pareto.** Acontece ao rodar
  `12_deployment_metrics.py` isolado, que não treina o baseline MLP (um aviso é
  emitido no log). Use o `run_all.py` para o resultado correto.

## 16. Testes

```powershell
pytest tests/ -v
```

São 40 testes que rodam offline, sem GPU, em cerca de 6 segundos. Cobrem:
importabilidade dos subpacotes, carregamento de `config.py`, instanciação do
`ExperimentContext`, funções da API de `data/` com DataFrames sintéticos,
métricas básicas (`standard_report`) e a persistência do `PersistentDict`. Os
dois testes que abrem subprocessos exigem o pacote instalado (`pip install -e .`)
ou `PYTHONPATH` apontando para `src/`.

## 17. Citação acadêmica

Enquanto o manuscrito submetido ao ENIAC 2026 estiver em avaliação, cite este
repositório (ver `CITATION.cff`) e entre em contato com o autor para obter a
referência atualizada do manuscrito. Não utilize dados de publicação
provisórios: autores, título, veículo e ano definitivos serão divulgados após
a decisão editorial.

## 18. Licença e uso acadêmico

Distribuído sob a licença MIT (ver `LICENSE`). O uso para fins
acadêmicos e de pesquisa é livre; ao reutilizar, mantenha a atribuição e cite o
trabalho associado. Os datasets têm licenças próprias: consulte os termos do
FakeRecogna 2.0 (Hugging Face) e do Fake.br-Corpus em suas respectivas fontes.

## 19. Autoria e contato

Autor: Guilherme de Oliveira Silva (`guilherme.oliveirasilva@gmail.com`).
