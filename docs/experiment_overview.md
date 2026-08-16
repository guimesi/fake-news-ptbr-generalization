# Visão Geral do Experimento — FakeRecogna 2.0

Este documento resume, em alto nível, o que cada bloco do experimento faz e
onde ele vive no projeto.

## Pergunta de pesquisa

Avaliar, de maneira rigorosa e reprodutível, a viabilidade da detecção
automática de **fake news em português brasileiro** sobre o corpus
**FakeRecogna 2.0**, considerando: viés do dataset, robustez fora-da-distribuição,
robustez adversarial, interpretabilidade e prontidão para deployment.

## Pipeline experimental (em ordem)

| Etapa | O que faz                                                  | Onde está no projeto                                     |
|------:|-------------------------------------------------------------|----------------------------------------------------------|
| 1     | Setup determinístico (seeds, logging, dirs)                | `src/fakerecogna2/{config,utils}`                        |
| 2     | Carregamento + mapeamento determinístico de colunas        | `src/fakerecogna2/data/loading.py`                       |
| 3     | Integridade: dedup, fonte, comprimento, NER, temporal      | `src/fakerecogna2/data/integrity_checks.py`              |
| 4     | EDA lexical: log-odds (Monroe), Chi-square TF-IDF          | `src/fakerecogna2/statistics/log_odds.py`                |
| 5     | Pré-processamento textual                                  | `src/fakerecogna2/preprocessing/text_cleaning.py`        |
| 6     | Splits estratificados (train/val/test)                     | `src/fakerecogna2/data/splits.py`                        |
| 7     | Embeddings BERTimbau (token-level)                         | `src/fakerecogna2/features/embeddings.py`                |
| 8     | Baselines (LinearSVC, LogReg, MLP)                         | `src/fakerecogna2/models/baselines.py`                   |
| 9     | Deep models (CNN, LSTM, ConvLSTM) × 3 seeds → IC           | `src/fakerecogna2/models/{neural_models,training}.py`    |
| 10    | Ensembles (média e ponderado)                              | `src/fakerecogna2/models/ensembles.py`                   |
| 11    | BERTimbau fine-tuned (end-to-end)                          | `src/fakerecogna2/models/bertimbau_finetune.py`          |
| 12    | Abstrativa vs Extrativa                                    | `src/fakerecogna2/models/ensembles.py`                   |
| 13    | Bootstrap CI, McNemar+Holm, calibração (ECE/Brier), 5-CV   | `src/fakerecogna2/evaluation/{bootstrap,calibration,cross_validation}.py` + `statistics/significance_tests.py` |
| 14    | Stress tests: fonte, NER mask, temporal                    | `src/fakerecogna2/evaluation/stress_tests.py`            |
| 15    | LIME + erros por comprimento/fonte                         | `src/fakerecogna2/explainability/lime_explainer.py`, `evaluation/error_analysis.py` |
| 16    | Tabela consolidada + gráfico final                         | `src/fakerecogna2/reports/consolidated.py`, `evaluation/plots.py` |
| 17    | Cross-dataset OOD (FakeRecogna → Fake.br)                  | `src/fakerecogna2/evaluation/cross_dataset.py`           |
| 18    | XAI gradiente: Integrated Gradients + Attention Rollout    | `src/fakerecogna2/explainability/{integrated_gradients,attention_rollout,comparison}.py` |
| 19    | Robustez adversarial (perturbações + back-translation)     | `src/fakerecogna2/adversarial/{perturbations,back_translation,runner}.py` |
| 20    | PLMs maiores (BERTimbau-large, XLM-R, mDeBERTa)            | `src/fakerecogna2/models/plm_finetune.py`                |
| 21    | Paraphrasing Equalizer                                     | `src/fakerecogna2/preprocessing/paraphrasing.py`         |
| 22    | Métricas de deployment (latência, disco, Pareto)           | `src/fakerecogna2/deployment/benchmarks.py`              |
| 23    | Relatório consolidado (Markdown + JSON)                    | `src/fakerecogna2/reports/consolidated.py`               |
| Ablações | Pré-proc, max_len, learning curve, faixa de comprimento, equalizador linguístico, estabilidade XAI, disk size | `src/fakerecogna2/evaluation/ablations.py`, `preprocessing/equalizer.py`, `explainability/stability.py` |

## Métricas usadas

- **Classificação**: accuracy, precision/recall/F1 macro, ROC-AUC
- **Calibração**: ECE, Brier score, reliability diagram
- **Significância**: McNemar pairwise + correção Holm
- **Robustez**: bootstrap CI (1000+ iterações), 5-fold CV sem leakage
- **XAI**: top-k LIME, Integrated Gradients, Attention Rollout, Jaccard de
  estabilidade
- **Deployment**: latência média/p95 com warmup, tamanho em disco, fronteira
  de Pareto F1 × latência

## Reprodutibilidade

- Seed determinística: `42` (constante `SEED` em
  `src/fakerecogna2/config.py`).
- Seeds para IC: `[42, 7, 2024]` (`SEEDS_MULTI`).
- `cudnn.deterministic = True` (custo de performance assumido).
- Splits estratificados fixos sobre `text_proc` (texto preprocessado).
- Hiperparâmetros congelados em `configs/config.yaml`.

## Datasets

- **FakeRecogna 2.0** — carregado do Hugging Face Hub (`FakeRecogna/FakeRecogna2`).
- **Fake.br-Corpus** — usado em cross-dataset; espera-se cópia local em
  `data/external/Fake.br-Corpus-master` (configurável em
  `configs/config.yaml`). Download:
  https://github.com/roneysco/Fake.br-Corpus

## Modelos

| Categoria        | Modelos                                                                  |
|------------------|--------------------------------------------------------------------------|
| Baselines        | LinearSVC, LogisticRegression, MLPClassifier (sobre TF-IDF e BERT[CLS])  |
| Deep             | CNN, BiLSTM, ConvLSTM sobre embeddings BERTimbau token-level             |
| Ensembles        | Média simples e ponderado (otimizado no validation)                      |
| BERTimbau FT     | `neuralmind/bert-base-portuguese-cased` fine-tuned                       |
| PLMs maiores     | BERTimbau-large, XLM-RoBERTa-base, mDeBERTa-v3-base                      |
| Back-translation | Helsinki-NLP/opus-mt (PT↔EN)                                             |

## Atualizações de agosto/2026 (pós-revisão ENIAC 2026)

Motivadas pela revisão do artigo aceito no ENIAC 2026:

- **Alinhamento de rótulos cross-dataset fixado ex-ante** (convenção canônica
  0=real, 1=fake em todo o caminho OOD); o auto-flip de polaridade foi
  removido — a verificação de polaridade agora apenas alerta erro de setup,
  sem alterar predições (`evaluation/cross_dataset.py`).
- **Isolamento do backbone**: `train_bertimbau_finetune` deepcopia o BERTimbau
  antes do fine-tuning, evitando que avaliações posteriores reutilizem um
  encoder mutado.
- **Sondas de atalho** (`scripts/16_shortcut_probes.py`): Regressões
  Logísticas usando apenas nº de tokens, apenas top-K termos (log-odds do
  treino) ou apenas metadados (categoria+ano). Destaque: metadados sozinhos
  atingem F1-macro 0,9611, empatando com o ensemble neural (0,9612).
- **Reexecução limpa da avaliação cross-dataset nas duas direções**
  (`scripts/17_cross_dataset_inverse.py` e
  `scripts/18_cross_dataset_direct_clean.py`). Números oficiais (F1-macro):
  FakeRecogna→Fake.br 0,6959 (Ens3) / 0,6833 (BERTimbau FT);
  Fake.br→FakeRecogna 0,6883 (Ens2) / 0,6772 (BERTimbau FT) — queda de
  27–28 pontos frente ao IID, simétrica nas duas direções.
- CSVs novos em `outputs/metrics/`: `23_shortcut_probes.csv`,
  `23_test_support_by_year.csv`; artefatos cross-dataset regravados sob o
  alinhamento canônico.
