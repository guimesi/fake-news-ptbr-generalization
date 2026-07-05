# Relatorio consolidado - XAI <-> Erros (standalone)

Gerado a partir dos CSVs em `outputs/metrics/`, sem re-executar o
pipeline. Para um cruzamento exemplo-por-exemplo (categoria x
celula x token), execute `run_all.py` que dispara a etapa 12d com
`consolidate_xai_errors`.

---

## 1. Categorias com taxa de erro acima da mediana - Ens3 (CNN+LSTM+ConvLSTM)

Mediana da taxa de erro entre categorias: **0.0368**.
Categorias com taxa de erro acima da mediana (15 listadas):

| Categoria          |    N |   Erros |   Taxa erro |    Acc |     F1 |
|:-------------------|-----:|--------:|------------:|-------:|-------:|
| Dinheiro           |   19 |       2 |      0.1053 | 0.8947 | 0.4722 |
| esporte            |   12 |       1 |      0.0833 | 0.9167 | 0.4783 |
| UOL Confere        |  471 |      39 |      0.0828 | 0.9172 | 0.4784 |
| Mundo              |  156 |      12 |      0.0769 | 0.9231 | 0.48   |
| Saúde              |  217 |      14 |      0.0645 | 0.9355 | 0.4833 |
| Política           |   16 |       1 |      0.0625 | 0.9375 | 0.4839 |
| entretenimento     |  837 |      49 |      0.0585 | 0.9415 | 0.4849 |
| Ciência            |   18 |       1 |      0.0556 | 0.9444 | 0.4857 |
| política           | 1675 |      90 |      0.0537 | 0.9463 | 0.8966 |
| Checamos           |   39 |       2 |      0.0513 | 0.9487 | 0.4868 |
| Esporte            |   79 |       4 |      0.0506 | 0.9494 | 0.487  |
| Políticas públicas |   20 |       1 |      0.05   | 0.95   | 0.4872 |
| Fora de Contexto   |   20 |       1 |      0.05   | 0.95   | 0.4872 |
| Pandemia           |   22 |       1 |      0.0455 | 0.9545 | 0.4884 |
| Correntes          |   22 |       1 |      0.0455 | 0.9545 | 0.4884 |

## 2. Categorias com taxa de erro acima da mediana - BERTimbau FT

Mediana da taxa de erro entre categorias: **0.0412**.
Categorias com taxa de erro acima da mediana (15 listadas):

| Categoria          |   N |   Erros |   Taxa erro |    Acc |     F1 |
|:-------------------|----:|--------:|------------:|-------:|-------:|
| Política           |  16 |       3 |      0.1875 | 0.8125 | 0.4483 |
| Checamos           |  39 |       5 |      0.1282 | 0.8718 | 0.4658 |
| Ciência            |  18 |       2 |      0.1111 | 0.8889 | 0.4706 |
| Saúde              | 217 |      23 |      0.106  | 0.894  | 0.472  |
| Fora de Contexto   |  20 |       2 |      0.1    | 0.9    | 0.4737 |
| Pandemia           |  47 |       4 |      0.0851 | 0.9149 | 0.4778 |
| esporte            |  12 |       1 |      0.0833 | 0.9167 | 0.4783 |
| UOL Confere        | 471 |      39 |      0.0828 | 0.9172 | 0.4784 |
| Fotos              |  14 |       1 |      0.0714 | 0.9286 | 0.4815 |
| Mundo              | 156 |      11 |      0.0705 | 0.9295 | 0.4817 |
| Dinheiro           |  19 |       1 |      0.0526 | 0.9474 | 0.4865 |
| Esporte            |  79 |       4 |      0.0506 | 0.9494 | 0.487  |
| ciência            | 339 |      17 |      0.0501 | 0.9499 | 0.8554 |
| Políticas públicas |  20 |       1 |      0.05   | 0.95   | 0.4872 |
| Animais            |  60 |       3 |      0.05   | 0.95   | 0.4872 |

## 3. Tokens LIME mais influentes por celula da matriz de confusao

Pesos absolutos somados sobre os exemplos LIME (n=3 por celula).
Top 10 tokens por celula da matriz de confusao:

### TP

| token      |   abs_weight_sum |   n_examples |
|:-----------|-----------------:|-------------:|
| reunião    |           0.1897 |            3 |
| senador    |           0.1829 |            3 |
| participou |           0.1597 |            3 |
| veja       |           0.128  |            3 |
| revista    |           0.1183 |            3 |
| conta      |           0.1134 |            3 |
| bndes      |           0.1116 |            3 |
| bolsonaro  |           0.0994 |            3 |
| presidente |           0.0947 |            3 |
| disse      |           0.0932 |            3 |

### TN

| token            |   abs_weight_sum |   n_examples |
|:-----------------|-----------------:|-------------:|
| justiça          |           0.0099 |            3 |
| investigado      |           0.008  |            3 |
| responsabilidade |           0.007  |            3 |
| leia             |           0.0063 |            3 |
| pública          |           0.0062 |            3 |
| flávio           |           0.0058 |            3 |
| segurança        |           0.0057 |            3 |
| informações      |           0.0051 |            3 |
| redes            |           0.005  |            3 |
| checagem         |           0.0049 |            3 |

### FP

| token      |   abs_weight_sum |   n_examples |
|:-----------|-----------------:|-------------:|
| deixa      |           0.2213 |            3 |
| moraes     |           0.1908 |            3 |
| selecionou |           0.164  |            3 |
| pediu      |           0.1554 |            3 |
| alexandre  |           0.1482 |            3 |
| tribunal   |           0.1433 |            3 |
| pautados   |           0.1309 |            3 |
| outubro    |           0.1256 |            3 |
| checar     |           0.1243 |            3 |
| enviada    |           0.1155 |            3 |

### FN

| token   |   abs_weight_sum |   n_examples |
|:--------|-----------------:|-------------:|
| afirma  |           0.1961 |            3 |
| coluna  |           0.1448 |            3 |
| anos    |           0.1345 |            3 |
| assim   |           0.1327 |            3 |
| apenas  |           0.1137 |            3 |
| cerca   |           0.1122 |            3 |
| seis    |           0.1101 |            3 |
| brasil  |           0.1068 |            3 |
| nascida |           0.1006 |            3 |
| mora    |           0.0991 |            3 |

## 4. Distribuicao de erros por fonte - Ens3

Top 15 fontes (por taxa de erro):

| Fonte                   |    N |   Erros |   Taxa erro |    Acc |     F1 |
|:------------------------|-----:|--------:|------------:|-------:|-------:|
| politica.estadao.com.br |  256 |      19 |      0.0742 | 0.9258 | 0.4807 |
| noticias.uol.com.br     | 1835 |     136 |      0.0741 | 0.9259 | 0.9076 |
| projetocomprova.com.br  |  173 |      10 |      0.0578 | 0.9422 | 0.4851 |
| boatos.org              | 1695 |      66 |      0.0389 | 0.9611 | 0.4901 |
| economia.uol.com.br     |   54 |       2 |      0.037  | 0.963  | 0.4906 |
| extra.globo.com         |  212 |       6 |      0.0283 | 0.9717 | 0.4928 |
| g1.globo.com            | 1135 |      32 |      0.0282 | 0.9718 | 0.9704 |
| aosfatos.org            |  477 |      10 |      0.021  | 0.979  | 0.4947 |
| e-farsas.com            |  717 |      15 |      0.0209 | 0.9791 | 0.4947 |
| uol.com.br              |  131 |       2 |      0.0153 | 0.9847 | 0.4962 |
| lupa.uol.com.br         |  644 |       7 |      0.0109 | 0.9891 | 0.4973 |
| cnnbrasil.com.br        |  276 |       3 |      0.0109 | 0.9891 | 0.4973 |
| checamos.afp.com        |  320 |       3 |      0.0094 | 0.9906 | 0.4976 |
| gov.br                  |   42 |       0 |      0      | 1      | 1      |
| estadao.com.br          |   16 |       0 |      0      | 1      | 1      |

## 5. Distribuicao de erros por ano - Ens3

Distribuicao de erros por ano:

|   Ano |    N |   Erros |   Taxa erro |    Acc |     F1 |
|------:|-----:|--------:|------------:|-------:|-------:|
|  2014 |   48 |       3 |      0.0625 | 0.9375 | 0.4839 |
|  2015 |   34 |       0 |      0      | 1      | 1      |
|  2016 |  101 |       1 |      0.0099 | 0.9901 | 0.4975 |
|  2017 |  265 |      17 |      0.0642 | 0.9358 | 0.4834 |
|  2018 |  384 |       9 |      0.0234 | 0.9766 | 0.4941 |
|  2019 |  495 |      15 |      0.0303 | 0.9697 | 0.6662 |
|  2020 | 2584 |     109 |      0.0422 | 0.9578 | 0.9437 |
|  2021 | 2307 |     120 |      0.052  | 0.948  | 0.9276 |
|  2022 |  663 |      33 |      0.0498 | 0.9502 | 0.8053 |
|  2023 |  350 |      11 |      0.0314 | 0.9686 | 0.492  |

## 6. Distribuicao de erros por confianca preditiva - Ens3

Distribuicao de erros por quartil de confianca preditiva:

| Quartil_conf   | Faixa       |    N |   Erros |   Taxa erro |    Acc |
|:---------------|:------------|-----:|--------:|------------:|-------:|
| Q1 (baixa)     | 0.000–0.911 | 2618 |     368 |      0.1406 | 0.8594 |
| Q2             | 0.911–0.938 | 2617 |      31 |      0.0118 | 0.9882 |
| Q3             | 0.938–0.947 | 2618 |       4 |      0.0015 | 0.9985 |
| Q4 (alta)      | 0.947–1.000 | 2618 |       3 |      0.0011 | 0.9989 |

---

## Interpretacao orientativa

Use este relatorio para responder perguntas como:

1. **Quais categorias concentram erros?** Cruze a Secao 1/2 com a
   Secao 5 (ano) e a Secao 4 (fonte) para verificar se uma categoria
   especifica esta associada a uma fonte/ano com cobertura desigual.

2. **Os tokens LIME refletem viseis de dominio?** Se TP/TN (acertos)
   priorizam tokens semanticamente plausiveis (ex.: "verificou",
   "checagem") e FP/FN (erros) priorizam tokens generales ou
   metatextuais, isso aponta para confianca em marcas de origem
   em vez de conteudo discriminativo.

3. **Erros por confianca:** se quartis baixos de confianca
   concentram mais erros que altos, isso e calibracao saudavel.
   Se erros aparecem em alta confianca, ha *overconfident wrong*,
   o que requer revisao de calibracao (ver `13_calibration.csv`).
