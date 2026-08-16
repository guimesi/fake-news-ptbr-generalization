"""Ablações finais (Seções A, B, D, E e quartil curto).

Cobre as células 94 (A.2), 96 (B.1), 100 (D.1), 102 (E.1) e 117. As ablações
C (extrativa) e F (equalizador linguístico) ficam nas funções de
`models.train_ensemble_on_variant` e `preprocessing.build_nometatext_dataset`
respectivamente (chamadas pelos scripts 04 e 11).
"""

from __future__ import annotations

import gc
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm.auto import tqdm

from ..config import BATCH_SIZE, SEED
from ..features.streaming import StreamingEmbDataset, make_streaming_collate
from ..models.training import train_model
from ..preprocessing.text_cleaning import (
    preprocess_base,
    preprocess_lemma,
    preprocess_stem,
)
from ..utils.io_utils import save_plot, save_table
from ..utils.logging_utils import get_logger

log = get_logger()


# -- A.2: Pré-processamento (cell 94) ----------------------------------------
def preprocessing_ablation(
    df: pd.DataFrame,
    n_samples: int = 15000,
    test_size: float = 0.2,
    seed: int = SEED,
    save_as: str | None = "preprocessing_ablation",
) -> pd.DataFrame:
    """Compara base / stemming / lemmatização com TF-IDF + LogReg."""
    n_samples = min(n_samples, len(df))
    df_ab = df.sample(n_samples, random_state=seed).reset_index(drop=True)
    log.info(f"[Ablação A] Gerando variantes em {n_samples} amostras...")

    rows = []
    variants = [
        ("Base (stopwords)", preprocess_base),
        ("Stemming (RSLP)", preprocess_stem),
        ("Lemmatização (spaCy)", preprocess_lemma),
    ]
    for prep_name, fn in variants:
        tqdm.pandas(desc=prep_name)
        texts = df_ab["text"].progress_apply(fn).tolist()
        labels = df_ab["label_enc"].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            texts, labels, test_size=test_size, stratify=labels, random_state=seed
        )
        tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        Xtr_v, Xte_v = tf.fit_transform(Xtr), tf.transform(Xte)
        lr = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr_v, ytr)
        yp = lr.predict(Xte_v)
        rows.append({
            "Preprocessamento": prep_name,
            "Accuracy": accuracy_score(yte, yp),
            "F1": f1_score(yte, yp, average="macro", zero_division=0),
            "Vocab": len(tf.get_feature_names_out()),
        })
        log.info(f"  {prep_name}: Acc={rows[-1]['Accuracy']:.4f} F1={rows[-1]['F1']:.4f}")

    df_out = pd.DataFrame(rows).round(4)
    print("\n=== Ablação A — Pré-processamento ===")
    print(tabulate(df_out, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df_out, save_as)
    return df_out


# -- B.1: Comprimento de sequência (cell 96) ---------------------------------
def seqlen_ablation(
    ctx,
    seq_len: int = 300,
    epochs: int = 15,
    save_as: str | None = "seqlen_ablation",
) -> pd.DataFrame:
    """Re-treina CNN+LSTM com `seq_len` (default 300) via streaming."""
    from torch.utils.data import DataLoader

    from ..models import TextCNN, TextLSTM
    from ..models.training import evaluate_model
    from ..utils.io_utils import RESULTS

    log.info(f"[Ablação B] Treinando com max_len={seq_len} (streaming)...")
    embed_dim = ctx.extras.get("embed_dim", 768)
    num_classes = len(ctx.extras.get("class_names", ["fake", "real"]))

    collate = make_streaming_collate(
        ctx.tokenizer, ctx.bert_model, device=ctx.device, max_len=seq_len
    )
    ld_tr = DataLoader(
        StreamingEmbDataset(ctx.X_train_text, ctx.y_train, max_len=seq_len),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
    )
    ld_vl = DataLoader(
        StreamingEmbDataset(ctx.X_val_text, ctx.y_val, max_len=seq_len),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
    )
    ld_te = DataLoader(
        StreamingEmbDataset(ctx.X_test_text, ctx.y_test, max_len=seq_len),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
    )

    cnn = TextCNN(embed_dim, num_classes, 0.5)
    cnn, _ = train_model(
        cnn, ld_tr, ld_vl, device=ctx.device, epochs=epochs,
        model_name=f"CNN_{seq_len}", seed=ctx.seed,
    )
    _, _, _, cnn_p = evaluate_model(cnn, ld_te, device=ctx.device, name=f"CNN_{seq_len}")

    lstm = TextLSTM(embed_dim, 128, 2, num_classes, 0.4)
    lstm, _ = train_model(
        lstm, ld_tr, ld_vl, device=ctx.device, epochs=epochs,
        model_name=f"LSTM_{seq_len}", seed=ctx.seed,
    )
    _, _, _, lstm_p = evaluate_model(lstm, ld_te, device=ctx.device, name=f"LSTM_{seq_len}")

    ens_preds = ((cnn_p + lstm_p) / 2).argmax(1).numpy()
    f1_new = f1_score(ctx.y_test, ens_preds, average="macro", zero_division=0)

    baseline_f1 = RESULTS.get("Ens2 (CNN+LSTM)", {}).get("F1", float("nan"))
    df = pd.DataFrame(
        [
            {"SEQ_LEN": ctx.max_seq_len, "Ensemble F1": baseline_f1},
            {"SEQ_LEN": seq_len, "Ensemble F1": f1_new},
        ]
    ).round(4)
    print("\n=== Ablação B — Comprimento de sequência ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)

    del cnn, lstm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


# -- D.1: Learning curve (cell 100) -------------------------------------------
def learning_curve_ablation(
    ctx,
    fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0),
    epochs: int = 10,
    save_as: str | None = "learning_curve",
    save_plot_as: str | None = "D_learning_curve",
) -> pd.DataFrame:
    """Treina ensemble CNN+LSTM em subsets crescentes do treino."""
    from torch.utils.data import DataLoader

    from ..models import TextCNN, TextLSTM

    collate = make_streaming_collate(
        ctx.tokenizer, ctx.bert_model, device=ctx.device, max_len=ctx.max_seq_len
    )
    ld_vl = DataLoader(
        StreamingEmbDataset(ctx.X_val_text, ctx.y_val),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
    )
    ld_te = DataLoader(
        StreamingEmbDataset(ctx.X_test_text, ctx.y_test),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate,
    )

    embed_dim = ctx.extras.get("embed_dim", 768)
    num_classes = len(ctx.extras.get("class_names", ["fake", "real"]))
    rows = []

    for frac in tqdm(fractions, desc="Learning curve"):
        n = int(len(ctx.X_train_text) * frac)
        sub_X = ctx.X_train_text[:n]
        sub_y = ctx.y_train[:n]
        ld_tr = DataLoader(
            StreamingEmbDataset(sub_X, sub_y),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate,
        )

        lc_cnn = TextCNN(embed_dim, num_classes, 0.5)
        lc_cnn, _ = train_model(
            lc_cnn, ld_tr, ld_vl, device=ctx.device, epochs=epochs,
            model_name=f"LC_CNN_{frac}", seed=ctx.seed,
        )
        lc_lstm = TextLSTM(embed_dim, 128, 2, num_classes, 0.4)
        lc_lstm, _ = train_model(
            lc_lstm, ld_tr, ld_vl, device=ctx.device, epochs=epochs,
            model_name=f"LC_LSTM_{frac}", seed=ctx.seed,
        )

        lc_cnn.eval()
        lc_lstm.eval()
        ps, ls = [], []
        with torch.no_grad():
            for bx, by in ld_te:
                bx = bx.to(ctx.device)
                p = (torch.softmax(lc_cnn(bx), 1) + torch.softmax(lc_lstm(bx), 1)) / 2
                ps.extend(p.argmax(1).cpu().numpy())
                ls.extend(by.numpy())
        rows.append({
            "Fração": frac,
            "N_treino": n,
            "Accuracy": accuracy_score(ls, ps),
            "F1": f1_score(ls, ps, average="macro", zero_division=0),
        })
        del lc_cnn, lc_lstm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).round(4)
    print("\n=== Ablação D — Learning curve ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)

    if save_plot_as is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["Fração"] * 100, df["Accuracy"], "o-", lw=2, label="Accuracy", color="#1976D2")
        ax.plot(df["Fração"] * 100, df["F1"], "s-", lw=2, label="F1 macro", color="#C2185B")
        ax.set_xlabel("% dados de treino")
        ax.set_ylabel("Score")
        ax.set_title("Learning curve — Ensemble CNN+LSTM", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_plot(fig, save_plot_as)
    return df


# -- E.1: Performance por faixa (cell 102) -----------------------------------
def performance_by_length_ensemble(
    ctx,
    predict_fn: Callable[[list[str]], np.ndarray],
    model_name: str = "Ens3",
    save_as: str | None = "performance_by_length",
    save_plot_as: str | None = "E_performance_by_length",
) -> pd.DataFrame:
    """Acc/F1 por quartil de comprimento, usando `predict_fn`."""
    log.info("[Ablação E] Performance por comprimento (ensemble)...")
    probs = predict_fn(ctx.X_test_text)
    preds = probs.argmax(1)

    tl = np.array([len(t.split()) for t in ctx.X_test_text])
    qs = np.percentile(tl, [25, 50, 75])
    bins = [0, qs[0], qs[1], qs[2], tl.max() + 1]
    names = ["Curto", "Médio-Curto", "Médio-Longo", "Longo"]
    y = np.asarray(ctx.y_test)

    rows = []
    for i, nm in enumerate(names):
        mk = (tl >= bins[i]) & (tl < bins[i + 1])
        if mk.sum() > 0:
            rows.append({
                "Faixa": nm,
                "Tokens": f"{int(bins[i])}–{int(bins[i+1]-1)}",
                "N": int(mk.sum()),
                "Acc": accuracy_score(y[mk], preds[mk]),
                "F1": f1_score(y[mk], preds[mk], average="macro", zero_division=0),
            })
    df = pd.DataFrame(rows).round(4)
    print(f"\n=== Ablação E — Performance por comprimento ({model_name}) ===")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    if save_as is not None:
        save_table(df, save_as)

    if save_plot_as is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(df))
        w = 0.4
        ax.bar([i - w / 2 for i in x], df["Acc"], w, label="Accuracy", color="#1976D2", alpha=0.85)
        ax.bar([i + w / 2 for i in x], df["F1"], w, label="F1 macro", color="#C2185B", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [f'{r["Faixa"]}\n({r["N"]})' for _, r in df.iterrows()], fontsize=9
        )
        ax.set_ylabel("Score")
        ax.set_ylim(0.85, 1.0)
        ax.set_title(f"Performance por comprimento ({model_name})", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_plot(fig, save_plot_as)
    return df


# -- Quartil curto (cell 117) -------------------------------------------------
def short_quartile_class_distribution(
    X_test: list[str],
    y_test,
    max_tokens: int = 24,
) -> dict[str, float | int]:
    """Distribuição de classes no quartil curto (≤max_tokens)."""
    tl = np.array([len(t.split()) for t in X_test])
    mk = tl <= max_tokens
    y_short = np.asarray(y_test)[mk]
    n0 = int((y_short == 0).sum())
    n1 = int((y_short == 1).sum())
    total = len(y_short)

    print(f"Quartil Curto (≤{max_tokens} tokens): N = {total}")
    print(f"  Classe 0: {n0:4d}  ({n0/max(1,total):.1%})")
    print(f"  Classe 1: {n1:4d}  ({n1/max(1,total):.1%})")
    print(f"  Desbalanceamento: {max(n0, n1) / max(1, min(n0, n1)):.2f}x")

    y_all = np.asarray(y_test)
    print(f"\nTest completo (referência): N = {len(y_all)}")
    print(f"  Classe 0: {(y_all == 0).sum():5d}  ({(y_all == 0).mean():.1%})")
    print(f"  Classe 1: {(y_all == 1).sum():5d}  ({(y_all == 1).mean():.1%})")

    return {
        "n_short": total,
        "n_class_0": n0,
        "n_class_1": n1,
        "ratio_0": n0 / max(1, total),
        "ratio_1": n1 / max(1, total),
        "imbalance": max(n0, n1) / max(1, min(n0, n1)),
    }


__all__ = [
    "preprocessing_ablation",
    "seqlen_ablation",
    "learning_curve_ablation",
    "performance_by_length_ensemble",
    "short_quartile_class_distribution",
]
