"""Fine-tuning genérico de qualquer PLM HuggingFace (cells 73, 74)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ..config import MAX_LEN, PLM_CANDIDATES, SEED
from ..utils.io_utils import RESULTS, cleanup, save_table
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed

log = get_logger()




# -- API limpa ----------------------------------------------------------------
def fine_tune_plm(
    model_name: str,
    X_train, y_train, X_val, y_val, X_test, y_test,
    num_classes: int = 2,
    max_seq_len: int = MAX_LEN,
    batch_size: int = 16,
    lr: float = 2e-5,
    epochs: int = 5,
    es_patience: int = 2,
    trainable_last: int = 30,
    device: str = "cpu",
    seed: int = SEED,
) -> dict[str, float | int | None]:
    """Fine-tuning genérico de PLM. Limpa modelo da memória ao final."""
    from transformers import AutoModel, AutoTokenizer

    set_seed(seed)
    log.info(f"Fine-tuning {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    # torch_dtype=float32: alguns checkpoints (ex.: mdeberta-v3-base) estão
    # armazenados em FP16 e quebram o nn.Linear da cabeça (criada em FP32)
    # com "mat1 and mat2 must have the same dtype, but got Half and Float".
    plm = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32).to(device)

    class PLMClassifier(nn.Module):
        def __init__(self, backbone, n_classes, dropout=0.1):
            super().__init__()
            self.backbone = backbone
            self.do = nn.Dropout(dropout)
            self.clf = nn.Linear(backbone.config.hidden_size, n_classes)
            params = list(self.backbone.parameters())
            for p in params[:-trainable_last]:
                p.requires_grad = False

        def forward(self, ids, mask):
            h = self.backbone(ids, attention_mask=mask).last_hidden_state[:, 0, :]
            return self.clf(self.do(h))

    model = PLMClassifier(plm, num_classes).to(device)

    def _make_loader(X, y, bs=batch_size):
        enc = tok(
            list(X), padding="max_length", truncation=True,
            max_length=max_seq_len, return_tensors="pt",
        )
        return DataLoader(
            TensorDataset(
                enc["input_ids"], enc["attention_mask"],
                torch.tensor(np.asarray(y), dtype=torch.long),
            ),
            batch_size=bs,
        )

    loader_tr = _make_loader(X_train, y_train)
    loader_vl = _make_loader(X_val, y_val)
    loader_te = _make_loader(X_test, y_test)

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    best_vl, best_st, pat = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for batch in tqdm(loader_tr, desc=f"{model_name[:30]} ep{ep+1}", leave=False):
            ids, mask, lab = [x.to(device) for x in batch]
            opt.zero_grad()
            loss = crit(model(ids, mask), lab)
            loss.backward()
            opt.step()
            tl += loss.item()
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in loader_vl:
                ids, mask, lab = [x.to(device) for x in batch]
                vl += crit(model(ids, mask), lab).item()
        vl /= max(1, len(loader_vl))
        log.info(f"  ep{ep+1}: trL={tl/max(1,len(loader_tr)):.4f} vlL={vl:.4f}")
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= es_patience:
                break
    if best_st:
        model.load_state_dict(best_st)
    model.eval()

    yp, yl, probs_all = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in loader_te:
            ids, mask, lab = [x.to(device) for x in batch]
            logits = model(ids, mask)
            probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
            yp.extend(logits.argmax(1).cpu().numpy())
            yl.extend(lab.cpu().numpy())
    inf_ms = (time.time() - t0) / max(1, len(yl)) * 1000
    probs_arr = np.concatenate(probs_all, axis=0)

    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if device.startswith("cuda") else None
    )

    results = {
        "Accuracy": accuracy_score(yl, yp),
        "Precision": precision_score(yl, yp, average="macro", zero_division=0),
        "Recall": recall_score(yl, yp, average="macro", zero_division=0),
        "F1": f1_score(yl, yp, average="macro", zero_division=0),
        "Inference (ms)": inf_ms,
        "VRAM peak (MB)": peak_mem_mb,
        "Params (M)": sum(p.numel() for p in plm.parameters()) / 1e6,
        # Predicoes/labels/probabilidades para downstream
        # (Bootstrap CI, ECE/reliability, CM por classe).
        "y_pred": np.asarray(yp),
        "y_true": np.asarray(yl),
        "probs": probs_arr,
    }
    log.info(
        f'  → F1={results["F1"]:.4f}  VRAM={peak_mem_mb or 0:.0f}MB  '
        f'params={results["Params (M)"]:.0f}M'
    )

    del model, plm
    cleanup()
    return results


def evaluate_plm_candidates(
    candidates: list[tuple[str, int]] | None = None,
    X_train=None, y_train=None,
    X_val=None, y_val=None,
    X_test=None, y_test=None,
    num_classes: int = 2,
    max_seq_len: int = MAX_LEN,
    epochs: int = 4,
    es_patience: int = 2,
    device: str = "cpu",
    save_as: str | None = "20_larger_models",
    seeds: list[int] | None = None,
) -> dict[str, dict]:
    """Roda `fine_tune_plm` em varios candidatos, opcionalmente multi-seed.

    Args:
        candidates: lista de (model_name, batch_size).
        seeds: se fornecido (ex.: [42, 7]), roda cada PLM com cada seed e
            agrega metricas (mean+/-std). A primeira seed e usada como
            principal para downstream (y_pred/probs/CM). Default `[SEED]`
            = 1 seed (comportamento original).
    """
    if candidates is None:
        candidates = [
            (name, 8 if "large" in name else 16) for name in PLM_CANDIDATES
        ]
    if seeds is None:
        seeds = [SEED]
    multi_seed = len(seeds) > 1

    results: dict[str, dict] = {}
    multi_seed_log: list[dict] = []
    for plm_name, bs in candidates:
        try:
            per_seed: list[dict] = []
            principal_res: dict | None = None
            for sd in seeds:
                res = fine_tune_plm(
                    plm_name, X_train, y_train, X_val, y_val, X_test, y_test,
                    num_classes=num_classes, max_seq_len=max_seq_len,
                    batch_size=bs, epochs=epochs, es_patience=es_patience,
                    device=device, seed=sd,
                )
                per_seed.append(res)
                if principal_res is None:
                    principal_res = res
            short = plm_name.split("/")[-1]

            if multi_seed:
                # Agrega metricas escalares (mean+/-std) mas preserva
                # y_pred/y_true/probs da primeira seed (principal).
                agg = {}
                for metric in ("Accuracy", "Precision", "Recall", "F1",
                               "Inference (ms)", "VRAM peak (MB)"):
                    vals = [r[metric] for r in per_seed if r.get(metric) is not None]
                    if vals:
                        agg[f"{metric}_mean"] = float(np.mean(vals))
                        agg[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                        agg[metric] = float(np.mean(vals))
                agg["Params (M)"] = principal_res["Params (M)"]
                agg["seeds"] = list(seeds)
                agg["y_pred"] = principal_res["y_pred"]
                agg["y_true"] = principal_res["y_true"]
                agg["probs"] = principal_res["probs"]
                results[short] = agg
                multi_seed_log.append({
                    "Model": f"PLM: {short}",
                    "F1_mean": agg.get("F1_mean"),
                    "F1_std": agg.get("F1_std"),
                    "Acc_mean": agg.get("Accuracy_mean"),
                    "Acc_std": agg.get("Accuracy_std"),
                    "seeds": str(list(seeds)),
                })
            else:
                results[short] = principal_res

            # IMPORTANTE: RESULTS deve refletir o escalar da seed PRINCIPAL
            # (principal_res = primeira seed), pois sao as predicoes dessa seed
            # (principal_res["y_pred"]) que alimentam matriz de confusao,
            # bootstrap CI e a Tabela 1 do artigo. Em multi-seed, results[short]
            # carrega a MEDIA (mean+/-std) — usar a media aqui produziria um F1
            # em 16_final_results.csv inconsistente com cm_iid_* / 13_bootstrap_ci.
            # A media multi-seed vive apenas em plms_multiseed.csv / 20_larger_models.csv.
            RESULTS[f"PLM: {short}"] = {
                "Accuracy": principal_res["Accuracy"],
                "Precision": principal_res["Precision"],
                "Recall": principal_res["Recall"],
                "F1": principal_res["F1"],
                "Inference (ms)": principal_res["Inference (ms)"],
            }
        except Exception as e:
            log.error(f"Falha em {plm_name}: {e}")
            cleanup()

    if multi_seed and multi_seed_log:
        save_table(pd.DataFrame(multi_seed_log).round(6), "plms_multiseed")

    if results:
        # Remove arrays do CSV mas mantem no return (para downstream).
        results_csv = {
            k: {kk: vv for kk, vv in v.items()
                if kk not in ("y_true", "y_pred", "probs")}
            for k, v in results.items()
        }
        df = pd.DataFrame(results_csv).T.round(4)
        if "Params (M)" in df.columns:
            df["Params (M)"] = df["Params (M)"].astype(int)
        print("\n=== Modelos maiores ===")
        print(tabulate(df, headers="keys", tablefmt="github"))
        if save_as is not None:
            save_table(df.reset_index().rename(columns={"index": "Model"}), save_as)
    return results


__all__ = [
    "fine_tune_plm",
    "evaluate_plm_candidates",
]
