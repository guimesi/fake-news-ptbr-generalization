"""Fine-tuning fim-a-fim do BERTimbau (cell 37)."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from ..config import MAX_LEN
from ..utils.io_utils import RESULTS
from ..utils.logging_utils import get_logger
from ..utils.seed import set_seed

log = get_logger()




# -- API limpa ----------------------------------------------------------------
class BERTClassifier(nn.Module):
    """BERTimbau backbone + dropout + Linear (binário). Congela tudo exceto últimas `trainable_last` params."""

    def __init__(
        self,
        bert,
        num_classes: int,
        dropout: float = 0.1,
        trainable_last: int = 30,
    ):
        super().__init__()
        self.bert = bert
        self.do = nn.Dropout(dropout)
        self.clf = nn.Linear(bert.config.hidden_size, num_classes)
        params = list(self.bert.parameters())
        for p in params[:-trainable_last]:
            p.requires_grad = False

    def forward(self, ids, mask):
        h = self.bert(ids, mask).last_hidden_state[:, 0, :]
        return self.clf(self.do(h))


def make_bert_loader(
    tokenizer, texts: list[str], labels, batch_size: int = 16, max_seq_len: int = MAX_LEN
) -> DataLoader:
    """Tokeniza + monta DataLoader pro fine-tuning."""
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    return DataLoader(
        TensorDataset(
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor(np.asarray(labels), dtype=torch.long),
        ),
        batch_size=batch_size,
    )


def train_bertimbau_finetune(
    bert_model,
    tokenizer,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    num_classes: int = 2,
    epochs: int = 10,
    lr: float = 2e-5,
    es_patience: int = 3,
    batch_size: int = 16,
    max_seq_len: int = MAX_LEN,
    device: str = "cpu",
    label_smoothing: float = 0.1,
    weight_decay: float = 1e-4,
    seed: int = 42,
    result_key: str = "BERTimbau FT",
) -> tuple[BERTClassifier, torch.Tensor]:
    """Treina BERTimbau FT com early stop. Registra em RESULTS. Retorna (clf, probs_test).

    O backbone recebido é DEEPCOPIADO antes do fine-tuning: sem isso, as
    últimas camadas de `bert_model` eram mutadas in-place e qualquer etapa
    posterior que reutilizasse o mesmo objeto para extrair embeddings (p.ex.
    `make_predict_ens3` nas avaliações OOD) usava um encoder diferente do
    que gerou os embeddings de treino (fragilidade apontada na auditoria da
    qualificação; isolamento previsto na Tarefa 2 do Cap. 6).
    """
    import copy

    set_seed(seed)
    b_tr = make_bert_loader(tokenizer, X_train, y_train, batch_size, max_seq_len)
    b_vl = make_bert_loader(tokenizer, X_val, y_val, batch_size, max_seq_len)
    b_te = make_bert_loader(tokenizer, X_test, y_test, batch_size, max_seq_len)

    clf = BERTClassifier(copy.deepcopy(bert_model), num_classes).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = optim.AdamW(
        [p for p in clf.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )

    best_vl, best_st, pat = float("inf"), None, 0
    for ep in range(epochs):
        clf.train()
        tl = 0.0
        for batch in tqdm(b_tr, desc=f"BERT ep{ep+1}", leave=False):
            ids, mask, lab = [x.to(device) for x in batch]
            opt.zero_grad()
            loss = crit(clf(ids, mask), lab)
            loss.backward()
            opt.step()
            tl += loss.item()
        clf.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in b_vl:
                ids, mask, lab = [x.to(device) for x in batch]
                vl += crit(clf(ids, mask), lab).item()
        vl /= max(1, len(b_vl))
        log.info(f"BERT ep{ep+1}: trL={tl/max(1,len(b_tr)):.4f} vlL={vl:.4f}")
        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.clone() for k, v in clf.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= es_patience:
                log.info(f"BERT early stop ep {ep+1}")
                break

    if best_st:
        clf.load_state_dict(best_st)
    clf.eval()

    bp, bl, probs_list = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in b_te:
            ids, mask, lab = [x.to(device) for x in batch]
            logits = clf(ids, mask)
            p = torch.softmax(logits, 1).cpu()
            probs_list.append(p)
            bp.extend(logits.argmax(1).cpu().numpy())
            bl.extend(lab.cpu().numpy())
    probs_all = torch.cat(probs_list, 0)
    RESULTS[result_key] = {
        "Accuracy": accuracy_score(bl, bp),
        "Precision": precision_score(bl, bp, average="macro", zero_division=0),
        "Recall": recall_score(bl, bp, average="macro", zero_division=0),
        "F1": f1_score(bl, bp, average="macro", zero_division=0),
        "Inference (ms)": (time.time() - t0) / max(1, len(bl)) * 1000,
    }
    log.info(f"{result_key}: {RESULTS[result_key]}")
    return clf, probs_all


__all__ = [
    "BERTClassifier",
    "make_bert_loader",
    "train_bertimbau_finetune",
]
