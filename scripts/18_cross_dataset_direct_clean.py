"""Etapa 18 — Reexecução limpa da direção direta do Ens3 (→ Fake.br e → Extrativa).

Motivação (auditoria da qualificação + CHANGELOG §16): no run original, a
avaliação OOD do Ens3 extraía embeddings com o backbone BERTimbau já mutado
pelo fine-tuning (`BERTClassifier` não copiava o modelo). Este script
retreina o ensemble profundo e reavalia a direção direta com o backbone
pré-treinado intocado (nenhum fine-tuning acontece no processo).

As linhas do BERTimbau FT em `17_cross_dataset_ood.csv` NÃO são tocadas
(o predict do BERT FT usa o próprio modelo fim-a-fim e já era limpo):
apenas as linhas do Ens3 são regravadas, junto com `cm_ood_ens3_*`,
`17_cross_dataset_delta.csv` e `17_polarity_audit_log.csv`.

Backup prévio: `outputs/metrics/_pre_rerun_direto_2026-08/`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_features
from _training import make_predict_ens3, train_deep_ensemble

METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
BACKUP = METRICS / "_pre_rerun_direto_2026-08"

AFFECTED = [
    METRICS / "17_cross_dataset_ood.csv",
    METRICS / "17_cross_dataset_delta.csv",
    METRICS / "17_polarity_audit_log.csv",
    METRICS / "cm_ood_ens3_fakebr_cm.csv",
    METRICS / "cm_ood_ens3_fakebr_per_class.csv",
    METRICS / "cm_ood_ens3_extrativa_cm.csv",
    METRICS / "cm_ood_ens3_extrativa_per_class.csv",
    FIGURES / "cm_ood_ens3_fakebr_cm.png",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-extrativa", action="store_true",
                   help="pula a reavaliação → Extrativa (mais rápida de todas, mas 52k textos)")
    args = p.parse_args()

    BACKUP.mkdir(exist_ok=True)
    for f in AFFECTED:
        if f.exists():
            shutil.copy2(f, BACKUP / f.name)

    # Dados + embeddings do corpus principal (backbone pré-treinado; nenhum
    # fine-tuning ocorre neste script, logo o backbone permanece limpo).
    ctx = prepare_through_features(
        do_fakebr=True, do_extrativa=not args.skip_extrativa,
        fit_tfidf_vec=False,
    )
    if "df_fakebr" not in ctx.extras:
        print("Fake.br não carregado. Ajuste `data.fakebr_local_path` no config.")
        return 1

    train_deep_ensemble(ctx)

    import pandas as pd

    from fakerecogna2.evaluation import evaluate_on_external_corpus
    from fakerecogna2.evaluation.confusion_matrices import save_cm
    from fakerecogna2.utils.io_utils import RESULTS, save_table

    predict_ens3 = make_predict_ens3(ctx, batch_size=32)

    results = [
        evaluate_on_external_corpus(
            ctx.extras["df_fakebr"], "text", "label", "Ens3 → Fake.br",
            predict_ens3, encoder=ctx.extras.get("label_encoder"),
        )
    ]
    if not args.skip_extrativa and ctx.extras.get("df_extr") is not None:
        results.append(
            evaluate_on_external_corpus(
                ctx.extras["df_extr"], "text", "label", "Ens3 → Extrativa",
                predict_ens3, encoder=ctx.extras.get("label_encoder"),
            )
        )

    # CMs com nomes semânticos na convenção canônica (0=real, 1=fake)
    for r in results:
        suffix = "fakebr" if "Fake.br" in r["model"] else "extrativa"
        save_cm(
            r["y_true"], r["y_pred"], model_name=r["model"],
            class_names=["real", "fake"],
            save_as_prefix=f"cm_ood_ens3_{suffix}",
            title_suffix="cross-dataset",
        )

    # Merge nas tabelas canônicas: substitui só as linhas do Ens3
    rows_new = [{k: v for k, v in r.items() if k not in ("y_true", "y_pred")}
                for r in results]
    ood = pd.read_csv(METRICS / "17_cross_dataset_ood.csv")
    ood = ood[~ood["model"].isin([r["model"] for r in rows_new])]
    ood = pd.concat([pd.DataFrame(rows_new), ood], ignore_index=True)
    ood.to_csv(METRICS / "17_cross_dataset_ood.csv", index=False)

    audit = [{"model": r["model"], "N": r["N"],
              "acc_direct": r["polarity_acc_direct"],
              "acc_flipped_hypothetical": r["polarity_acc_flipped"],
              "delta_flip_minus_direct":
                  r["polarity_acc_flipped"] - r["polarity_acc_direct"],
              "polarity_alert": r["polarity_alert"]} for r in rows_new]
    old_audit = pd.read_csv(METRICS / "17_polarity_audit_log.csv")
    old_audit = old_audit[~old_audit["model"].isin([a["model"] for a in audit])]
    pd.concat([pd.DataFrame(audit), old_audit], ignore_index=True).to_csv(
        METRICS / "17_polarity_audit_log.csv", index=False)

    f1_iid_ens3 = RESULTS.get("Ens3 (CNN+LSTM+ConvLSTM)", {}).get("F1")
    if f1_iid_ens3 is not None:
        delta = pd.DataFrame([
            {"Model": "Ens3", "OOD corpus": r["model"].split("→")[-1].strip(),
             "F1 IID": round(float(f1_iid_ens3), 4),
             "F1 OOD": round(float(r["F1"]), 4),
             "ΔF1 (IID-OOD)": round(float(f1_iid_ens3) - float(r["F1"]), 4)}
            for r in rows_new
        ])
        save_table(delta, "17_cross_dataset_delta")

    print(ood.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
