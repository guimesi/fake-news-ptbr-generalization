"""Pipeline completo — executa todas as etapas em um único processo.

Compartilha o `ExperimentContext` entre etapas (não há overhead de re-extrair
embeddings ou re-treinar modelos entre passos consecutivos).

Uso típico::

    python scripts/run_all.py                     # tudo (pode levar 3-4h em GPU)
    python scripts/run_all.py --quick             # skip cross/PLM/adv/ablações/paraphr
    python scripts/run_all.py --skip-plm          # pula PLMs maiores (lento)
    python scripts/run_all.py --skip-bt           # pula back-translation (lento)

Para rodar apenas uma etapa, use o script numerado correspondente
(`python scripts/04_train_deep_models.py`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _pipeline import prepare_through_features
from _training import (
    make_predict_bert_ft,
    make_predict_ens3,
    train_bert_classifier,
    train_deep_ensemble,
)


def _section(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="Atalho: pula cross-dataset, PLM, adversarial, ablações, paraphr.")
    p.add_argument("--skip-plm", action="store_true", help="Pula PLMs maiores.")
    p.add_argument("--skip-bt", action="store_true", help="Pula back-translation.")
    p.add_argument("--skip-cross", action="store_true", help="Pula cross-dataset.")
    p.add_argument("--skip-adv", action="store_true", help="Pula robustez adversarial.")
    p.add_argument("--skip-ablations", action="store_true", help="Pula ablações.")
    p.add_argument("--skip-xai", action="store_true", help="Pula XAI (LIME/IG/Attention).")
    p.add_argument("--skip-paraphr", action="store_true", help="Pula paraphrasing equalizer.")
    p.add_argument("--skip-cv", action="store_true", help="Pula 5-fold CV (caro).")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--trans-seeds", type=int, nargs="+", default=[42, 7],
                   help="Sementes para multi-seed dos Transformers FT (BERTimbau-base/large, XLM-R, mDeBERTa). Default [42, 7].")
    args = p.parse_args()

    skip_plm = args.skip_plm or args.quick
    skip_cross = args.skip_cross or args.quick
    skip_adv = args.skip_adv or args.quick
    skip_ablations = args.skip_ablations or args.quick
    skip_paraphr = args.skip_paraphr or args.quick

    t0_total = time.time()

    # 1. Dados + preprocessing + features
    _section("1. Dados, preprocessing, embeddings BERTimbau, TF-IDF")
    ctx = prepare_through_features(
        do_integrity_checks=True,
        do_extrativa=True,
        do_fakebr=not skip_cross,
    )
    class_names = ctx.extras.get("class_names", ["fake", "real"])

    # 2. Análise lexical (log-odds Dirichlet + Chi²) — gera CSVs pro relatório
    _section("2. Análise lexical (log-odds + Chi²)")
    from fakerecogna2.statistics import chi2_top, lex_analysis
    ctx.extras["lex_abst"] = lex_analysis(ctx.df, name="abstrativa")
    if "df_extr" in ctx.extras:
        ctx.extras["lex_extr"] = lex_analysis(ctx.extras["df_extr"], name="extrativa")
    chi2_top(ctx.df, class_names=class_names)

    # 3. Baselines clássicos
    _section("3. Baselines clássicos (LogReg / SVM / MLP)")
    from fakerecogna2.config import SEEDS_MULTI
    from fakerecogna2.models import (
        make_baseline_configs,
        train_baselines,
        train_baselines_multiseed,
    )
    from fakerecogna2.utils.io_utils import save_table
    configs = make_baseline_configs(
        ctx.X_train_tfidf, ctx.X_test_tfidf,
        ctx.cls_train.numpy(), ctx.cls_test.numpy(),
        seed=ctx.seed,
    )
    trained_baselines, _ = train_baselines(configs, ctx.y_train, ctx.y_test)
    ctx.extras["baseline_configs"] = trained_baselines

    # 3b. Baselines multi-seed (3 sementes) - mean+/-std para sustentar
    # afirmacoes de estabilidade no Cap. 5
    _section("3b. Baselines multi-seed (3 sementes)")
    df_bl_ms = train_baselines_multiseed(
        ctx.X_train_tfidf, ctx.X_test_tfidf,
        ctx.cls_train.numpy(), ctx.cls_test.numpy(),
        ctx.y_train, ctx.y_test,
        seeds=SEEDS_MULTI,
    )
    save_table(df_bl_ms, "baselines_multiseed")

    # Coleta predicoes (e probabilidades, quando disponiveis) de TODOS os
    # baselines em ctx.predictions / ctx.probabilities, para alimentar
    # Bootstrap CI, McNemar, ECE e Matriz de Confusao downstream.
    for name, _, Xte, mdl in trained_baselines:
        ctx.predictions[name] = mdl.predict(Xte)
        if hasattr(mdl, "predict_proba"):
            ctx.probabilities[name] = mdl.predict_proba(Xte)

    # 4. Modelos deep + ensembles
    _section("4. Deep models multi-seed + ensembles")
    train_deep_ensemble(ctx, epochs=args.epochs)

    # Persiste resultados multi-seed (mean ± std por modelo/métrica) num CSV
    # leve. Evita perda desse dado entre re-runs / regerações isoladas.
    from fakerecogna2.utils.io_utils import RESULTS_MULTISEED, save_table
    if RESULTS_MULTISEED:
        ms_rows = []
        for model, metrics in RESULTS_MULTISEED.items():
            row = {"Model": model}
            for metric, (mean, std) in metrics.items():
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
            ms_rows.append(row)
        save_table(pd.DataFrame(ms_rows).round(6), "iid_multiseed")

    # 5. BERTimbau FT (multi-seed se --trans-seeds tem 2+ sementes)
    _section(f"5. BERTimbau Fine-Tuned (seeds={args.trans_seeds})")
    train_bert_classifier(ctx, seeds=args.trans_seeds)

    # Helpers de predict pra etapas seguintes (batched)
    _predict_ens3 = make_predict_ens3(ctx, batch_size=32)
    _predict_bert_ft = make_predict_bert_ft(ctx, batch_size=32)

    # 6. Bootstrap / McNemar / Calibração
    _section("6. Bootstrap CI / McNemar / Calibração")
    from fakerecogna2.evaluation import (
        bootstrap_table,
        calibration_table,
        cross_validate_ensemble,
        plot_reliability,
        save_all_cms,
        save_cm,
    )
    from fakerecogna2.statistics import mcnemar_pairwise_holm

    bootstrap_table(ctx.y_test, ctx.predictions)

    # 6b. Matrizes de confusao IID + metricas por classe (Cap. 5.3)
    # Gera cm_iid_<model>_cm.csv/png e _per_class.csv para os 17 modelos
    save_all_cms(
        ctx.y_test, ctx.predictions, class_names,
        prefix="cm_iid", title_suffix="split IID",
    )

    # McNemar pareado na FAMÍLIA PRÉ-DEFINIDA de 7 configurações (escolha
    # metodológica declarada no Cap. 4 §4.4 da dissertação: uma por
    # representação de interesse; 21 pares). PLMs/WEns/BERT[CLS] ficam fora
    # por definição da família — não mover esta chamada para depois da etapa
    # 12 sem revisar o protocolo declarado.
    mcnemar_set = (
        "CNN", "LSTM", "ConvLSTM",
        "Ens2 (CNN+LSTM)", "Ens3 (CNN+LSTM+ConvLSTM)",
        "BERTimbau FT", "TFIDF+MLP",
    )
    mcnemar_preds = {k: ctx.predictions[k] for k in mcnemar_set if k in ctx.predictions}
    mcnemar_pairwise_holm(ctx.y_test, mcnemar_preds)
    # ECE/Brier ESTENDIDO: aplica calibration_table a TODOS os modelos com
    # probabilidades disponiveis (baselines com predict_proba, deep, ensembles,
    # PLMs). Reliability diagram permanece para os 4 modelos principais para
    # legibilidade visual.
    probs_cal_principais = {
        k: v for k, v in ctx.probabilities.items()
        if k in ("CNN", "LSTM", "Ens3 (CNN+LSTM+ConvLSTM)", "BERTimbau FT")
    }
    plot_reliability(probs_cal_principais, ctx.y_test)
    calibration_table(ctx.probabilities, ctx.y_test)

    # 7. 5-fold cross-validation
    if not args.skip_cv:
        _section("7. 5-fold cross-validation")
        from fakerecogna2.models import TextCNN, TextLSTM
        all_emb = torch.cat([ctx.token_train, ctx.token_val, ctx.token_test], 0)
        all_y = np.concatenate([ctx.y_train, ctx.y_val, ctx.y_test])
        all_texts = list(ctx.X_train_text) + list(ctx.X_val_text) + list(ctx.X_test_text)
        embed_dim = ctx.extras.get("embed_dim", 768)
        num_classes = len(class_names)
        cross_validate_ensemble(
            all_emb, all_y, all_texts,
            model_classes=[
                (TextCNN, dict(embed_dim=embed_dim, num_classes=num_classes, dropout=0.5), "CNN"),
                (TextLSTM, dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
                                num_classes=num_classes, dropout=0.4), "LSTM"),
            ],
            device=ctx.device,
        )

    # 8. Stress tests
    _section("8. Stress tests (NER masking + source split + temporal split)")
    from fakerecogna2.evaluation.stress_tests import (
        build_masked_test_loader,
        compare_split_results,
        evaluate_models_on_loader,
        ner_ablation_table,
    )
    from fakerecogna2.utils.io_utils import RESULTS

    # 8a. NER masking
    extractor = ctx.extras["embedding_extractor"]
    masked_loader = build_masked_test_loader(ctx.X_test_text, ctx.y_test, extractor)
    preds_masked = evaluate_models_on_loader(ctx.models, masked_loader, device=ctx.device)
    preds_orig = {k: ctx.predictions[k] for k in ctx.models}
    ner_ablation_table(ctx.y_test, preds_orig, preds_masked)

    baseline = RESULTS.get("Ens2 (CNN+LSTM)", {})

    # 8b. Split por fonte
    if "splits_source" in ctx.extras:
        from fakerecogna2.features import make_loaders
        from fakerecogna2.models import TextCNN, TextLSTM, train_ensemble_on_variant
        df_src_tr, df_src_te = ctx.extras["splits_source"]
        rng = np.random.RandomState(ctx.seed)
        idx = np.arange(len(df_src_tr))
        rng.shuffle(idx)
        n_val = int(0.1 * len(idx))
        v_idx, t_idx = idx[:n_val], idx[n_val:]
        Xs_tr = df_src_tr.iloc[t_idx]["text_proc"].tolist()
        Xs_vl = df_src_tr.iloc[v_idx]["text_proc"].tolist()
        Xs_te = df_src_te["text_proc"].tolist()
        ys_tr = df_src_tr.iloc[t_idx]["label_enc"].to_numpy()
        ys_vl = df_src_tr.iloc[v_idx]["label_enc"].to_numpy()
        ys_te = df_src_te["label_enc"].to_numpy()
        e_tr = extractor.extract_token_embs(Xs_tr)
        e_vl = extractor.extract_token_embs(Xs_vl)
        e_te = extractor.extract_token_embs(Xs_te)
        ld_tr, ld_vl, ld_te = make_loaders(e_tr, e_vl, e_te, ys_tr, ys_vl, ys_te)
        embed_dim = ctx.extras.get("embed_dim", 768)
        nc = len(class_names)
        src_res = train_ensemble_on_variant(
            TextCNN, TextLSTM,
            dict(embed_dim=embed_dim, num_classes=nc, dropout=0.5),
            dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
                 num_classes=nc, dropout=0.4),
            ld_tr, ld_vl, ld_te, ys_te,
            device=ctx.device, epochs=15, suffix="src", seed=ctx.seed,
        )
        compare_split_results(
            baseline.get("Accuracy", float("nan")), baseline.get("F1", float("nan")),
            src_res["Accuracy"], src_res["F1"],
            baseline_label="Random", variant_label="By-Source",
            save_as="14_source_split",
        )
        # CM do split by-source (Cap. 5.4)
        save_cm(
            ys_te, src_res["y_pred"], model_name="Ens2 (CNN+LSTM)",
            class_names=class_names, save_as_prefix="cm_source",
            title_suffix="split by-source",
        )

    # 8c. Split temporal
    if "splits_temporal" in ctx.extras:
        from fakerecogna2.features import make_loaders
        from fakerecogna2.models import TextCNN, TextLSTM, train_ensemble_on_variant
        df_t_tr, df_t_vl, df_t_te = ctx.extras["splits_temporal"]
        Xt_tr = df_t_tr["text_proc"].tolist()
        Xt_vl = df_t_vl["text_proc"].tolist()
        Xt_te = df_t_te["text_proc"].tolist()
        yt_tr = df_t_tr["label_enc"].to_numpy()
        yt_vl = df_t_vl["label_enc"].to_numpy()
        yt_te = df_t_te["label_enc"].to_numpy()
        e_tr = extractor.extract_token_embs(Xt_tr)
        e_vl = extractor.extract_token_embs(Xt_vl)
        e_te = extractor.extract_token_embs(Xt_te)
        ld_tr, ld_vl, ld_te = make_loaders(e_tr, e_vl, e_te, yt_tr, yt_vl, yt_te)
        embed_dim = ctx.extras.get("embed_dim", 768)
        nc = len(class_names)
        tmp_res = train_ensemble_on_variant(
            TextCNN, TextLSTM,
            dict(embed_dim=embed_dim, num_classes=nc, dropout=0.5),
            dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
                 num_classes=nc, dropout=0.4),
            ld_tr, ld_vl, ld_te, yt_te,
            device=ctx.device, epochs=15, suffix="tmp", seed=ctx.seed,
        )
        compare_split_results(
            baseline.get("Accuracy", float("nan")), baseline.get("F1", float("nan")),
            tmp_res["Accuracy"], tmp_res["F1"],
            baseline_label="Random", variant_label="Temporal",
            save_as="14_temporal_split",
        )
        # CM do split temporal (Cap. 5.4)
        save_cm(
            yt_te, tmp_res["y_pred"], model_name="Ens2 (CNN+LSTM)",
            class_names=class_names, save_as_prefix="cm_temporal",
            title_suffix="split temporal",
        )

    # 8d. Split anti-vies (categoria x periodo)
    # Cap. 6 secao 7.4: 'particionamento que reduza diferencas sistematicas
    # entre classes' por topico e periodo.
    if "category" in ctx.df.columns or "date_parsed" in ctx.df.columns:
        _section("8d. Split anti-vies (categoria x periodo)")
        from fakerecogna2.data import make_anti_bias_splits
        from fakerecogna2.features import make_loaders
        from fakerecogna2.models import TextCNN, TextLSTM, train_ensemble_on_variant
        text_col = "text_proc" if "text_proc" in ctx.df.columns else "text"
        Xab_tr, Xab_vl, Xab_te, yab_tr, yab_vl, yab_te, df_te_meta = (
            make_anti_bias_splits(ctx.df, text_col=text_col, seed=ctx.seed)
        )
        e_tr = extractor.extract_token_embs(Xab_tr)
        e_vl = extractor.extract_token_embs(Xab_vl)
        e_te = extractor.extract_token_embs(Xab_te)
        ld_tr, ld_vl, ld_te = make_loaders(e_tr, e_vl, e_te, yab_tr, yab_vl, yab_te)
        embed_dim = ctx.extras.get("embed_dim", 768)
        nc = len(class_names)
        ab_res = train_ensemble_on_variant(
            TextCNN, TextLSTM,
            dict(embed_dim=embed_dim, num_classes=nc, dropout=0.5),
            dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
                 num_classes=nc, dropout=0.4),
            ld_tr, ld_vl, ld_te, yab_te,
            device=ctx.device, epochs=15, suffix="ab", seed=ctx.seed,
        )
        compare_split_results(
            baseline.get("Accuracy", float("nan")),
            baseline.get("F1", float("nan")),
            ab_res["Accuracy"], ab_res["F1"],
            baseline_label="Random", variant_label="Anti-vies (cat x periodo)",
            save_as="14_anti_bias_split",
        )
        save_cm(
            yab_te, ab_res["y_pred"], model_name="Ens2 (CNN+LSTM)",
            class_names=class_names, save_as_prefix="cm_anti_bias",
            title_suffix="split anti-vies",
        )
        # Erros por categoria/ano no split anti-vies
        from fakerecogna2.evaluation import error_distribution_full
        error_distribution_full(
            df_te_meta, yab_te, ab_res["y_pred"], probs=None,
            model_name="Ens2 anti-vies",
            save_as_prefix="error_distrib_anti_bias",
        )

    ens3_preds = ctx.predictions["Ens3 (CNN+LSTM+ConvLSTM)"]

    # 9. XAI - TP/TN/FP/FN (Secao 5.10 do Cap. 5) + tokens agregados
    if not args.skip_xai:
        _section("9. XAI: LIME + IG + Attention Rollout (TP/TN/FP/FN)")
        from fakerecogna2.explainability import (
            aggregate_lime_tokens_by_tag,
            compare_xai_for_examples, lime_stability,
            run_lime_explanations, select_lime_targets_4cells,
        )
        # 4 celulas x 3 exemplos = 12 (Cap. 5.10: amplia para TP/TN/FP/FN)
        targets = select_lime_targets_4cells(
            ctx.y_test, ens3_preds, class_names,
            positive_class_idx=0, n_per_bucket=3,
        )
        records_df, _ = run_lime_explanations(
            ctx.X_test_text, ctx.y_test, ens3_preds, _predict_ens3,
            class_names, targets=targets,
        )
        # Agregacao de tokens por celula (TP/TN/FP/FN)
        aggregate_lime_tokens_by_tag(records_df, top_n=20)

        compare_xai_for_examples(
            ctx.X_test_text, ctx.y_test, ens3_preds, class_names,
            ctx.bert_clf, ctx.tokenizer, targets=targets, device=ctx.device,
        )
        lime_stability(
            ctx.X_test_text, ctx.y_test, ens3_preds, _predict_ens3, class_names,
        )

    # 10. Cross-dataset (Fake.br + Extrativa como OOD)
    if not skip_cross and "df_fakebr" in ctx.extras:
        _section("10. Cross-dataset (FakeRecogna ↔ Fake.br + Extrativa OOD)")
        from fakerecogna2.evaluation import (
            error_diagnosis_cross_dataset, evaluate_on_external_corpus,
            save_polarity_audit_log,
        )
        from fakerecogna2.utils.io_utils import save_table

        df_fbr = ctx.extras["df_fakebr"]
        encoder = ctx.extras.get("label_encoder")
        df_extr = ctx.extras.get("df_extr")

        results = []
        for name, fn in [("Ens3", _predict_ens3), ("BERTimbau FT", _predict_bert_ft)]:
            results.append(
                evaluate_on_external_corpus(
                    df_fbr, "text", "label", f"{name} → Fake.br",
                    fn, encoder=encoder,
                )
            )
            if df_extr is not None:
                results.append(
                    evaluate_on_external_corpus(
                        df_extr, "text", "label", f"{name} → Extrativa",
                        fn, encoder=encoder,
                    )
                )

        # CM cross-dataset (Cap. 5.6) - antes de remover y_true/y_pred para CSV.
        # Nomes semânticos na convenção canônica (0=real, 1=fake).
        for r in results:
            if "y_true" in r and "y_pred" in r:
                save_cm(
                    r["y_true"], r["y_pred"],
                    model_name=r["model"], class_names=["real", "fake"],
                    save_as_prefix=f"cm_ood_{r['model'].split(' →')[0].lower().replace(' ', '_')}_"
                                   f"{r['model'].split('→')[-1].strip().lower().replace('.', '').replace(' ', '_')}",
                    title_suffix="cross-dataset",
                )

        # Remove y_true/y_pred antes de salvar como CSV (sao arrays)
        results_csv = [
            {k: v for k, v in r.items() if k not in ("y_true", "y_pred")}
            for r in results
        ]
        df_ood = pd.DataFrame(results_csv)
        save_table(df_ood, "17_cross_dataset_ood")

        # Audit trail da salvaguarda de polaridade (Seção 4.7).
        save_polarity_audit_log(results_csv)

        # Delta IID vs OOD
        delta_rows = []
        iid_f1 = {
            "Ens3": RESULTS.get("Ens3 (CNN+LSTM+ConvLSTM)", {}).get("F1"),
            "BERTimbau FT": RESULTS.get("BERTimbau FT", {}).get("F1"),
        }
        for r in results:
            model_short = r["model"].split(" →")[0]
            corpus = r["model"].split("→")[-1].strip()
            f1_iid = iid_f1.get(model_short)
            if f1_iid is not None:
                delta_rows.append({
                    "Model": model_short,
                    "OOD corpus": corpus,
                    "F1 IID": f1_iid,
                    "F1 OOD": r["F1"],
                    "ΔF1 (IID-OOD)": f1_iid - r["F1"],
                })
        if delta_rows:
            save_table(pd.DataFrame(delta_rows).round(4), "17_cross_dataset_delta")

        # Diagnóstico de erros (BERT FT em Fake.br) — mesmo pré-processamento
        # e mesma convenção de rótulos (0=real, 1=fake) da avaliação oficial.
        from fakerecogna2.preprocessing.text_cleaning import preprocess_base
        texts_fbr = df_fbr["text"].astype(str).apply(preprocess_base).tolist()
        probs_fbr = _predict_bert_ft(texts_fbr)
        y_fbr = np.array([{"real": 0, "fake": 1}.get(str(l).lower(), -1) for l in df_fbr["label"]])
        mask = y_fbr >= 0
        error_diagnosis_cross_dataset(
            probs_fbr[mask], y_fbr[mask], model_name="BERTimbau FT (Fake.br)",
        )

        # 10b. Cross-dataset INVERSO: treina em Fake.br, avalia em FakeRecogna
        # (Cap. 4.E - cross-dataset bidirecional)
        from fakerecogna2.evaluation import train_on_fakebr_eval_main
        train_on_fakebr_eval_main(
            df_fakebr=df_fbr,
            X_main_test=ctx.X_test_text, y_main_test=ctx.y_test,
            extractor=ctx.extras["embedding_extractor"],
            tokenizer=ctx.tokenizer, bert_model=ctx.bert_model,
            device=ctx.device, seed=ctx.seed,
        )

    # 11. Adversarial
    if not skip_adv:
        _section("11. Robustez adversarial")
        from fakerecogna2.adversarial import run_adversarial_eval
        run_adversarial_eval(
            ctx.X_test_text, ctx.y_test,
            models={"Ens3": _predict_ens3, "BERT FT": _predict_bert_ft},
            do_back_translation=not args.skip_bt,
            device=ctx.device,
        )

    # 12. PLMs maiores (multi-seed se --trans-seeds tem 2+ sementes)
    if not skip_plm:
        _section(f"12. PLMs maiores (XLM-R, BERTimbau-large, mDeBERTa) seeds={args.trans_seeds}")
        from fakerecogna2.models import evaluate_plm_candidates
        plm_results = evaluate_plm_candidates(
            X_train=ctx.X_train_text, y_train=ctx.y_train,
            X_val=ctx.X_val_text, y_val=ctx.y_val,
            X_test=ctx.X_test_text, y_test=ctx.y_test,
            num_classes=len(class_names),
            max_seq_len=ctx.max_seq_len, device=ctx.device,
            seeds=args.trans_seeds,
        )
        # Cole predicoes/probabilidades dos PLMs (alimenta Bootstrap+ECE+CM).
        for short, res in plm_results.items():
            if "y_pred" in res:
                ctx.predictions[f"PLM: {short}"] = res["y_pred"]
            if "probs" in res:
                ctx.probabilities[f"PLM: {short}"] = res["probs"]

        # 12b. Re-executa Bootstrap CI + ECE + CM com TODOS os 17 modelos
        # (etapa 6 capturou 14; os 3 PLMs entram agora). Sobrescreve os CSVs.
        _section("12b. Bootstrap CI + ECE + CM - 17 modelos completos")
        bootstrap_table(ctx.y_test, ctx.predictions)
        calibration_table(ctx.probabilities, ctx.y_test)
        save_all_cms(
            ctx.y_test, ctx.predictions, class_names,
            prefix="cm_iid", title_suffix="split IID",
        )

    # 12c. Analise de erros estratificada (fonte/categoria/ano/confianca)
    # Cap. 5.9: 'distribuicao de erros por classe, fonte, ano, comprimento
    # e confianca preditiva'.
    if "df_test_meta" in ctx.extras:
        _section("12c. Analise de erros por fonte/categoria/ano/confianca")
        from fakerecogna2.evaluation import error_distribution_full
        ens3_probs = ctx.probabilities.get("Ens3 (CNN+LSTM+ConvLSTM)")
        error_distribution_full(
            ctx.extras["df_test_meta"], ctx.y_test, ens3_preds,
            probs=ens3_probs, model_name="Ens3 (CNN+LSTM+ConvLSTM)",
            save_as_prefix="error_distrib_ens3",
        )
        bert_preds = ctx.predictions.get("BERTimbau FT")
        bert_probs = ctx.probabilities.get("BERTimbau FT")
        if bert_preds is not None:
            error_distribution_full(
                ctx.extras["df_test_meta"], ctx.y_test, bert_preds,
                probs=bert_probs, model_name="BERTimbau FT",
                save_as_prefix="error_distrib_bert_ft",
            )

    # 12d. Integracao XAI <-> erros (item 60 do STATUS.md)
    # Cruza TP/TN/FP/FN com categoria/fonte/ano + tokens LIME agregados.
    if "df_test_meta" in ctx.extras and not args.skip_xai:
        _section("12d. Integracao XAI <-> erros")
        from fakerecogna2.evaluation import consolidate_xai_errors
        # records_df foi gerado em etapa 9 (XAI); recarrega do CSV salvo
        import pandas as pd
        lime_csv = ROOT / "outputs" / "metrics" / "15_lime_records.csv"
        lime_df = pd.read_csv(lime_csv) if lime_csv.exists() else None
        consolidate_xai_errors(
            y_true=ctx.y_test,
            y_pred=ens3_preds,
            probs=ctx.probabilities.get("Ens3 (CNN+LSTM+ConvLSTM)"),
            df_meta=ctx.extras["df_test_meta"],
            lime_records_df=lime_df,
            model_name="Ens3 (CNN+LSTM+ConvLSTM)",
            positive_class_idx=0,
            save_as_prefix="xai_err",
        )

    # 13. Paraphrasing Equalizer (Seção 21)
    if not skip_paraphr:
        _section("13. Paraphrasing Equalizer")
        from fakerecogna2.data.splits import make_random_splits
        from fakerecogna2.features import make_loaders
        from fakerecogna2.models import TextCNN, TextLSTM, train_ensemble_on_variant
        from fakerecogna2.preprocessing import build_equalized_dataset
        from fakerecogna2.utils.io_utils import save_table

        df_eq = build_equalized_dataset(ctx.df)
        Xeq_tr, Xeq_vl, Xeq_te, yeq_tr, yeq_vl, yeq_te = make_random_splits(
            df_eq, text_col="text_eq_proc", seed=ctx.seed,
        )
        e_tr = extractor.extract_token_embs(Xeq_tr)
        e_vl = extractor.extract_token_embs(Xeq_vl)
        e_te = extractor.extract_token_embs(Xeq_te)
        ld_tr, ld_vl, ld_te = make_loaders(e_tr, e_vl, e_te, yeq_tr, yeq_vl, yeq_te)
        embed_dim = ctx.extras.get("embed_dim", 768)
        nc = len(class_names)
        eq_res = train_ensemble_on_variant(
            TextCNN, TextLSTM,
            dict(embed_dim=embed_dim, num_classes=nc, dropout=0.5),
            dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
                 num_classes=nc, dropout=0.4),
            ld_tr, ld_vl, ld_te, yeq_te,
            device=ctx.device, epochs=15, suffix="eq", seed=ctx.seed,
        )
        df_eq_cmp = pd.DataFrame([
            {"Setup": "ORIGINAL (fake cru, real sumarizado)",
             "Accuracy": baseline.get("Accuracy"), "F1": baseline.get("F1")},
            {"Setup": "EQUALIZADO (ambos sumarizados)",
             "Accuracy": eq_res["Accuracy"], "F1": eq_res["F1"]},
        ]).round(4)
        df_eq_cmp["Δ F1"] = df_eq_cmp["F1"].diff().fillna(0).round(4)
        save_table(df_eq_cmp, "21_paraphrasing_equalizer")

    # 14. Deployment
    _section("14. Deployment (latência, VRAM, disco, Pareto, auditoria params)")
    from fakerecogna2.deployment import (
        audit_model_parameters,
        benchmark_models, count_parameters, measure_disk_sizes,
        plot_pareto_f1_latency,
    )
    predict_fns = {
        "BERTimbau FT": _predict_bert_ft,
        "Ens3 (CNN+LSTM+ConvLSTM)": _predict_ens3,
    }
    extras_bench = {
        "BERTimbau FT": {
            "Params (M)": count_parameters(ctx.bert_clf) / 1e6,
            "F1": RESULTS.get("BERTimbau FT", {}).get("F1"),
        },
        "Ens3 (CNN+LSTM+ConvLSTM)": {
            "Params (M)": sum(
                count_parameters(ctx.models[k]) for k in ("CNN", "LSTM", "ConvLSTM")
            ) / 1e6,
            "F1": RESULTS.get("Ens3 (CNN+LSTM+ConvLSTM)", {}).get("F1"),
        },
    }
    df_bench = benchmark_models(
        ctx.X_test_text[:200], predict_fns, extras=extras_bench, device=ctx.device,
    )
    plot_pareto_f1_latency(df_bench)
    measure_disk_sizes({
        "BERTimbau FT": ctx.bert_clf,
        "Ens3 (CNN+LSTM+ConvLSTM)": list(ctx.models.values()),
    })

    # 14b. Auditoria de parametros para todos os modelos disponiveis
    # (Cap. 5.11 - auditar coluna de parametros na versao final)
    audit_models = {
        "CNN": ctx.models.get("CNN"),
        "LSTM": ctx.models.get("LSTM"),
        "ConvLSTM": ctx.models.get("ConvLSTM"),
        "Ens3 (CNN+LSTM+ConvLSTM)": list(ctx.models.values()),
        "BERTimbau FT": ctx.bert_clf,
    }
    # Adiciona baselines treinados (sklearn) — toma o objeto modelo da tupla
    for name, _, _, mdl in trained_baselines:
        audit_models[name] = mdl
    audit_models = {k: v for k, v in audit_models.items() if v is not None}
    audit_model_parameters(audit_models)

    # 15. Ablações
    if not skip_ablations:
        _section("15. Ablações (A: preproc, B: seqlen, C: extrativa, D: learning curve, E: by length, Q: quartil)")
        from fakerecogna2.evaluation import (
            learning_curve_ablation,
            performance_by_length_ensemble,
            preprocessing_ablation,
            seqlen_ablation,
            short_quartile_class_distribution,
        )
        preprocessing_ablation(ctx.df, n_samples=15000, seed=ctx.seed)
        seqlen_ablation(ctx, seq_len=300)
        learning_curve_ablation(ctx)
        performance_by_length_ensemble(ctx, _predict_ens3, model_name="Ens3")
        short_quartile_class_distribution(ctx.X_test_text, ctx.y_test)

        # Abstrativa vs Extrativa (ablação C)
        if "splits_extrativa" in ctx.extras:
            from fakerecogna2.features import make_loaders
            from fakerecogna2.models import TextCNN, TextLSTM, train_ensemble_on_variant
            from fakerecogna2.utils.io_utils import save_table
            se = ctx.extras["splits_extrativa"]
            e_tr = extractor.extract_token_embs(se["X_train"])
            e_vl = extractor.extract_token_embs(se["X_val"])
            e_te = extractor.extract_token_embs(se["X_test"])
            ld_tr, ld_vl, ld_te = make_loaders(
                e_tr, e_vl, e_te, se["y_train"], se["y_val"], se["y_test"],
            )
            embed_dim = ctx.extras.get("embed_dim", 768)
            nc = len(class_names)
            ext_res = train_ensemble_on_variant(
                TextCNN, TextLSTM,
                dict(embed_dim=embed_dim, num_classes=nc, dropout=0.5),
                dict(embed_dim=embed_dim, hidden_dim=128, num_layers=2,
                     num_classes=nc, dropout=0.4),
                ld_tr, ld_vl, ld_te, se["y_test"],
                device=ctx.device, epochs=15, suffix="ext", seed=ctx.seed,
            )
            df_ext = pd.DataFrame([
                {"Versão": "Abstrativa",
                 "Ens2 Acc": baseline.get("Accuracy"), "Ens2 F1": baseline.get("F1")},
                {"Versão": "Extrativa",
                 "Ens2 Acc": ext_res["Accuracy"], "Ens2 F1": ext_res["F1"]},
            ]).round(4)
            save_table(df_ext, "abstrativa_vs_extrativa")

    # 16. Tabela final + Classification Report + Relatórios MD/JSON
    _section("16. Tabela final + Classification Report + Relatórios MD/JSON")
    from fakerecogna2.evaluation import plot_final_comparison
    from fakerecogna2.reports import (
        build_consolidated_report,
        build_final_table,
        build_manifest,
        build_results_json,
        print_classification_report,
    )

    build_final_table()
    plot_final_comparison()
    print_classification_report(ctx.y_test, ens3_preds, class_names, "Ens3")

    build_manifest(
        n_samples=len(ctx.df),
        class_names=class_names,
        device=ctx.device,
    )
    build_consolidated_report(
        df_abst=ctx.df, df_extr=ctx.extras.get("df_extr"),
        X_train=ctx.X_train_text, X_val=ctx.X_val_text, X_test=ctx.X_test_text,
        class_names=class_names, device=ctx.device,
    )
    build_results_json(
        df_abst=ctx.df, df_extr=ctx.extras.get("df_extr"),
        X_train=ctx.X_train_text, X_val=ctx.X_val_text, X_test=ctx.X_test_text,
        class_names=class_names, device=ctx.device,
    )

    elapsed = time.time() - t0_total
    _section(f"PIPELINE COMPLETO em {elapsed/60:.1f} min")
    print(f"  Relatório: outputs/relatorio_final.md")
    print(f"  JSON:      outputs/resultados.json")
    print(f"  Métricas:  outputs/metrics/")
    print(f"  Figuras:   outputs/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
