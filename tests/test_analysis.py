"""Aggregation logic behind the report tables (src/analysis.py)."""
import pandas as pd
import pytest

from src.analysis import baseline_row, default_mask, final_full_evals, mean_std_table


def _row(run, model, seed, epoch, quick=False, fid=10.0, beta=1.0, latent_dim=32):
    return {"run": run, "model": model, "seed": seed, "epoch": epoch, "quick": quick,
            "fid_classifier": fid, "beta": beta if model == "vae" else None, "latent_dim": latent_dim}


def test_final_eval_is_the_last_full_one():
    df = pd.DataFrame([
        _row("vae_s0", "vae", 0, 25, fid=30.0),
        _row("vae_s0", "vae", 0, 30, fid=20.0),
        _row("vae_s0", "vae", 0, 30, quick=True, fid=99.0),  # quick evals never enter the tables
        _row("gan_s0", "gan", 0, 30, fid=15.0),
    ])
    final = final_full_evals(df).set_index("run")
    assert final.loc["vae_s0", "fid_classifier"] == 20.0
    assert final.loc["gan_s0", "fid_classifier"] == 15.0
    assert not final["quick"].any()


def test_default_mask_selects_the_main_comparison():
    df = pd.DataFrame([
        _row("a", "vae", 0, 30), _row("b", "vae", 0, 30, beta=4.0),
        _row("c", "gan", 0, 30), _row("d", "gan", 0, 30, latent_dim=8),
    ])
    assert df[default_mask(df, latent_dim=32, beta=1.0)]["run"].tolist() == ["a", "c"]


def test_table_reports_mean_and_sample_std_over_seeds():
    df = pd.DataFrame([_row(f"vae_s{s}", "vae", s, 30, fid=f) for s, f in enumerate([10.0, 12.0, 14.0])]
                      + [_row("gan_s0", "gan", 0, 30, fid=5.0)])
    table = mean_std_table(df, ["model"], ["fid_classifier"]).set_index("model")
    assert table.loc["VAE", "FID-classifier ↓"] == "12 ± 2"  # std with ddof=1
    assert table.loc["VAE", "seeds"] == 3
    assert table.loc["DCGAN", "FID-classifier ↓"] == "5"     # one seed: no ± shown


def test_baseline_row_marks_missing_metrics():
    row = baseline_row({"fid_classifier": 0.0339}, ["fid_classifier", "recon_psnr"])
    assert row["model"].startswith("Real data")
    assert row["FID-classifier ↓"] == "0.0339"
    assert row["Recon PSNR ↑"] == "–"  # not defined for real data


@pytest.mark.parametrize("missing", [[], ["fid_classifier"]])
def test_table_handles_missing_columns(missing):
    df = pd.DataFrame([_row("vae_s0", "vae", 0, 30)]).drop(columns=missing)
    table = mean_std_table(df, ["model"], ["fid_classifier"])
    assert len(table) == 1
