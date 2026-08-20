"""Build attribution_methods.csv — long format, one row per (run_id, method, variant).

experiments.csv carries the three headline metrics (metasmoothness, MAGIC LDS, EK-FAC LDS) as
columns. Everything else measured on the same banks — Shampoo powers, SOURCE variants, TrackStar,
and the non-gradient baselines — lives here so the wide table stays plottable. `run_id` joins to
experiments.csv.

NB `apply_power` is TWICE the "Shampoo -1/N" label used here: -1.0 / -0.5 / -0.25 in the yaml
correspond to -1/2 / -1/4 / -1/8 in this table.
"""

import csv
import os

COLUMNS = ["run_id", "method", "variant", "lds", "ci_lo", "ci_hi", "n_subsets",
           "n_queries", "run_dir", "source_doc", "notes"]

R = []


def add(run_id: str, method: str, variant: str, lds: float | str, lo: float | str = "",
        hi: float | str = "", nsub: int | str = 50, nq: int | str = "", run_dir: str = "",
        doc: str = "LDS_RESULTS.md", notes: str = ""):
    R.append(dict(run_id=run_id, method=method, variant=variant, lds=lds, ci_lo=lo, ci_hi=hi,
                  n_subsets=nsub, n_queries=nq, run_dir=run_dir, source_doc=doc, notes=notes))


# --- Shampoo powers on the SmolLM2 GPT-2 banks ---
for rid, var, lds, lo, hi in [
        ("sm_adam_eps1e6_4k", "-1/2", 0.3071, 0.270, 0.342),
        ("sm_adam_eps1e6_4k", "-1/4", 0.3264, 0.294, 0.358),
        ("sm_adam_eps0_4k", "-1/2", 0.2145, 0.178, 0.249),
        ("sm_adam_eps0_4k", "-1/4", 0.1562, 0.122, 0.192),
        ("sm_adam_eps0_4k", "-1/8", 0.1111, 0.076, 0.145),
        ("sm_muon_eps1e6_5e5_4k", "-1/2", 0.5217, 0.481, 0.561),
        ("sm_muon_eps1e6_5e5_4k", "-1/4", 0.4304, 0.388, 0.471),
        ("sm_muon_eps1e6_5e5_4k", "-1/8", 0.3026, 0.260, 0.343),
        ("sm_muon_eps0_5e5_4k", "-1/2", 0.5208, 0.479, 0.561),
        ("sm_muon_eps0_5e5_4k", "-1/4", 0.4306, 0.389, 0.471),
        ("sm_muon_eps0_5e5_4k", "-1/8", 0.3010, 0.260, 0.343),
        ("sm_muon_eps0_1e4_4k", "-1/2", 0.5206, 0.479, 0.560),
        ("sm_muon_eps0_1e4_4k", "-1/4", 0.4093, 0.371, 0.447),
        ("sm_muon_eps0_1e4_4k", "-1/8", 0.2682, 0.229, 0.307)]:
    add(rid, "shampoo", var, lds, lo, hi, notes="apply_power = 2x this label.")

# --- WikiText lotus bank: every scorer run against it ---
for meth, var, lds, d in [
        ("magic", "bwd eval", 0.9688, "runs/lotus_bwd_eval"),
        ("source", "damp0", 0.3902, "runs/lotus_source_q50_damp0_validate"),
        ("source", "adam", 0.2068, "runs/lotus_source_adam_q50_validate"),
        ("source", "default", -0.3871, "runs/lotus_source_q50_validate"),
        ("ekfac", "docspace", 0.2588, "runs/lotus_ekfac50q_docspace_vs_lotus_bank"),
        ("ekfac", "allium-0", 0.0543, "runs/lotus_scores_ekfac50q_allium-0_validate"),
        ("trackstar", "docs p32 noopt", 0.2002, "runs/gpt2_lotus_trackstar50q_docs_p32_noopt_vs_lotus_bank"),
        ("trackstar", "docs", 0.1838, "runs/gpt2_lotus_trackstar50q_docs_vs_lotus_bank"),
        ("trackstar", "default", 0.1767, "runs/lotus_trackstar_q50_validate")]:
    add("wt_adam_eps1e6_bs64_lotus", meth, var, lds, nsub=400, nq=50, run_dir=d)

# --- WikiText eps_root=0 bank ---
for meth, var, lds, d in [
        ("source", "source2", 0.1531, "runs/epsroot0_source2_q50_validate"),
        ("source", "source2 adam hybrid", 0.1446, "runs/epsroot0_source2_adam_hybrid_validate"),
        ("source", "source2 adam", 0.0811, "runs/epsroot0_source2_adam_q50_validate"),
        ("ekfac", "allium-0", -0.0109, "runs/epsroot0_scores_ekfac50q_allium-0_validate")]:
    add("wt_adam_eps0_bs64", meth, var, lds, nq=50, run_dir=d)
add("wt_adam_eps0_bs64", "magic", "spotcheck", "", nq=1, run_dir="runs/epsroot0_bank",
    notes="Returned NaN.")

# --- BASELINE_LDS.md: non-gradient and cheap baselines, 100-subset WikiText banks ---
BANKS = {"adamw_8e-4": "wt_adamw_eps1e8_lr0.0008_bank100",
         "muon_8e-4": "wt_muon_eps1e8_lr0.0008_bank100",
         "muon_2e-4": "wt_muon_eps1e8_lr0.0002_bank100",
         "muon_8e-5": "wt_muon_eps1e8_lr8e-05_bank100"}
BASE = {
    "trackstar_no_adam_norm": (0.1764, 0.0374, 0.1680, 0.2395),
    "bm25_lexical":           (0.16,   0.0625, 0.1821, 0.2552),
    "activation_similarity":  (0.09,   0.0690, 0.0956, 0.1220),
    "qwen3_embedding_8b":     (0.11,   0.0536, 0.1036, 0.1590),
    "jina_v3_semantic":       (0.06,   "",     0.0881, 0.0945),
    "gradient_cosine":        (0.05,   0.0038, 0.0298, 0.0505),
}
for meth, vals in BASE.items():
    for (bank, rid), v in zip(BANKS.items(), vals):
        add(rid, meth, "", v, nsub=100, nq=50, doc="BASELINE_LDS.md",
            notes="jina FAILED on this bank." if (meth == "jina_v3_semantic" and v == "") else "")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attribution_methods.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(R)
    print(f"wrote {out}: {len(R)} rows, {len({r['method'] for r in R})} methods")


if __name__ == "__main__":
    main()
