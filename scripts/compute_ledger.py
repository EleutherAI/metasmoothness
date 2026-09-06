#!/usr/bin/env python3
"""GPU-hour accounting for the paper: per stage, per run, per figure.

    python scripts/compute_ledger.py            # print the ledger from the cached scan
    python scripts/compute_ledger.py --scan     # re-walk the run trees and logs first
    python scripts/compute_ledger.py --write    # also refresh data/compute_ledger.csv

Figure membership is derived from scripts/scaling_plot_mpl.py's own axis
constants, the way scripts/which_figure.py does it, so the ledger cannot drift
from what is actually drawn.

Two things this script is careful about, because both were wrong on the first
pass:

  * **Retrains are counted from artifacts, not from launches.** The unit is a
    subset index in `validation*.csv` or a query index in `filter_proponents.csv`,
    unioned across shards and deduplicated against merge copies (`.merge_tmp`,
    `.partial*`) and relaunch races (`_irisrace`, `_tailrace`, ...). So the totals
    are *retained* work; anything redone after a crash or a resume shows up only
    in the discarded column of COMPUTE.md, never here.
  * **Both roots.** Rows live under ssd-2 and, for anything not yet migrated,
    ssd-1 -- the same two roots the plot scripts read. Counting only ssd-2 loses
    the 16k bs512 row and both 32k bs32 rows. A run present in both is one run,
    mirrored, not two (D23: ssd-1 is read-only, it is the older location).

Costs come from completed tqdm bars in the run logs. Every rank writes its own
bar to the same log, so a naive "elapsed went down, that's a new pass" reset
detector counts each pass once per rank and inflates the total by ~17x; the scan
below keys on 100%-complete bars only and collapses consecutive duplicate lines.
Where a row's logs did not survive, the cost law supplies the number, and the
`src` column says which.
"""
import argparse, csv, os, pathlib, re, sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROOTS = ["/mnt/ssd-2/lucia/paper_runs/experiments",
         "/mnt/ssd-1/lucia/paper_runs/experiments"]
LOG_TREE = "/mnt/ssd-2/lucia/paper_runs"          # ssd-1 keeps no usable logs
INVENTORY = ROOT / "data" / "retrain_inventory.csv"
TIMINGS = ROOT / "data" / "log_timings.csv"
LEDGER = ROOT / "data" / "compute_ledger.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--scan", action="store_true",
                help="re-walk the run trees and ~1 GB of logs (slow; several minutes)")
ap.add_argument("--write", action="store_true", help="refresh data/compute_ledger.csv")
args = ap.parse_args()

# ---------------------------------------------------------------- cost law ---
# Measured: one 2-epoch retrain costs a fixed number of wall-seconds per training
# document at nproc 2, and it held independently at 4k/8k/16k/32k/64k/128k/256k.
# The fleet is mixed, so this is a fleet average -- an A100 row runs ~2x faster
# than an A40 row at the same N, which is why measured per-retrain costs at one
# corpus size are not identical across rows.
NPROC = 2                       # world size for essentially every run
SEC_PER_DOC = {"gpt2": 0.0347,  # gpt2-124M
               "med": 0.0444,   # gpt2-medium-355M, from its own measured rows
               "qwen": 0.1007}  # Qwen-1.5B, from the qwen 16k/32k filter bars
# MAGIC is one reverse pass per query over the whole corpus, strictly serial.
MAGIC_SEC_PER_DOC = {"gpt2": 0.245, "med": 0.51, "qwen": 0.71}
MAGIC_SMALL_BATCH = 1.4         # bs<=32 rows measured ~1.4x the bs256 rate
BANK_SUBSETS = 100              # an LDS is 100 leave-1%-out retrains plus the base
FILTER_CONTROLS = 3             # per ROW, shared across variants via retrained_dir


def fam(run):
    return "qwen" if run.startswith("qwen") else ("med" if "medium" in run else "gpt2")


# ------------------------------------------------------ scan: run inventory ---
SHARD = re.compile(r"_q\d+_\d+(?:_[a-z0-9]+)?$")
MERGE = re.compile(r"\.(merge_tmp|partial[\w]*|nan_seed\d+)$")
RACE = re.compile(r"_(irisrace|tailrace|bell01|shared\d+)$")


def scan_inventory():
    """(run, bank_subsets, [(campaign, n_query_retrains)]) over both roots."""
    out = {}
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for run in sorted(os.listdir(root)):
            p = os.path.join(root, run)
            if not os.path.isdir(p):
                continue
            subs = set()
            for f in sorted(os.listdir(p)):
                if not f.startswith("validation") or ".csv" not in f:
                    continue
                try:
                    for r in csv.DictReader(open(os.path.join(p, f))):
                        if r.get("subset") is not None:
                            subs.add(r["subset"])
                except OSError:
                    pass
            camp = {}
            for d in sorted(os.listdir(p)):
                fd = os.path.join(p, d)
                if not d.startswith("filter_") or not os.path.isdir(fd):
                    continue
                if MERGE.search(d):          # a merge copy re-lists finished work
                    continue
                stem = RACE.sub("", d)
                variant, sharded = SHARD.sub("", stem), bool(SHARD.search(stem))
                fp = os.path.join(fd, "filter_proponents.csv")
                if not os.path.isfile(fp):
                    continue
                try:
                    nq = len({r["query"] for r in csv.DictReader(open(fp))})
                except OSError:
                    continue
                if not nq:
                    continue
                c = camp.setdefault(variant, {"parent": 0, "shards": 0, "n": 0})
                # Shard query indices are LOCAL to the shard, so they cannot be
                # unioned; sum the shards instead and cap at the 20-query plan.
                if sharded:
                    c["shards"] += nq
                    c["n"] += 1
                else:
                    c["parent"] = max(c["parent"], nq)
            camps = {v: (min(20, c["shards"]) if c["n"] else c["parent"])
                     for v, c in camp.items()}
            if not (subs or camps):
                continue
            if run in out:                   # mirrored across roots: one run
                out[run]["bank"] = max(out[run]["bank"], len(subs))
                for v, n in camps.items():
                    out[run]["camps"][v] = max(out[run]["camps"].get(v, 0), n)
            else:
                out[run] = {"bank": len(subs), "camps": camps}
    rows = [{"run_id": r, "campaign": "", "bank_subsets": v["bank"], "retrains": ""}
            for r, v in sorted(out.items())]
    rows += [{"run_id": r, "campaign": c, "bank_subsets": "", "retrains": n}
             for r, v in sorted(out.items()) for c, n in sorted(v["camps"].items())]
    return rows


# ----------------------------------------------------------- scan: timings ---
# "Phase: 100%|####| 125/125 [1:04:07<00:00, 30.78s/it]". The `done == total`
# check below is load-bearing: tqdm rounds the percentage up, so a 128000-step
# bar prints "100%" from step 127360 onwards. Keying on the printed percentage
# alone turns the last few hundred frames of every bar into separate passes and
# inflates a 11-hour K-FAC fit into 9000 hours.
BAR = re.compile(r"([^\r\n|]*?):\s*100%\|[^|]*\|\s*(\d+)/(\d+)\s*\[(\d+:\d+(?::\d+)?)<")
PHASES = {"Validating", "filter-proponents", "Backward", "Training", "random",
          "Computing New worker - Collecting gradients", "Evaluating bank",
          "Computing Approximating Hessians with kfac",
          "Computing Approximating Hessians with kfac (eigenvalue correction)"}


def _secs(t):
    p = [int(x) for x in t.split(":")]
    return (p[0] * 3600 + p[1] * 60 + p[2]) if len(p) == 3 else p[0] * 60 + p[1]


def scan_timings():
    agg = defaultdict(lambda: [0.0, 0, 0])       # (log, phase) -> [sec, iters, passes]
    for dp, _, fns in os.walk(LOG_TREE):
        for fn in fns:
            if not fn.endswith(".log"):
                continue
            fp = os.path.join(dp, fn)
            try:
                if os.path.getsize(fp) == 0:
                    continue
            except OSError:
                continue
            rel, last = os.path.relpath(fp, LOG_TREE), None
            try:
                with open(fp, errors="replace", newline="") as f:
                    for chunk in f:
                        for line in chunk.replace("\r", "\n").split("\n"):
                            if "100%|" not in line:
                                continue
                            m = BAR.search(line)
                            if not m or m.group(1).strip() not in PHASES:
                                continue
                            if m.group(2) != m.group(3):     # rounded up, not done
                                continue
                            desc, n = m.group(1).strip(), int(m.group(3))
                            el = _secs(m.group(4))
                            key = (desc, n, el)
                            if key == last:      # same bar re-printed by another rank
                                continue
                            last = key
                            a = agg[(rel, desc)]
                            a[0] += el
                            a[1] += n
                            a[2] += 1
            except OSError:
                continue
    return [{"log": lg, "phase": ph, "seconds": round(v[0]), "iters": v[1], "passes": v[2]}
            for (lg, ph), v in sorted(agg.items())]


def _write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", file=sys.stderr)


if args.scan:
    _write(INVENTORY, scan_inventory(),
           ["run_id", "campaign", "bank_subsets", "retrains"])
    _write(TIMINGS, scan_timings(), ["log", "phase", "seconds", "iters", "passes"])

for p in (INVENTORY, TIMINGS):
    if not p.is_file():
        sys.exit(f"{p} missing -- run with --scan first")

# ------------------------------------------------------------- load inputs ---
rows = list(csv.DictReader(open(ROOT / "experiments.csv")))
by_id = {r["run_id"]: r for r in rows}
bank_of = defaultdict(int)                 # run -> retrains in its LDS bank
camps_of = defaultdict(dict)               # run -> {campaign: query retrains}
for r in csv.DictReader(open(INVENTORY)):
    if r["campaign"]:
        camps_of[r["run_id"]][r["campaign"]] = int(r["retrains"])
    else:
        bank_of[r["run_id"]] = int(r["bank_subsets"])
timings = list(csv.DictReader(open(TIMINGS)))


def n_docs(run):
    r = by_id.get(run)
    if r:
        return int(float(r["n_docs"]))
    m = re.search(r"(\d+)k", run)          # qwen / london rows are not in the csv
    return int(m.group(1)) * 1000 if m else 16000


# -------------------------------------------------- measured per-run costs ---
def _run_of(log):
    """The run a log belongs to, or None for the flat _logs/<tag>.log pool."""
    p = log.split("/")
    if p[0] == "experiments" and len(p) > 2 and not p[1].startswith("_"):
        return p[1]
    return None


retrain_sec, magic_sec = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
for t in timings:
    run, ph = _run_of(t["log"]), t["phase"]
    if run is None:
        continue
    if ph in ("Validating", "filter-proponents"):
        retrain_sec[run][0] += float(t["seconds"])
        retrain_sec[run][1] += int(t["iters"])
    elif ph == "Backward":                 # one bar per MAGIC query
        magic_sec[run][0] += float(t["seconds"])
        magic_sec[run][1] += int(t["passes"])


def retrain_h(run):
    """GPU-hours for one retrain on this row, and where the number came from."""
    s, n = retrain_sec.get(run, [0, 0])
    if n >= 15:                            # enough iterations to trust the rate
        return s / n * NPROC / 3600, "meas"
    return n_docs(run) * SEC_PER_DOC[fam(run)] * NPROC / 3600, "mod"


def magic_h(run):
    """GPU-hours to score all 20 queries with MAGIC."""
    s, n = magic_sec.get(run, [0, 0])
    if n >= 5:
        return 20 * s / n * NPROC / 3600
    bs = int(by_id.get(run, {}).get("batch_size") or 256)
    mult = MAGIC_SMALL_BATCH if bs <= 32 else 1.0
    return 20 * n_docs(run) * MAGIC_SEC_PER_DOC[fam(run)] * mult * NPROC / 3600


# EK-FAC (gradient collection, K-FAC fit, eigenvalue correction, scoring) is
# taken as measured -- there is no clean per-document law for it. Its logs live
# in the flat pool under names that only loosely track run ids, and the qwen fits
# ran on four GPUs, so world size has to come off the log name.
EKFAC_PHASES = {"Computing New worker - Collecting gradients", "Evaluating bank",
                "Computing Approximating Hessians with kfac",
                "Computing Approximating Hessians with kfac (eigenvalue correction)"}


def _ekfac_target(log):
    run = _run_of(log)
    tag = log.split("/")[-1][:-4]
    nproc = 4 if re.search(r"nproc4|g0_3|_0123|_4567|4shard", tag) else NPROC
    if run:
        return run, NPROC
    for pat, sub in ((r"(?:qwen15b?|q15b|qwen15)_(\d+)k", r"qwen15b_\1k_bs256"),
                     (r"^muon512", "plan_muon_eps1e17_512k_bs256"),
                     (r"^muon256", "plan_muon_eps1e17_256k_bs256"),
                     (r"^ekfac_512k_bs256$", "plan_adam_eps1e17_512k_bs256"),
                     (r"^heldout_ekfac_(\d+)k", r"plan_adam_eps1e17_\1k_bs256"),
                     (r"^(plan_(?:adam|muon)_eps1e17_[\w.]+?)_ekfac", r"\1")):
        m = re.match(pat, tag)
        if m:
            return re.sub(pat, sub, m.group(0)), nproc
    return None, nproc


ekfac_h, ekfac_unattributed = defaultdict(float), 0.0
for t in timings:
    if t["phase"] not in EKFAC_PHASES:
        continue
    run, nproc = _ekfac_target(t["log"])
    if run:
        ekfac_h[run] += float(t["seconds"]) / 3600 * nproc
    else:
        ekfac_unattributed += float(t["seconds"]) / 3600 * NPROC

# --------------------------------------------------------------- per run -----
R = {}
for run in sorted(set(bank_of) | set(camps_of) | set(ekfac_h)):
    rh, src = retrain_h(run)
    nb = bank_of[run]
    cs = camps_of[run]
    # The controls are 3 trained models for the whole row. A row that already
    # has a >=100-subset bank reuses it via retrained_dir and adds none.
    ctrl = 0 if nb >= BANK_SUBSETS else FILTER_CONTROLS * len(cs)
    has_magic = bool((by_id.get(run, {}).get("magic_lds") or "").strip()
                     or (by_id.get(run, {}).get("filter_magic_delta") or "").strip())
    qld = {c: (n + (ctrl / len(cs) if cs else 0)) * rh for c, n in cs.items()}
    R[run] = {"N": n_docs(run), "retrain_h": rh, "src": src, "n_bank": nb,
              "base": rh, "bank": (nb + 1) * rh if nb else 0.0,
              "magic": magic_h(run) if has_magic else 0.0,
              "ekfac": ekfac_h.get(run, 0.0), "qld": qld,
              "qld_h": sum(qld.values()), "n_qld": len(cs), "n_q": sum(cs.values()),
              "n_ctrl": ctrl}
    R[run]["total"] = sum(R[run][k] for k in ("base", "bank", "magic", "ekfac", "qld_h"))

# ---------------------------------------------- figure membership (derived) ---
src = (ROOT / "scripts" / "scaling_plot_mpl.py").read_text()


def const(name, pat=r"(\[.*?\])\n"):
    m = re.search(rf"^{name} = {pat}", src, re.S | re.M)
    if not m:
        sys.exit(f"scaling_plot_mpl.py no longer defines {name} -- update this script")
    return eval(m.group(1))


NS = const("NS", r"(\[[^\]]*\])")
BATCHES = const("BATCHES", r"(\[[^\]]*\])")
VARIANT_ROWS = const("VARIANT_ROWS")
SERIES_MAX_N = const("SERIES_MAX_N", r"(\{[^}]*\})")
PREFER = const("PREFER", r"(\([^)]*\))")
ADAM, MUON = ("plan_adam_eps1e17_", "sm_adamw_eps1e17_"), ("plan_muon_eps1e17_", "sm_muon_eps1e17_")
# The method appendix truncates both panels at 64k, where serial MAGIC stops.
METHOD_CUT = 64000


def pick_scaling(prefixes, n):
    for r in sorted(rows, key=lambda r: r["run_id"] not in PREFER):
        rid = r["run_id"]
        if (rid.endswith("_bs256") or "_bs256_" in rid) and rid.startswith(prefixes) \
                and (r["n_docs"] or "").strip() and int(float(r["n_docs"])) == n:
            return rid
    return None


def pick_batch(prefixes, bs):
    return next((r["run_id"] for r in rows
                 if r["run_id"].startswith(prefixes) and r["run_id"].endswith(f"16k_bs{bs}")), None)


FIG, FIG_LABEL = defaultdict(set), {}


def need(fig, run, *components):
    if run:
        for c in components:
            FIG[fig].add((run, c))


F1, F2 = "filter_scaling.pdf", "filter_muon_appendix.pdf"
F3, F4 = "filter_method_appendix.pdf", "filter_variants_appendix.pdf"
F5, F6 = "filter_vs_lds.pdf", "filter_scaling_qwen.pdf"
F7 = "filter_heldout.pdf"
FIG_LABEL = {F1: "EK-FAC 1% and top-40 vs corpus size, AdamW",
             F2: "Muon vs AdamW: corpus scaling and the 16k batch sweep",
             F3: "EK-FAC vs MAGIC vs BM25, 1% and top-40, to 64k",
             F4: "Training-setup variants at 16k documents",
             F5: "QLD vs LDS scatter (the only figure needing banks)",
             F6: "Qwen-1.5B replication",
             F7: "Held-out vs in-distribution queries at 64k+"}

for n in NS:                                              # main: 1% and top-40
    need(F1, pick_scaling(ADAM, n), "base", "ekfac",
         "filter_proponents_ekfac", "filter_top40_ekfac")
for n in NS:                                              # muon appendix
    need(F2, pick_scaling(ADAM, n), "base", "ekfac", "filter_proponents_ekfac")
for n in [x for x in NS if x <= SERIES_MAX_N.get("Muon", NS[-1])]:
    need(F2, pick_scaling(MUON, n), "base", "ekfac", "filter_proponents_ekfac")
for b in BATCHES:
    for pre in (ADAM, MUON):
        need(F2, pick_batch(pre, b), "base", "ekfac", "filter_proponents_ekfac")
for n in [x for x in NS if x <= METHOD_CUT]:              # method appendix
    need(F3, pick_scaling(ADAM, n), "base", "ekfac", "magic",
         "filter_proponents_ekfac", "filter_proponents_magic", "filter_proponents_bm25",
         "filter_top40_ekfac", "filter_top40_magic", "filter_top40_bm25")
for _, run in VARIANT_ROWS:                               # variants appendix
    need(F4, run, "base", "ekfac", "filter_proponents_ekfac")
for r in rows:                                            # QLD-vs-LDS scatter
    if r.get("model") != "gpt2" or float(r.get("logit_scale") or 1.0) != 1.0:
        continue
    if not (r.get("steps") or "").strip():
        continue
    for sc in ("magic", "ekfac"):
        if (r.get(f"{sc}_lds") or "").strip() and (r.get(f"filter_{sc}_delta") or "").strip():
            need(F5, r["run_id"], "base", "bank", sc, f"filter_proponents_{sc}")
for run in sorted(R):                                     # qwen and held-out
    if run.startswith("qwen15b_") and R[run]["n_qld"]:
        need(F6, run, "base", "ekfac", "filter_proponents_ekfac", "filter_top40_ekfac")
    for c in R[run]["qld"]:
        if "heldout" in c:
            need(F7, run, "base", "ekfac", c)


def cost(run, component):
    v = R.get(run)
    if not v:
        return 0.0
    return v[component] if component in ("base", "bank", "ekfac", "magic") \
        else v["qld"].get(component, 0.0)


def stage(component):
    return component if component in ("base", "bank", "ekfac", "magic") else "qld"


# ------------------------------------------------- supporting work, no bank ---
# The tuning grid and the metasmoothness probes carry no bank and no filter, so
# they never appear in R; they are training runs and price straight off the law.
# An ms probe is a finite difference around a trained model -- 4 trainings, from
# the ms512 and ms256 logs.
MS_TRAININGS = 4


def _train_h(n, model_family="gpt2"):
    return n * SEC_PER_DOC[model_family] * NPROC / 3600


tuning = defaultdict(lambda: [0, 0.0])
for r in csv.DictReader(open(ROOT / "tuning.csv")):
    fm = "med" if ("medium" in r["run_id"] or "large" in r["run_id"]) else "gpt2"
    t = tuning[r["status"]]
    t[0] += 1
    t[1] += _train_h(int(float(r["n_docs"])), fm)

ms_h = 0.0
for root in ROOTS:
    if not os.path.isdir(root):
        continue
    for run in os.listdir(root):
        for d in ("ms", "ms_seeds"):
            if os.path.isdir(os.path.join(root, run, d)):
                ms_h += MS_TRAININGS * _train_h(n_docs(run), fam(run))

# ------------------------------------------------------------------ output ---
def rule(w):
    print("-" * w)


print("COST LAW -- one 2-epoch GPT-2-124M retrain is %.4f wall-s per document at nproc %d"
      % (SEC_PER_DOC["gpt2"], NPROC))
print()
hdr = "%9s %6s %9s %10s %9s %12s" % ("documents", "mult", "1 retrain", "LDS(101)",
                                     "QLD(23)", "MAGIC(20q)")
print(hdr)
rule(len(hdr))
unit = NS[0]
for n in NS:
    h = n * SEC_PER_DOC["gpt2"] * NPROC / 3600
    print("%9s %5dx %9.2f %10.0f %9.1f %12.0f"
          % (f"{n // 1000}k", n // unit, h, (BANK_SUBSETS + 1) * h,
             (20 + FILTER_CONTROLS) * h,
             20 * n * MAGIC_SEC_PER_DOC["gpt2"] * NPROC / 3600))
print("\n(GPU-hours. EK-FAC is measured per row, not modelled.)\n")

print("PER FIGURE")
hdr = "%-34s %5s %7s %7s %7s %7s %8s %10s" % ("figure", "rows", "LDS", "MAGIC",
                                              "EK-FAC", "QLD", "total", "exclusive")
print(hdr)
rule(len(hdr))
shared = defaultdict(set)
for f, pairs in FIG.items():
    for p in pairs:
        shared[p].add(f)
union = set().union(*FIG.values()) if FIG else set()
for f in (F1, F2, F3, F4, F5, F6, F7):
    s = defaultdict(float)
    for run, c in FIG[f]:
        s[stage(c)] += cost(run, c)
    ex = sum(cost(*p) for p in FIG[f] if len(shared[p]) == 1)
    print("%-34s %5d %7.0f %7.0f %7.0f %7.0f %8.0f %10.0f"
          % (f, len({p[0] for p in FIG[f]}), s["bank"], s["magic"], s["ekfac"],
             s["qld"], sum(s.values()), ex))
rule(len(hdr))
s = defaultdict(float)
for run, c in union:
    s[stage(c)] += cost(run, c)
print("%-34s %5d %7.0f %7.0f %7.0f %7.0f %8.0f %10s"
      % ("union (deduplicated)", len({p[0] for p in union}), s["bank"], s["magic"],
         s["ekfac"], s["qld"], sum(s.values()), "-"))
print()

print("PER RUN")
hdr = "%-40s %7s %7s %5s %7s %7s %7s %7s %3s %8s  %s" % (
    "run", "N", "retr_h", "src", "LDS", "MAGIC", "EK-FAC", "QLD", "n", "total", "figures")
print(hdr)
rule(len(hdr))
tot = defaultdict(float)
for run in sorted(R, key=lambda r: -R[r]["total"]):
    v = R[run]
    figs = ",".join(sorted({f for f, ps in FIG.items() for p in ps if p[0] == run})) or "-"
    print("%-40s %6dk %7.2f %5s %7.0f %7.0f %7.0f %7.0f %3d %8.0f  %s"
          % (run[:40], v["N"] // 1000, v["retrain_h"], v["src"], v["bank"], v["magic"],
             v["ekfac"], v["qld_h"], v["n_qld"], v["total"], figs))
    for k in ("base", "bank", "magic", "ekfac", "qld_h", "total"):
        tot[k] += v[k]
rule(len(hdr))
print("%-40s %7s %7s %5s %7.0f %7.0f %7.0f %7.0f %3s %8.0f"
      % (f"{len(R)} rows", "", "", "", tot["bank"], tot["magic"], tot["ekfac"],
         tot["qld_h"], "", tot["total"]))
if ekfac_unattributed > 1:
    print("(%.0f GPU-h of EK-FAC could not be tied to a run id)" % ekfac_unattributed)
print()
print("SUPPORTING WORK (no bank, no filter -- priced off the cost law)")
for status in sorted(tuning):
    print("  %-28s %4d runs %8.0f" % ("lr tuning, " + status, *tuning[status]))
print("  %-28s %4s       %8.0f" % ("metasmoothness probes", "", ms_h))
print()
print("retrains on disk: %d bank + %d QLD query + %d QLD control + %d base = %d"
      % (sum(v["n_bank"] for v in R.values()), sum(v["n_q"] for v in R.values()),
         sum(v["n_ctrl"] for v in R.values()), len(R),
         sum(v["n_bank"] + v["n_q"] + v["n_ctrl"] + 1 for v in R.values())))
kept = tot["total"] + tuning["measured"][1] + tuning["cut"][1] + ms_h
drawn = sum(cost(*p) for p in union)
in_fig = {p[0] for p in union}
orphan_rows = [r for r in R if r not in in_fig]
orphan_h = sum(R[r]["total"] for r in orphan_rows)
print("figures %.0f | all rows %.0f | + tuning and ms = %.0f GPU-h retained"
      % (drawn, tot["total"], kept))
print("not drawn by any figure: %.0f -- %d rows in no figure (%.0f) + "
      "undrawn components on figure rows (%.0f)"
      % (tot["total"] - drawn, len(orphan_rows), orphan_h,
         tot["total"] - drawn - orphan_h))

if args.write:
    out = []
    for run in sorted(R, key=lambda r: -R[r]["total"]):
        v = R[run]
        out.append({"run_id": run, "n_docs": v["N"], "retrain_gpu_h": round(v["retrain_h"], 3),
                    "cost_source": v["src"], "n_bank_retrains": v["n_bank"],
                    "n_qld_campaigns": v["n_qld"], "n_qld_retrains": v["n_q"] + v["n_ctrl"],
                    "lds_gpu_h": round(v["bank"], 1), "magic_gpu_h": round(v["magic"], 1),
                    "ekfac_gpu_h": round(v["ekfac"], 1), "qld_gpu_h": round(v["qld_h"], 1),
                    "total_gpu_h": round(v["total"], 1),
                    "figures": " ".join(sorted({f for f, ps in FIG.items()
                                                for p in ps if p[0] == run}))})
    _write(LEDGER, out, list(out[0]))
