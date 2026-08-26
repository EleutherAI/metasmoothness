import csv, glob, os
D = {r["run"]: r for r in csv.DictReader(open("data/filter_deltas.csv"))}
rows = list(csv.DictReader(open("experiments.csv")))
print("%-38s %-22s %-22s" % ("run", "MAGIC lds/delta", "EKFAC lds/delta"))
gaps = []
for r in rows:
    rid = r["run_id"]
    ml = (r.get("magic_lds") or "").strip()
    el = (r.get("ekfac_lds") or "").strip()
    if not (ml or el):
        continue
    d = D.get(rid, {})
    md = (d.get("magic_mean") or "").strip()
    ed = (d.get("ekfac_mean") or "").strip()
    m = "%s / %s" % (ml[:7] or "-", md[:7] or "MISSING")
    e = "%s / %s" % (el[:7] or "-", ed[:7] or "MISSING")
    flag = ""
    if ml and not md:
        gaps.append((rid, "magic")); flag = "  <-- MAGIC delta missing"
    if el and not ed:
        gaps.append((rid, "ekfac")); flag += "  <-- EKFAC delta missing"
    print("%-38s %-22s %-22s%s" % (rid, m, e, flag))
print()
print("%d (row, scorer) pairs have an LDS but no delta" % len(gaps))
for rid, sc in gaps:
    root = None
    for b in ("/mnt/ssd-1", "/mnt/ssd-2"):
        p = "%s/lucia/paper_runs/experiments/%s" % (b, rid)
        if os.path.isdir(p):
            root = p
    nb = len(glob.glob(root + "/retrained/subset_*")) + len(glob.glob(root + "/bank_from_filter/retrained/subset_*")) if root else 0
    sdir = "scores" if sc == "magic" else "ekfac_scores/scores"
    has = os.path.isfile(os.path.join(root or "", sdir, "info.json"))
    print("   %-38s %-6s bank=%d scores=%s" % (rid, sc, nb, has))
