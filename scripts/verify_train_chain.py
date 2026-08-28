from datasets import load_from_disk

DS = "/mnt/ssd-2/lucia/datasets_local"
new = load_from_disk(DS + "/train_1M.hf")
print(f"  train_1M rows: {len(new):,}")
fails = 0
for name, n in [("train_4k", 4000), ("train_16k", 16000), ("train_64k", 64000),
                ("train_128k", 128000), ("train_256k", 256000), ("train_512k", 512000)]:
    old = load_from_disk(f"{DS}/{name}.hf")
    probes = [0, 1, n // 3, n // 2, n - 2, n - 1]
    bad = [i for i in probes if old[i]["input_ids"] != new[i]["input_ids"]]
    msg = "prefix OK" if not bad else f"MISMATCH {bad}"
    print(f"  {name}: {msg}")
    fails += bool(bad)
s = {tuple(r) for r in new["input_ids"]}
print(f"  distinct: {len(s):,}  duplicates: {1000000 - len(s)}")
for pool, root in (("query_20", DS), ("query_50", DS),
                   ("heldout_4k", "/mnt/ssd-1/lucia/bergson-damping/runs/ekfac_vs_n/datasets")):
    d = load_from_disk(f"{root}/{pool}.hf")
    ov = sum(tuple(r) in s for r in d["input_ids"])
    print(f"  {pool}: overlap {ov}/{len(d)}")
    fails += bool(ov)
print("  RESULT:", "ALL PASSED" if fails == 0 and len(s) == 1000000 else "FAILED")
