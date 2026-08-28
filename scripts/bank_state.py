"""Which subsets a bank still owes, and whether its ground truth is merged."""
import glob
import os
import re
import sys

D = sys.argv[1]
done = {int(re.search(r"subset_(\d+)$", d).group(1))
        for d in glob.glob(os.path.join(D, "retrained", "subset_*")) if os.path.isdir(d)}
missing = [i for i in range(100) if i not in done]
shards = glob.glob(os.path.join(D, "validation_*.csv"))
merged = os.path.isfile(os.path.join(D, "validation_merged.csv"))
print("  subsets    %d/100" % len(done))
print("  missing    %s" % (missing if len(missing) <= 25 else str(missing[:25]) + " ..."))
print("  val shards %d   merged: %s" % (len(shards), "yes" if merged else "NO"))
