"""
ScienceQA Dataset Sanity Check v2 — Multi-dimensional joint check
Covers: missing values / choices parsing / answer bounds / id uniqueness / cross-split leakage /
     text length / image physical features / sample_submission alignment / baseline acc
"""

import os
import ast
import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# =====================================================================
# 0. Configuration — Modify here
# =====================================================================
BASE_DIR    = Path(r"C:\Users\benxi\Downloads\images")  # Change to your local root directory
TRAIN_CSV   = BASE_DIR / "train.csv"
VAL_CSV     = BASE_DIR / "val.csv"
TEST_CSV    = BASE_DIR / "test.csv"
SAMPLE_CSV  = BASE_DIR / "sample_submission.csv"

# Image folders (In your Windows, it's structured as BASE/train, BASE/val, BASE/test)
IMG_DIRS = {
    "train": BASE_DIR / "train",
    "val":   BASE_DIR / "val",
    "test":  BASE_DIR / "test",
}

SCAN_IMAGE_PHYSICAL = True   # Whether to scan image physical features (slow)
TEXT_COLS = ['question', 'choices', 'hint', 'lecture', 'solution',
             'subject', 'topic', 'category', 'skill']

# =====================================================================
# 1. General Tools
# =====================================================================
def safe_parse(s):
    if pd.isna(s):
        return None
    s = str(s)
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return None

def basename_x(p):
    """Cross-platform basename: handles both '/' and '\\' separators"""
    return str(p).replace('\\', '/').split('/')[-1]

def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

# =====================================================================
# 2. Load
# =====================================================================
banner("[0] Load CSVs")
train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV)
test  = pd.read_csv(TEST_CSV)
samp  = pd.read_csv(SAMPLE_CSV) if SAMPLE_CSV.exists() else None
print(f"  train={len(train)}  val={len(val)}  test={len(test)}  "
      f"sample_submission={len(samp) if samp is not None else 'N/A'}")

DFS = {"train": train, "val": val, "test": test}

# =====================================================================
# 3. Missing values + Empty strings
# =====================================================================
banner("[1] NaN + empty/whitespace check")
for name, df in DFS.items():
    print(f"\n  --- {name} ---")
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss):
        for c, n in miss.items():
            print(f"    NaN  {c}: {n} ({n/len(df)*100:.1f}%)")
    for c in TEXT_COLS:
        if c not in df.columns: continue
        s = df[c].astype(str)
        blank = ((s == 'nan') | (s.str.strip() == '')).sum()
        nan_count = df[c].isnull().sum()
        only_blank = blank - nan_count
        if only_blank > 0:
            print(f"    BLANK {c}: {only_blank} (Blank but not NaN)")

# =====================================================================
# 4. Choices parsing + num_choices consistency
# =====================================================================
banner("[2] Choices parse + num_choices consistency")
for name, df in DFS.items():
    fail = mism = empty = 0
    for _, r in df.iterrows():
        parsed = safe_parse(r['choices'])
        if parsed is None or not isinstance(parsed, list):
            fail += 1; continue
        if len(parsed) != r['num_choices']:
            mism += 1
        if any((not isinstance(c, str)) or str(c).strip()=='' for c in parsed):
            empty += 1
    print(f"  {name}: parse_fail={fail}  len_mismatch={mism}  has_empty_opt={empty}")

# =====================================================================
# 5. Answer bounds
# =====================================================================
banner("[3] Answer bounds (must in [0, num_choices-1])")
for name, df in DFS.items():
    if 'answer' not in df.columns:
        print(f"  {name}: no answer column (expected for test)"); continue
    bad = df[(df['answer'] < 0) | (df['answer'] >= df['num_choices'])]
    print(f"  {name}: out_of_bounds={len(bad)}, "
          f"answer∈[{df['answer'].min()},{df['answer'].max()}], "
          f"nc∈[{df['num_choices'].min()},{df['num_choices'].max()}]")

# =====================================================================
# 6. ID uniqueness + Cross-split non-overlap + sample_submission alignment
# =====================================================================
banner("[4] id uniqueness / cross-split / submission alignment")
for name, df in DFS.items():
    print(f"  {name}: rows={len(df)}, unique_id={df['id'].nunique()}, "
          f"dup={df['id'].duplicated().sum()}")
print(f"  train ∩ val ids:   {len(set(train['id']) & set(val['id']))}")
print(f"  train ∩ test ids:  {len(set(train['id']) & set(test['id']))}")
print(f"  val   ∩ test ids:  {len(set(val['id']) & set(test['id']))}")

if samp is not None:
    print(f"\n  sample == test set? "
          f"{set(samp['id'])==set(test['id'])}")
    print(f"  sample order == test order? "
          f"{samp['id'].tolist()==test['id'].tolist()}")
    print(f"  sample columns: {list(samp.columns)} (must be ['id','answer'])")

# =====================================================================
# 7. id ↔ image_path consistency + Physical existence of images
# =====================================================================
banner("[5] image_path validity + on-disk existence")
for name, df in DFS.items():
    miss = []
    bad_align = 0
    for _, r in df.iterrows():
        fname = basename_x(r['image_path'])
        # Filename (without extension) should equal id
        stem = os.path.splitext(fname)[0]
        if stem != r['id']:
            bad_align += 1
        # Check physical existence
        full = IMG_DIRS[name] / fname
        if not full.exists():
            miss.append(r['id'])
    print(f"  {name}: samples with stem!=id = {bad_align}; "
          f"physical missing = {len(miss)}/{len(df)}")
    if miss[:3]:
        print(f"    Example missing: {miss[:3]}")

# =====================================================================
# 8. Text length distribution (determines tokenizer max_length)
# =====================================================================
banner("[6] Text length distribution (chars)")
for name, df in DFS.items():
    print(f"\n  --- {name} ---")
    for c in ['question', 'lecture', 'hint', 'solution', 'choices']:
        if c not in df.columns: continue
        L = df[c].fillna('').astype(str).apply(len)
        print(f"    {c:9s} mean={L.mean():.0f} med={L.median():.0f} "
              f"p95={L.quantile(0.95):.0f} p99={L.quantile(0.99):.0f} "
              f"max={L.max()}")
    # Total length (estimating token budget)
    total = sum(df[c].fillna('').astype(str).apply(len)
                for c in ['question','lecture','hint','choices'] if c in df.columns)
    print(f"    TOTAL (q+lec+hint+ch): "
          f"p95={total.quantile(0.95):.0f} p99={total.quantile(0.99):.0f} max={total.max()}")

# =====================================================================
# 9. Label distribution (split by num_choices) + baseline acc
# =====================================================================
banner("[7] Label imbalance — answer x num_choices")
for name, df in DFS.items():
    if 'answer' not in df.columns: continue
    print(f"\n  --- {name} ---")
    print(f"    overall: {dict(df['answer'].value_counts().sort_index())}")
    for nc in sorted(df['num_choices'].unique()):
        sub = df[df['num_choices']==nc]
        vc = sub['answer'].value_counts().sort_index()
        line = "  ".join(f"{k}:{v}({v/len(sub)*100:.0f}%)" for k,v in vc.items())
        print(f"    nc={nc} (n={len(sub)}): {line}")

# Baseline: predict majority on val
banner("[8] Baseline accuracy on val")
mode_per_nc = train.groupby('num_choices')['answer'].agg(lambda s: s.mode().iloc[0]).to_dict()
val_pred_mode = val['num_choices'].map(mode_per_nc)
print(f"  predict majority-per-nc: val acc = {(val_pred_mode==val['answer']).mean()*100:.2f}%")
print(f"  predict 0:               val acc = {(val['answer']==0).mean()*100:.2f}%")
print(f"  uniform random:          val acc = {(1/val['num_choices']).mean()*100:.2f}%")

# =====================================================================
# 10. Cross-split question / signature overlap (Data leakage detection)
# =====================================================================
banner("[9] Question overlap (potential leakage)")
def sig(df, with_choices=True, with_image=False):
    s = df['question'].astype(str)
    if with_choices: s = s + '||' + df['choices'].astype(str)
    if with_image:   s = s + '||' + df['image_path'].apply(basename_x)
    return set(s)

for combo, kw in [("question only", dict(with_choices=False, with_image=False)),
                  ("question+choices", dict(with_choices=True, with_image=False)),
                  ("question+choices+image", dict(with_choices=True, with_image=True))]:
    print(f"\n  -- {combo} --")
    a = sig(train, **kw); b = sig(val, **kw); c = sig(test, **kw)
    print(f"    train ∩ val:  {len(a&b)}")
    print(f"    train ∩ test: {len(a&c)}")
    print(f"    val   ∩ test: {len(b&c)}")

# Same-question clustering within train
banner("[10] Repeated questions in train")
g = train['question'].value_counts()
print(f"  unique questions: {len(g)} / {len(train)} (Duplication rate={(1-len(g)/len(train))*100:.1f}%)")
print(f"  Top 3 most-repeated:")
for q, n in g.head(3).items():
    print(f"    n={n}: {q[:70]}...")

# =====================================================================
# 11. Hint/lecture leaking answer text (Strong signal friendly to small models)
# =====================================================================
banner("[11] Hint/lecture leak answer text")
for name, df in [("train", train), ("val", val)]:
    leak_hint = leak_lec = 0
    for _, r in df.iterrows():
        choices = safe_parse(r['choices'])
        if not choices: continue
        ai = int(r['answer'])
        if ai >= len(choices): continue
        ans = str(choices[ai]).strip().lower()
        if len(ans) < 5: continue
        if pd.notna(r.get('hint')) and ans in str(r['hint']).lower():
            leak_hint += 1
        if pd.notna(r.get('lecture')) and ans in str(r['lecture']).lower():
            leak_lec += 1
    print(f"  {name}: hint contains answer {leak_hint}({leak_hint/len(df)*100:.1f}%), "
          f"lecture contains answer {leak_lec}({leak_lec/len(df)*100:.1f}%)")

# =====================================================================
# 12. Visual dependency
# =====================================================================
banner("[12] Visual dependency keywords in question")
KW = ['this map','this image','this picture','this diagram','this figure',
      'this graph','this chart','this drawing','in the picture',
      'in the image','in the diagram','shown above','as shown']
for name, df in DFS.items():
    cnt = df['question'].astype(str).str.lower().apply(
        lambda q: any(k in q for k in KW)
    ).sum()
    print(f"  {name}: {cnt}/{len(df)} ({cnt/len(df)*100:.1f}%)")

# =====================================================================
# 13. Image physical feature scan (Optional, slow)
# =====================================================================
if SCAN_IMAGE_PHYSICAL:
    banner("[13] Image physical scan (size / brightness / variance)")
    for name, img_dir in IMG_DIRS.items():
        if not img_dir.exists():
            print(f"  {name}: dir not found, skip"); continue
        ids_in_csv = set(DFS[name]['id'])
        files = list(img_dir.glob("*.png"))
        print(f"\n  --- {name} ({len(files)} files on disk) ---")
        # Compare with CSV
        files_id = {f.stem for f in files}
        only_in_csv = ids_in_csv - files_id
        only_on_disk = files_id - ids_in_csv
        print(f"    Images missing in CSV: {len(only_in_csv)}; Extra files on disk: {len(only_on_disk)}")

        stats = []
        corrupted = []
        for f in tqdm(files, desc=f"scan {name}", leave=False):
            try:
                with Image.open(f) as im:
                    arr = np.asarray(im.convert('L'))
                stats.append({
                    'id': f.stem, 'w': arr.shape[1], 'h': arr.shape[0],
                    'kb': f.stat().st_size/1024,
                    'bright': arr.mean(), 'var': arr.var()
                })
            except Exception as e:
                corrupted.append((f.name, str(e)[:80]))
        print(f"    corrupted: {len(corrupted)}")
        for c in corrupted[:3]:
            print(f"      {c}")
        if stats:
            S = pd.DataFrame(stats)
            print(f"    size W: med={S['w'].median():.0f} max={S['w'].max()}; "
                  f"H: med={S['h'].median():.0f} max={S['h'].max()}")
            print(f"    bright p1={S['bright'].quantile(0.01):.0f} "
                  f"p99={S['bright'].quantile(0.99):.0f}")
            print(f"    var p1={S['var'].quantile(0.01):.1f} (The lower, the closer to a solid color)")
            # List extreme low variance (possibly problematic images)
            ultra_flat = S[S['var'] < 50]
            print(f"    Images with var<50 (almost solid color): {len(ultra_flat)}")
            if len(ultra_flat):
                print(f"      ids: {ultra_flat['id'].head(5).tolist()}")

print("\n" + "=" * 72)
print("✅ All checks completed")
print("=" * 72)
