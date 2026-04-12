#!/usr/bin/env python3
"""
Run both benchmark and accuracy comparisons, then print side-by-side tables.

Usage:
    python3 compare.py                 # binaries expected in ./build/
    python3 compare.py -b              # benchmark only
    python3 compare.py -a              # accuracy only
"""

import subprocess, sys, os, csv, io, argparse

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build')

def run(exe):
    r = subprocess.run([exe], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR running {exe}:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return r.stdout

def parse_bench(output):
    """Return dict (type, dim) -> ms"""
    d = {}
    for row in csv.DictReader(io.StringIO(output)):
        d[(row['type'], int(row['dim']))] = float(row['ms'])
    return d

def parse_accuracy(output):
    """Return dict case_name -> {n_queries, wrong_exact, wrong_radius}"""
    d = {}
    for row in csv.DictReader(io.StringIO(output)):
        d[row['case']] = {
            'n':      int(row['n_queries']),
            'exact':  int(row['wrong_exact']),
            'radius': int(row['wrong_radius']),
        }
    return d

# --------------------------------------------------------------------------
# Benchmark table
# --------------------------------------------------------------------------

def print_bench(orig_exe, impr_exe):
    print(f"\nRunning {os.path.basename(orig_exe)} ...", flush=True)
    orig = parse_bench(run(orig_exe))
    print(f"Running {os.path.basename(impr_exe)} ...", flush=True)
    impr = parse_bench(run(impr_exe))

    TYPES  = ['build', 'knn_approx', 'knn_exact', 'radius_approx']
    DIMS   = sorted(set(d for (_, d) in orig))
    labels = {
        'build':         'Build index',
        'knn_approx':    'KNN approx  (checks=32)',
        'knn_exact':     'KNN exact   (checks=∞) ',
        'radius_approx': 'Radius approx          ',
    }

    W = 12
    hdr = f"{'Benchmark':<28} {'dim':>5}  {'Original':>{W}}  {'Improved':>{W}}  {'Speedup':>{W}}"
    sep = '-' * len(hdr)

    print()
    print("PERFORMANCE")
    print(sep)
    print(hdr)
    print(sep)
    for typ in TYPES:
        for dim in DIMS:
            key = (typ, dim)
            if key not in orig or key not in impr:
                continue
            o, i = orig[key], impr[key]
            sp = o / i if i > 0 else float('inf')
            note = '  <--' if sp >= 1.4 else ''
            print(f"{labels[typ]:<28} {dim:>5}  {o:>{W}.1f}ms  {i:>{W}.1f}ms  {sp:>{W}.2f}x{note}")
        print()
    print(sep)
    print("N=10 000 points, K=10, Q=500 queries, trees=1, median of multiple runs")

# --------------------------------------------------------------------------
# Accuracy table
# --------------------------------------------------------------------------

def print_accuracy(orig_exe, impr_exe):
    print(f"\nRunning {os.path.basename(orig_exe)} ...", flush=True)
    orig = parse_accuracy(run(orig_exe))
    print(f"Running {os.path.basename(impr_exe)} ...", flush=True)
    impr = parse_accuracy(run(impr_exe))

    cases = list(orig.keys())

    hdr = f"{'Test case':<20} {'queries':>8}  {'Orig exact':>10}  {'Impr exact':>10}  {'Orig radius':>12}  {'Impr radius':>12}"
    sep = '-' * len(hdr)

    print()
    print("CORRECTNESS  (wrong = query where result differs from brute-force ground truth)")
    print(sep)
    print(hdr)
    print(sep)
    for c in cases:
        o = orig[c]
        i = impr.get(c, {'n': 0, 'exact': '?', 'radius': '?'})
        n = o['n']
        oe, ie = o['exact'],  i['exact']
        orr, ir = o['radius'], i['radius']
        note_e = '  BUG' if isinstance(oe, int) and oe > 0 else ''
        note_r = '  BUG' if isinstance(orr, int) and orr > 0 else ''
        print(f"{c:<20} {n:>8}  {oe:>10}{note_e}")
        print(f"{'':20} {'':8}  {'':10}  {ie:>10}  {orr:>12}{note_r}  {ir:>12}")
    print(sep)

    # Simple summary
    total_orig = sum(v['exact'] + v['radius'] for v in orig.values())
    total_impr = sum(v['exact'] + v['radius'] for v in impr.values())
    print(f"\nTotal wrong queries — Original: {total_orig},  Improved: {total_impr}")
    if total_impr == 0:
        print("Improved implementation: PERFECT accuracy on all test cases.")

# --------------------------------------------------------------------------
# Better accuracy table — side by side
# --------------------------------------------------------------------------

def print_accuracy_v2(orig_exe, impr_exe):
    print(f"\nRunning {os.path.basename(orig_exe)} ...", flush=True)
    orig = parse_accuracy(run(orig_exe))
    print(f"Running {os.path.basename(impr_exe)} ...", flush=True)
    impr = parse_accuracy(run(impr_exe))

    cases = list(orig.keys())

    hdr = (f"{'Test case':<20} {'N_q':>6}  "
           f"{'Orig wrong':>11}  {'Impr wrong':>11}  "
           f"{'Orig radius':>12}  {'Impr radius':>12}")
    sep = '-' * len(hdr)

    hdr = (f"{'Test case':<18} {'N_q':>6}  "
           f"{'KNN wrong':>10}  {'KNN wrong':>10}  "
           f"{'Rad wrong':>10}  {'Rad wrong':>10}")
    sub = (f"{'':18} {'':6}  "
           f"{'(original)':>10}  {'(improved)':>10}  "
           f"{'(original)':>10}  {'(improved)':>10}")
    sep = '-' * len(hdr)

    print()
    print("CORRECTNESS  (wrong = queries where exact knnSearch result differs from brute-force)")
    print(sep)
    print(hdr)
    print(sub)
    print(sep)
    for c in cases:
        o  = orig[c]
        im = impr.get(c, {'n': 0, 'exact': 0, 'radius': 0})
        n  = o['n']
        oe, ie   = o['exact'],  im['exact']
        orr, irr = o['radius'], im['radius']
        flag_e = ' !!!' if isinstance(oe, int) and oe > 0 else '    '
        flag_r = ' !!!' if isinstance(orr, int) and orr > 0 else '    '
        print(f"{c:<18} {n:>6}  {oe:>10}{flag_e}  {ie:>10}      {orr:>10}{flag_r}  {irr:>10}")
    print(sep)

    total_orig_e = sum(v['exact']  for v in orig.values())
    total_orig_r = sum(v['radius'] for v in orig.values())
    total_impr_e = sum(v['exact']  for v in impr.values())
    total_impr_r = sum(v['radius'] for v in impr.values())
    print(f"\nTotal KNN wrong    — Original: {total_orig_e:4d},  Improved: {total_impr_e}")
    print(f"Total radius wrong — Original: {total_orig_r:4d},  Improved: {total_impr_r}")
    if total_impr_e == 0 and total_impr_r == 0:
        print("Improved: PERFECT accuracy across all test cases.")

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--bench-only',    action='store_true')
    ap.add_argument('-a', '--accuracy-only', action='store_true')
    args = ap.parse_args()

    do_bench    = not args.accuracy_only
    do_accuracy = not args.bench_only

    if do_bench:
        print_bench(
            os.path.join(BUILD_DIR, 'bench_original'),
            os.path.join(BUILD_DIR, 'bench_improved'),
        )

    if do_accuracy:
        print_accuracy_v2(
            os.path.join(BUILD_DIR, 'accuracy_original'),
            os.path.join(BUILD_DIR, 'accuracy_improved'),
        )

    print()

if __name__ == '__main__':
    main()
