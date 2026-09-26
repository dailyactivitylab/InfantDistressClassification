"""
Drop brief distress runs from per-second labels or start,end,label CSVs.

The released models use 5 s windows with 4 s overlap. A 1–3 s rustle, cloth
on the microphone, or similar sound can flip a few overlapping windows and
show up as a short distress streak. This script is optional. Default predict
output is unchanged.

  python clean_short_distress.py --labels 0 0 1 1 0 2 2 2 0 --min_sec 3
  python clean_short_distress.py --csv path/to/P12_1.csv --min_sec 3
"""

import argparse
import csv
import os


def drop_short_distress(labels, min_sec=3):
    """Set nonzero runs shorter than min_sec seconds to 0. 1 and 2 both count as distress."""
    out = [int(x) for x in labels]
    i, n = 0, len(out)
    while i < n:
        if out[i] == 0:
            i += 1
            continue
        j = i
        while j < n and out[j] != 0:
            j += 1
        if j - i < min_sec:
            for k in range(i, j):
                out[k] = 0
        i = j
    return out


def csv_to_seconds(rows):
    labels = []
    for start, end, lab in rows:
        s, e = int(float(start)), int(round(float(end)))
        raw = str(lab).strip().lower()
        try:
            val = int(float(raw))
        except ValueError:
            if raw in {"fuss"}:
                val = 1
            elif raw in {"cry", "scream"}:
                val = 2
            else:
                val = 0
        if e < s:
            continue
        if e > len(labels):
            labels.extend([0] * (e - len(labels)))
        for t in range(s, e):
            if t < len(labels):
                labels[t] = val
    return labels


def seconds_to_csv(labels):
    rows = []
    i, n = 0, len(labels)
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        rows.append((i, j, labels[i]))
        i = j
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Remove distress episodes shorter than --min_sec seconds"
    )
    parser.add_argument("--min_sec", type=int, default=3, help="Keep distress runs of this length or longer")
    parser.add_argument("--csv", help="Input headerless start,end,label CSV")
    parser.add_argument("--out", help="Where to write the cleaned CSV (default: *_clean.csv)")
    parser.add_argument(
        "--labels",
        nargs="*",
        type=int,
        help="Per-second labels on the command line, e.g. 0 0 1 1 0",
    )
    args = parser.parse_args()
    if args.min_sec < 1:
        raise SystemExit("--min_sec must be >= 1")

    if args.csv:
        with open(args.csv, newline="") as f:
            rows = [tuple(row[:3]) for row in csv.reader(f) if len(row) >= 3]
        cleaned = drop_short_distress(csv_to_seconds(rows), args.min_sec)
        dest = args.out or os.path.splitext(args.csv)[0] + "_clean.csv"
        with open(dest, "w", newline="") as f:
            csv.writer(f).writerows(seconds_to_csv(cleaned))
        before = sum(1 for _, _, lab in rows if str(lab) not in {"0", "0.0"})
        after = sum(1 for x in cleaned if x != 0)
        print(f"Wrote {dest}")
        print(f"Distress seconds: {before} -> {after} (dropped runs < {args.min_sec}s)")
        return

    if args.labels is None:
        raise SystemExit("Pass --csv path.csv or --labels 0 1 1 0 ...")

    cleaned = drop_short_distress(args.labels, args.min_sec)
    print("input:  ", " ".join(str(x) for x in args.labels))
    print("output: ", " ".join(str(x) for x in cleaned))


if __name__ == "__main__":
    main()
