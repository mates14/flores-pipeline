"""
One-off conversion of an IRAF `identify` database dump (the format written to
e.g. ~/tmp/flores/260716/id) into the plain CSV seed line list this pipeline
bootstraps from.

Usage:
    python3 convert_seed.py ../../260716/id flores_seed_linelist.csv

The IRAF feature table columns are:
    pixel  wavelength_user  wavelength_fit  fwhm_guess  flag1  flag2
(same format handled by perek_pipelines/calibrate.py:parse_idcomp). We take
the fitted wavelength (2nd wavelength column) as ground truth, matching what
that parser and fit_comparison() use downstream.
"""
import sys
import csv


def parse_identify_db(path):
    rows = []
    in_table = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("features"):
                in_table = True
                continue
            if not in_table:
                continue
            if not line or not line[0].isdigit():
                in_table = False
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pixel = float(parts[0])
            wl_fit = float(parts[2])
            fwhm = float(parts[3])
            if pixel < 0 or wl_fit <= 0:
                continue
            rows.append((pixel, wl_fit, fwhm))
    return rows


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <iraf_id_file> <out_csv>")
        sys.exit(1)

    rows = parse_identify_db(sys.argv[1])
    rows.sort(key=lambda r: r[0])

    with open(sys.argv[2], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pixel", "wavelength_air", "fwhm_guess_pix"])
        w.writerows(rows)

    print(f"wrote {len(rows)} seed lines to {sys.argv[2]}")


if __name__ == "__main__":
    main()
