# flores-pipeline

Reduction pipeline for **FLORES**, the low-resolution prism spectrograph on
the D50 (50cm) telescope at Ondřejov. Turns raw two-fiber FITS frames into
wavelength-calibrated, optionally flux-calibrated 1D spectra.

No interactive steps: point it at a directory of raw frames and it produces
`(wavelength, flux, error)` text files, skipping (with a reason printed to
stderr) anything that fails a quality gate rather than silently emitting a
bad spectrum.

## Install

```
pip install -r requirements.txt
```

Python 3.9+ should work; developed against 3.14. No non-Python dependencies.

## Instrument background

FLORES feeds two fibers (two sky positions) into every raw frame, ~18 rows
apart. Which fiber is "object" and which is "sky" is the observer's choice
per observation and isn't inferred by the pipeline — both fibers are always
reduced and written out separately, clearly labelled by tag
(`tilted`/`untilted`, based on which fiber shows a real geometric tilt in
its spectral lines — see `trace.py`). It's on the user to decide downstream
which tag corresponds to which target for a given night.

Being a prism (not a grating/echelle), the resolution runs from ~2000 in
the blue down to ~200 in the NIR, and the wavelength solution holds for as
long as the input mirror isn't moved — so a comparison-lamp (ThAr) exposure
is a session-level anchor (typically one at the start of the night, one at
the end), not something needed per science frame.

## Pipeline stages

1. **Frame classification** (`frames.py`) — sorts raw FITS by `IMAGETYP`
   into dark / comp (ThAr) / science. Older/un-migrated nights that don't
   tag frames correctly can be told apart explicitly with `--comps`.
2. **Trace + tilt + wavelength calibration** (`trace.py`, `wavecal.py`) —
   each comp frame is traced (both fibers found), tilt is measured per
   fiber, and a monotonic PCHIP dispersion solution is bootstrapped from a
   one-time seed line list against a full ThAr atlas. Comp solutions are
   cross-checked across the night within each fiber; a warning is printed
   if they disagree beyond expectation (e.g. the input mirror moved).
3. **Science extraction** (`trace.py`, `reduce_flores.py`) — each science
   frame is traced fresh (target position can drift during the night), and
   each fiber found is matched to its nearest-in-time comp's same fiber by
   trace row position, then extracted with two-pass optimal extraction
   (Horne 1986), which also does the cosmic-ray rejection.
4. **Error propagation** (`noise.py`) — CCD equation from `GAIN`/`RDNOISE`
   headers when available, otherwise a DER_SNR-style local-scatter fallback.
   Barycentric velocity correction is applied when the frame has the needed
   headers.
5. **Optional coadding** (`coadd_light.py`, `coadd_sets.py`) — an explicit,
   opt-in step for combining several sub-exposures of one target *before*
   reduction (e.g. a series of 5-minute shots instead of one long,
   riskier exposure). Never done automatically — not every target is a
   single shot, and guessing which frames "belong together" from headers
   alone is exactly the kind of fragile heuristic this pipeline avoids.
6. **Optional flux calibration** (`fluxcal.py`, `flux_calibrate.py`) —
   derives an instrumental response curve from spectrophotometric standards
   (`fluxstd/`) and applies it to convert reduced (relative, ADU) spectra to
   physical units (erg/s/cm²/Å). No airmass/extinction correction is
   possible with this dataset (no headers carry it) — every calibrated
   output's header says so explicitly.

## Night-directory layout

The night/run directory is the toplevel: every stage assumes it's the
current directory and reads/writes a fixed subdirectory of it, so a night
stays self-contained and its stages don't get lost track of:

```
20260823/
├── raw/                raw acquired frames (science/dark/comp) - input only
├── sets/                manifests - see below
│   ├── list                 ordered list of target names to coadd
│   ├── <name>               one per target, tab-separated frame list
│   ├── comp                 comp-frame manifest (for header-untagged nights)
│   ├── darks                dark-frame manifest (ditto)
│   └── standards            which set is which flux standard - see below
├── coadded/            coadd_sets.py output (optional stage)
├── reduced/            reduce_flores.py output
└── fluxcal/            flux_calibrate.py output
```

`sets/comp` and `sets/darks` play a different role from the rest of
`sets/`: they identify calibration frames by an explicit filename list
(for nights where `IMAGETYP` doesn't tag them correctly), not a target to
combine. Everything else in `sets/` is a per-target frame list whose
filename becomes the stem of that target's combined output
(`sets/vega` → `coadded/vega_3x1s.fits` → `reduced/vega_3x1s_fiber-*.dat`).

## Usage

`cd` into the night directory and run each stage with no arguments - every
default is relative to the current directory per the layout above:

```
cd 20260823
python3 /path/to/coadd_sets.py                       # sets/ + raw/ -> coadded/
python3 /path/to/reduce_flores.py                     # raw/ (+ coadded/ if present) -> reduced/
python3 /path/to/flux_calibrate.py                    # reduced/ + coadded/ + raw/ + sets/standards -> fluxcal/
```

`reduce_flores.py` reads comps/darks only from `raw/` but automatically
pulls in `coadded/`'s frames as additional science input if that directory
exists - a coadded product carries no comp/dark of its own (see
`coadd_light.py`), so there's no need to symlink it into `raw/` (which
would make it indistinguishable from a raw frame - e.g. eligible for
dark-subtraction it's already had) or pass `--comp-dir` by hand.

Every path is still overridable and independent of the others (`--out`,
`--sets`, `--comp-dir`, `--coadd-dir`, `--raw-dir`, `--standards`, ...) for nights that
don't follow this layout, or a raw location that lives elsewhere (e.g.
`/images/2026/20260823` on an imaging server) - see each script's `-h`.

Nights where comp frames aren't tagged correctly in `IMAGETYP` (same
manifest convention for `--darks`):

```
python3 reduce_flores.py raw --comps comp1.fits comp2.fits
# or, using sets/comp directly (this is also reduce_flores.py's default
# when sets/comp exists and --comps isn't given):
python3 reduce_flores.py raw --comps sets/comp
```

## Flux standard pairing

`flux_calibrate.py` needs to know which reduced target *is* a flux
standard, and which known standard (from `fluxstd/`) it is. This is never
guessed from the target name - it's declared in `sets/standards`, one line
per standard: `<target_name> <standard_key>` (whitespace-separated, `#`
comments/blank lines ignored), e.g.:

```
vega    vega
hr7596  hr7596
```

`<target_name>` is the leading token of the reduced `.dat` stem (e.g.
`etauma` out of `etauma_6x5s_fiber-untilted.dat`); `<standard_key>` is one
of the keys in `flux_calibrate.py`'s `STANDARDS` dict (`etauma`,
`zetacas`, `vega`, `hr7596`). This file is required - a night observing a
standard under a different name, or a new standard added to `STANDARDS`,
just needs one line here, not a renamed set.

Run any script with `-h` for the full set of options — each has a detailed
module docstring covering the reasoning behind its defaults and edge cases.

## Layout

```
reduce_flores.py     driver: raw frames -> calibrated spectra
frames.py             FITS classification, dark handling
trace.py              trace finding, tilt correction, optimal extraction
wavecal.py            bootstrapped wavelength calibration
noise.py              per-pixel error estimation
coadd_light.py         combine raw sub-exposures of one frame
coadd_sets.py          drive coadd_light.py from a per-night sets/ directory
fluxcal.py             flux calibration from spectrophotometric standards
flux_calibrate.py      CLI driver for fluxcal.py
common.py              small shared numerical utilities
seeds/                 one-time seed line list + the script that built it
fluxstd/               reference spectra for flux standards (Vega, HR7596, ...)
thar_lovis_pepe_clean.csv   ThAr atlas used to grow the wavelength solution
```

## Status

This is a working pipeline shared for student use; interfaces may still
change. Questions/issues welcome.
