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

## Usage

Basic reduction of one night:

```
python3 reduce_flores.py /path/to/raw_dir
```

Output defaults to `~/flores/reduced/<basename of raw_dir>`; override with
`--out`. Each `<science_stem>_fiber-<tag>.dat` file has three columns:
wavelength (Å), flux, error.

Nights where comp frames aren't tagged correctly in `IMAGETYP`:

```
python3 reduce_flores.py /path/to/raw_dir --comps comp1.fits comp2.fits
# or, using a manifest file (one filename per line):
python3 reduce_flores.py /path/to/raw_dir --comps comp
```

Coadding sub-exposures before reduction:

```
python3 coadd_sets.py /path/to/raw_dir --out /path/to/coadd_dir
python3 reduce_flores.py /path/to/coadd_dir --comp-dir /path/to/raw_dir --comps comp
```

Flux-calibrating a reduced night:

```
python3 flux_calibrate.py /path/to/reduced_dir
```

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
coadd_sets.py          drive coadd_light.py from a per-night `sets` manifest
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
