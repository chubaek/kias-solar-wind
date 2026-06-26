# Running `run_magnetogram_stage2_full.py`

Use the existing virtual environment from the repository root:

```bash
cd /home/sibaekyi/kias_group_work/kias-solar-wind
.venv/bin/python run_magnetogram_stage2_full.py --help
```

The current `.venv` is Python 3.9 and has the packages needed by this script:
`numpy`, `pandas`, `scikit-learn`, `scipy`, `astropy`, `drms`, `sunpy`, and
`torch`.

If `.venv` is missing, rebuild it with pip:

```bash
cd /home/sibaekyi/kias_group_work/kias-solar-wind
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements-magnetogram-stage2.txt
```

JSOC downloads require an email address:

```bash
export JSOC_EMAIL="your_email@example.com"
```

Recommended commands:

```bash
# Check planned JSOC requests without downloading.
.venv/bin/python run_magnetogram_stage2_full.py --dry-run --start-date 2011-01-01 --end-date 2025-12-31

# Smoke-test one month of downloads.
.venv/bin/python run_magnetogram_stage2_full.py --download --start-date 2011-01-01 --end-date 2025-12-31 --max-months 1

# Validate downloaded raw FITS files for one month.
.venv/bin/python run_magnetogram_stage2_full.py --validate-raw --start-date 2011-01-01 --end-date 2025-12-31 --max-months 1

# Run full download and feature extraction.
.venv/bin/python run_magnetogram_stage2_full.py --download --extract --start-date 2011-01-01 --end-date 2025-12-31

# Run stage-2 evaluation after features exist.
.venv/bin/python run_magnetogram_stage2_full.py --evaluate --start-date 2011-01-01 --end-date 2025-12-31
```

Outputs are written under:

```text
data/magnetograms/
outputs/magnetogram_ch_features_72h/
```

The full 2011-2025 run downloads thousands of HMI FITS files, so expect it to
take substantial time and disk space.
