#!/usr/bin/env bash
# Rebuild every file under data/labels from the raw SIMMC 2.1 files.
# The committed files are its output; rerun only to check them or after changing the labeling.
set -euo pipefail
python -m promcr.data.build conditions
python -m promcr.data.build splitters
python -m promcr.data.build synthetic   # about 10 minutes
