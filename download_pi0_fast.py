#!/usr/bin/env python3
"""Download pi0-fast-droid checkpoint."""

import sys
import os
from pathlib import Path

# Add openpi to path
openpi_path = Path(__file__).parent / "openpi" / "src"
sys.path.insert(0, str(openpi_path))

from openpi.shared import download

# Set download location (will create openpi-assets/checkpoints/pi0_fast_droid inside)
os.environ['OPENPI_DATA_HOME'] = '/data1/zbs_files/data/HF/hub'

print('='*70)
print('Downloading pi0-droid checkpoint')
print('Size: ~19GB')
print('Location: /data1/zbs_files/data/HF/hub/openpi-assets/checkpoints/')
print('='*70)
print()

try:
    path = download.maybe_download('gs://openpi-assets/checkpoints/pi0_droid')
    print()
    print('='*70)
    print(f'✓ Download completed!')
    print(f'✓ Checkpoint saved to: {path}')
    print('='*70)
except Exception as e:
    print()
    print('='*70)
    print(f'✗ Download failed: {e}')
    print('='*70)
    sys.exit(1)
