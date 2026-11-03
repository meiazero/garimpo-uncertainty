#!/usr/bin/env bash
# Downloads Jasansky et al. (2024) metadata; the Sentinel-2 images are not part of the archive.
set -euo pipefail
mkdir -p data/zenodo
cd data/zenodo
curl -sSL -o mining_area_data.zip 'https://zenodo.org/records/14195737/files/mining_area_data.zip?download=1'
echo '1259ad8045ae648c547ef3db1fccab3b  mining_area_data.zip' | md5sum -c -
unzip -oq mining_area_data.zip mining_area_data.gpkg
