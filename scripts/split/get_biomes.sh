#!/usr/bin/env bash
# Downloads the IBGE biome polygons (Biomas do Brasil 1:250.000, 2019 edition) into data/biomes.
set -euo pipefail
URL=https://geoftp.ibge.gov.br/informacoes_ambientais/estudos_ambientais/biomas/vetores/Biomas_250mil.zip
mkdir -p data/biomes
[ -s data/biomes/Biomas_250mil.zip ] || curl -fsSL -o data/biomes/Biomas_250mil.zip "$URL"
echo '1d8a4d860d9c203b10ef41dd0701ced2e389ad86442e23a815a7cccdc98ca0c5  data/biomes/Biomas_250mil.zip' | sha256sum -c -
unzip -oq data/biomes/Biomas_250mil.zip -d data/biomes
