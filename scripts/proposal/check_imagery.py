"""Confirms each artisanal tile's Sentinel-2 item exists; writes outputs/imagery.csv."""

import concurrent.futures
import csv
import json
import pathlib
import urllib.error
import urllib.request

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/"


def probe(row: dict[str, str]) -> dict[str, str]:
    item_id = row["s2_tile_id"]
    try:
        with urllib.request.urlopen(STAC + item_id, timeout=60) as response:
            item = json.load(response)
        return {
            "tile_id": row["tile_id"],
            "region": row["region"],
            "item": item_id,
            "found": "1",
            "cloud": str(item["properties"].get("eo:cloud_cover", "")),
        }
    except urllib.error.HTTPError as error:
        status = "0" if error.code == 404 else f"http{error.code}"
    except OSError as error:
        status = f"error:{type(error).__name__}"
    return {
        "tile_id": row["tile_id"],
        "region": row["region"],
        "item": item_id,
        "found": status,
        "cloud": "",
    }


def main() -> None:
    with open("outputs/tiles_artisanal.csv", newline="") as source:
        rows = list(csv.DictReader(source))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(probe, rows))
    with open(pathlib.Path("outputs/imagery.csv"), "w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    counts: dict[tuple[str, str], int] = {}
    for result in results:
        key = (result["region"], result["found"])
        counts[key] = counts.get(key, 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
