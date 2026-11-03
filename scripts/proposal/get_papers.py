"""Downloads the open-access references to papers/; writes papers/INDEX.tsv for checking titles."""

import json
import pathlib
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

PAPERS = pathlib.Path("papers")
AGENT = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

ARXIV = {
    "cheung2026compass": "2509.22240",
    "singh2024conformal-eo": "2401.06421",
    "mossina2024conformal-segmentation": "2405.05145",
    "mossina2025morphological-sets": "2503.05618",
    "viti2026consign": "2505.14113",
    "wundram2024performance-range": "2407.13307",
    "cosarinsky2026confic-rca": "2503.04522",
    "lehmann2026calibration-geofm": "2608.16614",
    "gonzalez2026shrug-fm": "2511.10370",
    "cui2026eldor": "2605.15397",
    "lu2025regression-remote-sensing": "2407.13659",
    "angelopoulos2023ppi": "2301.09633",
    "shirota2026ppi-spatial": "2608.10356",
    "guo2017calibration": "1706.04599",
    "lakshminarayanan2017ensembles": "1612.01474",
    "monteiro2020ssn": "2006.06015",
    "kohl2018probabilistic-unet": "1806.05034",
    "lei2018distribution-free": "1604.04173",
    "ronneberger2015unet": "1505.04597",
    "tibshirani2019weighted-conformal": "1904.06019",
    "barber2023beyond-exchangeability": "2202.13415",
    "romano2019cqr": "1905.03222",
    "teng2022feature-conformal": "2210.00173",
    "angelopoulos2021gentle-intro": "2107.07511",
    "angelopoulos2022risk-control": "2208.02814",
    "chen2018deeplabv3plus": "1802.02611",
    "dosovitskiy2020vit": "2010.11929",
    "kirillov2023sam": "2304.02643",
    "ravi2024sam2": "2408.00714",
    "dunn2022hierarchical-conformal": "1809.07441",
    "xie2021segformer": "2105.15203",
}

DOI = {
    "maus2020mining-dataset": "10.1038/s41597-020-00624-w",
    "tang2023mining-footprint": "10.1038/s43247-023-00805-6",
    "jasansky2025tropics": "10.1038/s41893-025-01668-9",
    "couttenier2022asm-west-africa": "10.1371/journal.pone.0267963",
    "olofsson2014good-practices": "10.1016/j.rse.2014.02.015",
    "camalan2022amazon-change": "10.3390/rs14071746",
    "queimadas-multisensor-active-fire2021": "10.3390/rs13194005",
    "queimadas-cerrado-dataset2025": "10.1080/20964471.2025.2463731",
}

# Titles whose DOI or arXiv id was not settled in the session; resolved through OpenAlex search.
TITLES = {
    "smallminesds2025": (
        "SmallMinesDS: A Multimodal Dataset for Mapping Artisanal and Small-Scale Gold Mines"
    ),
    "queimadas-xingu-products2026": (
        "Assessment of Burned Area Products During the Severe Fire Season in the Xingu River Basin"
    ),
    "sentinela-ia2025": (
        "Sentinela-IA: Detecção de Garimpo Ilegal na Amazônia com Deep Learning "
        "e Imagens Sentinel-2"
    ),
}


def get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers=AGENT)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def save_pdf(slug: str, url: str) -> str:
    target = PAPERS / f"{slug}.pdf"
    if target.exists():
        return "already"
    data = get(url)
    if not data.startswith(b"%PDF"):
        return "not-a-pdf"
    target.write_bytes(data)
    return "ok"


def openalex_urls(work: dict) -> list[str]:
    urls = [(work.get("best_oa_location") or {}).get("pdf_url")]
    urls += [location.get("pdf_url") for location in work.get("locations") or []]
    return [url for url in dict.fromkeys(urls) if url]


def from_openalex(slug: str, work: dict) -> str:
    for url in openalex_urls(work):
        try:
            if save_pdf(slug, url) in ("ok", "already"):
                return f"ok {url}"
        except (OSError, ValueError):
            continue
    return "no-open-pdf"


def first_lines(path: pathlib.Path) -> str:
    text = subprocess.run(
        ["pdftotext", "-l", "1", str(path), "-"], capture_output=True, text=True, check=False
    ).stdout
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 15]
    return " / ".join(lines[:2])[:160]


def main() -> None:
    PAPERS.mkdir(exist_ok=True)
    status: dict[str, str] = {}
    for slug, arxiv_id in ARXIV.items():
        try:
            status[slug] = save_pdf(slug, f"https://arxiv.org/pdf/{arxiv_id}")
        except OSError as error:
            status[slug] = f"failed {error}"
        print(slug, status[slug], flush=True)
        if status[slug] != "already":
            time.sleep(3.5)  # arXiv asks automated clients to pace requests
    for slug, doi in DOI.items():
        if (PAPERS / f"{slug}.pdf").exists():
            status[slug] = "already"
            continue
        try:
            work = json.loads(get(f"https://api.openalex.org/works/doi:{doi}"))
            status[slug] = from_openalex(slug, work)
        except (OSError, ValueError) as error:
            status[slug] = f"failed {error}"
        print(slug, status[slug], flush=True)
    for slug, title in TITLES.items():
        try:
            query = urllib.parse.quote(title)
            results = json.loads(get(f"https://api.openalex.org/works?search={query}&per-page=1"))
            work = (results.get("results") or [{}])[0]
            status[slug] = from_openalex(slug, work) + f" | matched: {work.get('title')}"
        except (OSError, ValueError) as error:
            status[slug] = f"failed {error}"
        print(slug, status[slug], flush=True)
    with open(PAPERS / "INDEX.tsv", "w") as index:
        index.write("file\tstatus\tfirst_lines\n")
        for slug, state in status.items():
            path = PAPERS / f"{slug}.pdf"
            lines = first_lines(path) if path.exists() else ""
            index.write(f"{slug}\t{state}\t{lines}\n")


if __name__ == "__main__":
    main()
