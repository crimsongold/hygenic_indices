"""
Methodology PDF downloader for S&P Dow Jones Indices.

All methodology documents are publicly available from the S&P DJI website.
This module maintains a catalog of known URLs and provides functions to
list and download them into the local methodologies/ directory.
"""

from __future__ import annotations

from pathlib import Path

import requests
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/"

# Each entry: slug → {name, filename}
# The download URL is constructed as _BASE_URL + filename.
METHODOLOGY_CATALOG: dict[str, dict[str, str]] = {
    "sp-carbon-aware": {
        "name": "S&P Carbon Aware Index Series",
        "filename": "methodology-sp-carbon-aware-index-series.pdf",
    },
    "sp-esg": {
        "name": "S&P Scored & Screened Index Series",
        "filename": "methodology-sp-ss-index-series.pdf",
    },
    "sp-pact": {
        "name": "S&P PAB ESG & S&P CTB Indices",
        "filename": "methodology-sp-pab-esg-sp-ctb-indices.pdf",
    },
    "sp-carbon-efficient": {
        "name": "S&P Global Carbon Efficient Index Series",
        "filename": "methodology-sp-global-carbon-efficient-index-series.pdf",
    },
    "djsi-diversified": {
        "name": "Dow Jones Sustainability Diversified Indices",
        "filename": "methodology-dj-sustainability-diversified-indices.pdf",
    },
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def list_methodologies(dest_dir: Path) -> list[dict[str, str]]:
    """
    Return catalog entries enriched with local presence status.

    Each dict has keys: slug, name, filename, url, local_path, present.
    """
    entries = []
    for slug, meta in METHODOLOGY_CATALOG.items():
        local_path = dest_dir / meta["filename"]
        entries.append({
            "slug": slug,
            "name": meta["name"],
            "filename": meta["filename"],
            "url": _BASE_URL + meta["filename"],
            "local_path": str(local_path),
            "present": local_path.exists(),
        })
    return entries


def download_methodology(
    slug: str,
    dest_dir: Path,
    *,
    force: bool = False,
    console: Console | None = None,
) -> Path:
    """
    Download the methodology PDF for *slug* into *dest_dir*.

    Returns the path to the saved file.
    Raises KeyError if slug is not in the catalog.
    Raises requests.HTTPError on HTTP errors.
    Skips download if file already exists, unless force=True.
    """
    if slug not in METHODOLOGY_CATALOG:
        raise KeyError(f"Unknown methodology slug: {slug!r}. Run 'download list' to see available slugs.")

    meta = METHODOLOGY_CATALOG[slug]
    url = _BASE_URL + meta["filename"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / meta["filename"]

    if dest_path.exists() and not force:
        if console:
            console.print(f"  [yellow]Skipping[/yellow] {slug} — already present at {dest_path}")
        return dest_path

    _download_file(url, dest_path, label=meta["name"], console=console)
    return dest_path


def download_all(
    dest_dir: Path,
    *,
    force: bool = False,
    console: Console | None = None,
) -> list[Path]:
    """Download every methodology in the catalog. Returns list of saved paths."""
    paths = []
    for slug in METHODOLOGY_CATALOG:
        path = download_methodology(slug, dest_dir, force=force, console=console)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.spglobal.com/spdji/en/",
}


class DownloadBlockedError(Exception):
    """Raised when the server returns 403/401, typically due to bot-protection."""


def _download_file(url: str, dest: Path, *, label: str, console: Console | None) -> None:
    """
    Stream-download *url* to *dest*, showing a Rich progress bar.

    S&P's CDN may require a real browser session (JavaScript challenge).
    In that case a DownloadBlockedError is raised with the manual URL.
    """
    _console = console or Console()

    try:
        resp = requests.get(url, headers=_HEADERS, stream=True, timeout=60)
    except requests.RequestException as exc:
        raise requests.RequestException(f"Network error fetching {url}: {exc}") from exc

    if resp.status_code in (401, 403):
        resp.close()
        raise DownloadBlockedError(url)

    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0)) or None

    with resp, Progress(
        TextColumn(f"[bold cyan]{label}[/bold cyan]"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=_console,
        transient=True,
    ) as progress:
        task = progress.add_task("download", total=total)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
                    progress.advance(task, len(chunk))

    _console.print(f"  [green]Saved[/green] {dest}")
