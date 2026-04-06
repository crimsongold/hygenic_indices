"""
CLI entry point for the Hygenic Indices toolkit.

Usage:
    python cli.py [INDEX] [OPTIONS]

Examples:
    python cli.py sp-carbon-aware rebalance --input universe.csv
    python cli.py sp-carbon-aware rebalance --input universe.csv --universe-type emerging
    python cli.py sp-carbon-aware show-universe --input universe.csv
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

# Default location for methodology PDFs (relative to this file)
_METHODOLOGIES_DIR = Path(__file__).parent / "methodologies"


# ---------------------------------------------------------------------------
# Root command group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Hygenic Indices — S&P methodology implementations."""


# ---------------------------------------------------------------------------
# Methodology downloader sub-group
# ---------------------------------------------------------------------------


@main.group("download")
def download_group() -> None:
    """Download S&P DJI methodology PDFs from the public S&P website."""


@download_group.command("list")
@click.option(
    "--dest", "dest_dir",
    type=click.Path(path_type=Path),
    default=_METHODOLOGIES_DIR,
    show_default=True,
    help="Directory to check for locally saved PDFs.",
)
def cmd_download_list(dest_dir: Path) -> None:
    """List all methodology PDFs in the catalog and their local status."""
    from methodologies.downloader import list_methodologies

    entries = list_methodologies(dest_dir)

    table = Table(title="Methodology Catalog", show_lines=False)
    table.add_column("Slug", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Filename", style="white")
    table.add_column("Local", style="white")

    for e in entries:
        status = "[green]yes[/green]" if e["present"] else "[yellow]no[/yellow]"
        table.add_row(e["slug"], e["name"], e["filename"], status)

    console.print(table)


@download_group.command("fetch")
@click.option(
    "--index", "slug",
    default=None,
    help="Slug of the methodology to download (see 'download list').",
)
@click.option(
    "--all", "download_all",
    is_flag=True,
    default=False,
    help="Download all methodologies in the catalog.",
)
@click.option(
    "--dest", "dest_dir",
    type=click.Path(path_type=Path),
    default=_METHODOLOGIES_DIR,
    show_default=True,
    help="Directory to save PDFs into.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-download even if the file already exists.",
)
def cmd_download_fetch(
    slug: str | None,
    download_all: bool,
    dest_dir: Path,
    force: bool,
) -> None:
    """Download one or all methodology PDFs from the S&P DJI website."""
    from methodologies.downloader import (
        download_all as _download_all,
        download_methodology,
        METHODOLOGY_CATALOG,
    )

    if not slug and not download_all:
        raise click.UsageError("Specify --index <slug> or --all.")

    from methodologies.downloader import DownloadBlockedError

    def _handle_blocked(blocked_url: str) -> None:
        console.print(
            "[yellow]  Blocked by CDN (403).[/yellow] S&P's website requires a browser session.\n"
            f"  Please open this URL in your browser and save the PDF manually:\n"
            f"  [cyan]{blocked_url}[/cyan]"
        )

    if download_all:
        console.print(f"[bold]Downloading all {len(METHODOLOGY_CATALOG)} methodologies to {dest_dir}[/bold]")
        blocked_urls: list[str] = []
        for _slug, meta in METHODOLOGY_CATALOG.items():
            try:
                download_methodology(_slug, dest_dir, force=force, console=console)
            except KeyError as exc:
                raise click.ClickException(str(exc)) from exc
            except DownloadBlockedError as exc:
                url = str(exc)
                _handle_blocked(url)
                blocked_urls.append(url)
            except Exception as exc:
                console.print(f"  [red]Error[/red] downloading {_slug}: {exc}")
        if blocked_urls:
            console.print(
                f"\n[yellow]{len(blocked_urls)} PDF(s) require manual download.[/yellow] "
                "Save them to the [bold]methodologies/[/bold] directory."
            )
        else:
            console.print("[green]Done.[/green]")
    else:
        console.print(f"[bold]Downloading[/bold] {slug} to {dest_dir}")
        try:
            download_methodology(slug, dest_dir, force=force, console=console)
            console.print("[green]Done.[/green]")
        except KeyError as exc:
            raise click.ClickException(str(exc)) from exc
        except DownloadBlockedError as exc:
            _handle_blocked(str(exc))
        except Exception as exc:
            raise click.ClickException(f"Download failed: {exc}") from exc


# ---------------------------------------------------------------------------
# S&P Carbon Aware sub-group
# ---------------------------------------------------------------------------


@main.group("sp-carbon-aware")
def sp_carbon_aware() -> None:
    """S&P Carbon Aware Index Series commands."""


@sp_carbon_aware.command("rebalance")
@click.option(
    "--input", "csv_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the universe CSV file.",
)
@click.option(
    "--universe-type",
    type=click.Choice(["developed", "emerging"], case_sensitive=False),
    default="developed",
    show_default=True,
    help="Index variant: 'developed' or 'emerging'.",
)
@click.option(
    "--output", "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to write the rebalance output CSV.",
)
def cmd_rebalance(csv_path: Path, universe_type: str, output_path: Path | None) -> None:
    """
    Run a full index rebalance and print the optimized constituent weights.

    Applies eligibility filters (ESG, business activities, UNGC, controversies)
    then runs the carbon-intensity minimization optimizer.
    """
    from indices.sp_carbon_aware.rebalancer import (
        load_universe_from_csv,
        rebalance,
        result_to_dataframe,
    )

    console.print(f"[bold]Loading universe from:[/bold] {csv_path}")
    universe = load_universe_from_csv(csv_path)
    console.print(f"Universe size: {len(universe.stocks)} stocks")

    console.print(f"\n[bold]Running rebalance[/bold] (universe_type={universe_type})...")
    result = rebalance(universe, universe_type=universe_type)

    # --- Summary ---
    _print_summary(result)

    # --- Constituent weights table ---
    df = result_to_dataframe(result, universe)
    _print_weights_table(df)

    # --- Exclusions table ---
    if result.excluded_tickers:
        _print_exclusions_table(result.excluded_tickers)

    # --- Optional CSV export ---
    if output_path:
        df.to_csv(output_path, index=False)
        console.print(f"\n[green]Output written to:[/green] {output_path}")


@sp_carbon_aware.command("show-universe")
@click.option(
    "--input", "csv_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the universe CSV file.",
)
def cmd_show_universe(csv_path: Path) -> None:
    """Display the raw universe with underlying index weights."""
    from indices.sp_carbon_aware.rebalancer import load_universe_from_csv

    universe = load_universe_from_csv(csv_path)
    df = universe.to_dataframe()

    table = Table(title="Index Universe", show_lines=False)
    for col in ["ticker", "company_name", "country", "gics_industry_group",
                "underlying_weight", "esg_score", "carbon_intensity", "ungc_status"]:
        table.add_column(col, style="cyan" if col == "ticker" else "white")

    for _, row in df.sort_values("underlying_weight", ascending=False).iterrows():
        table.add_row(
            str(row["ticker"]),
            str(row["company_name"])[:30],
            str(row["country"]),
            str(row["gics_industry_group"])[:28],
            f"{row['underlying_weight']:.4%}",
            f"{row['esg_score']:.1f}" if row["esg_score"] == row["esg_score"] else "N/A",
            f"{row['carbon_intensity']:.1f}" if row["carbon_intensity"] == row["carbon_intensity"] else "N/A",
            str(row["ungc_status"]),
        )

    console.print(table)
    console.print(f"\nTotal stocks: {len(df)}")


# ---------------------------------------------------------------------------
# S&P ESG (Scored & Screened) sub-group
# ---------------------------------------------------------------------------


@main.group("sp-esg")
def sp_esg() -> None:
    """S&P ESG (Scored & Screened) Index Series commands."""


@sp_esg.command("rebalance")
@click.option("--input", "csv_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
def cmd_sp_esg_rebalance(csv_path: Path, output_path: Path | None) -> None:
    """Run a full S&P ESG rebalance."""
    from indices.sp_esg.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe

    universe = load_universe_from_csv(csv_path)
    console.print(f"[bold]Universe:[/bold] {len(universe.stocks)} stocks")
    result = rebalance(universe)
    _print_generic_summary(result, "rebalanced_weights")
    if result.excluded_tickers:
        _print_exclusions_table(result.excluded_tickers)
    if output_path:
        result_to_dataframe(result, universe).to_csv(output_path, index=False)
        console.print(f"[green]Output written to:[/green] {output_path}")


@sp_esg.command("show-universe")
@click.option("--input", "csv_path", required=True, type=click.Path(exists=True, path_type=Path))
def cmd_sp_esg_show(csv_path: Path) -> None:
    """Display the raw S&P ESG universe."""
    from indices.sp_esg.rebalancer import load_universe_from_csv
    _show_generic_universe(load_universe_from_csv(csv_path))


# ---------------------------------------------------------------------------
# S&P Carbon Efficient sub-group
# ---------------------------------------------------------------------------


@main.group("sp-carbon-efficient")
def sp_carbon_efficient() -> None:
    """S&P Global Carbon Efficient Index Series commands."""


@sp_carbon_efficient.command("rebalance")
@click.option("--input", "csv_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
def cmd_sp_ce_rebalance(csv_path: Path, output_path: Path | None) -> None:
    """Run a full carbon efficiency tilt rebalance."""
    from indices.sp_carbon_efficient.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe

    universe = load_universe_from_csv(csv_path)
    console.print(f"[bold]Universe:[/bold] {len(universe.stocks)} stocks")
    result = rebalance(universe)

    from rich.panel import Panel
    lines = [
        f"[bold]WACI (tilted):[/bold]     {result.weighted_avg_carbon_intensity:.2f}",
        f"[bold]WACI (underlying):[/bold] {result.underlying_waci:.2f}",
        f"[bold]WACI reduction:[/bold]    {result.waci_reduction_pct:.1f}%",
    ]
    console.print(Panel("\n".join(lines), title="Rebalance Summary", expand=False))

    if output_path:
        result_to_dataframe(result, universe).to_csv(output_path, index=False)
        console.print(f"[green]Output written to:[/green] {output_path}")


@sp_carbon_efficient.command("show-universe")
@click.option("--input", "csv_path", required=True, type=click.Path(exists=True, path_type=Path))
def cmd_sp_ce_show(csv_path: Path) -> None:
    """Display the raw Carbon Efficient universe."""
    from indices.sp_carbon_efficient.rebalancer import load_universe_from_csv
    _show_generic_universe(load_universe_from_csv(csv_path))


# ---------------------------------------------------------------------------
# DJSI Diversified sub-group
# ---------------------------------------------------------------------------


@main.group("djsi-diversified")
def djsi_diversified() -> None:
    """Dow Jones Sustainability Diversified Indices commands."""


@djsi_diversified.command("rebalance")
@click.option("--input", "csv_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
def cmd_djsi_rebalance(csv_path: Path, output_path: Path | None) -> None:
    """Run a full DJSI Diversified rebalance."""
    from indices.djsi_diversified.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe

    universe = load_universe_from_csv(csv_path)
    console.print(f"[bold]Universe:[/bold] {len(universe.stocks)} stocks")
    result = rebalance(universe)
    _print_generic_summary(result, "rebalanced_weights")
    if result.excluded_tickers:
        _print_exclusions_table(result.excluded_tickers)
    if output_path:
        result_to_dataframe(result, universe).to_csv(output_path, index=False)
        console.print(f"[green]Output written to:[/green] {output_path}")


@djsi_diversified.command("show-universe")
@click.option("--input", "csv_path", required=True, type=click.Path(exists=True, path_type=Path))
def cmd_djsi_show(csv_path: Path) -> None:
    """Display the raw DJSI Diversified universe."""
    from indices.djsi_diversified.rebalancer import load_universe_from_csv
    _show_generic_universe(load_universe_from_csv(csv_path))


# ---------------------------------------------------------------------------
# S&P PACT (PAB ESG / CTB) sub-group
# ---------------------------------------------------------------------------


@main.group("sp-pact")
def sp_pact() -> None:
    """S&P PAB ESG & S&P CTB Indices commands."""


@sp_pact.command("rebalance")
@click.option("--input", "csv_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--variant",
    type=click.Choice(["ctb", "pab"], case_sensitive=False),
    default="ctb",
    show_default=True,
    help="Index variant: 'ctb' (Climate Transition) or 'pab' (Paris-Aligned).",
)
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=None)
def cmd_sp_pact_rebalance(csv_path: Path, variant: str, output_path: Path | None) -> None:
    """Run a full PACT rebalance with constrained optimization."""
    from indices.sp_pact.models import Variant
    from indices.sp_pact.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe

    v = Variant(variant.lower())
    universe = load_universe_from_csv(csv_path)
    console.print(f"[bold]Universe:[/bold] {len(universe.stocks)} stocks  [bold]Variant:[/bold] {v.value.upper()}")
    result = rebalance(universe, variant=v)

    _print_summary(result)
    if result.excluded_tickers:
        _print_exclusions_table(result.excluded_tickers)
    if output_path:
        result_to_dataframe(result, universe).to_csv(output_path, index=False)
        console.print(f"[green]Output written to:[/green] {output_path}")


@sp_pact.command("show-universe")
@click.option("--input", "csv_path", required=True, type=click.Path(exists=True, path_type=Path))
def cmd_sp_pact_show(csv_path: Path) -> None:
    """Display the raw PACT universe."""
    from indices.sp_pact.rebalancer import load_universe_from_csv
    _show_generic_universe(load_universe_from_csv(csv_path))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _print_generic_summary(result, weights_attr: str) -> None:
    """Print a summary panel for indices without WACI diagnostics."""
    from rich.panel import Panel

    weights = getattr(result, weights_attr, {})
    lines = [
        f"[bold]Selected stocks:[/bold]  {len(weights)}",
        f"[bold]Excluded stocks:[/bold] {len(result.excluded_tickers)}",
    ]
    console.print(Panel("\n".join(lines), title="Rebalance Summary", expand=False))


def _show_generic_universe(universe) -> None:
    """Display a generic universe table."""
    df = universe.to_dataframe()
    table = Table(title="Index Universe", show_lines=False)
    cols = [c for c in ["ticker", "company_name", "country", "gics_sector", "underlying_weight"] if c in df.columns]
    for col in cols:
        table.add_column(col, style="cyan" if col == "ticker" else "white")
    for _, row in df.sort_values("underlying_weight", ascending=False).head(30).iterrows():
        vals = []
        for col in cols:
            v = row[col]
            if col == "underlying_weight":
                vals.append(f"{v:.4%}")
            elif isinstance(v, str):
                vals.append(v[:30])
            else:
                vals.append(str(v))
        table.add_row(*vals)
    console.print(table)
    console.print(f"\nTotal stocks: {len(df)}")


def _print_summary(result) -> None:
    from rich.panel import Panel

    lines = [
        f"[bold]Solver status:[/bold]     {result.solver_status}",
        f"[bold]Relaxation level:[/bold]  {result.relaxation_level}",
        f"[bold]Eligible stocks:[/bold]   {len(result.eligible_tickers)}",
        f"[bold]Excluded stocks:[/bold]   {len(result.excluded_tickers)}",
        f"[bold]WACI (optimized):[/bold]  {result.weighted_avg_carbon_intensity:.2f} tCO2e/$M rev",
        f"[bold]WACI (underlying):[/bold] {result.underlying_weighted_avg_carbon_intensity:.2f} tCO2e/$M rev",
    ]

    if (
        result.underlying_weighted_avg_carbon_intensity > 0
        and result.weighted_avg_carbon_intensity == result.weighted_avg_carbon_intensity
    ):
        reduction = (
            1 - result.weighted_avg_carbon_intensity
            / result.underlying_weighted_avg_carbon_intensity
        ) * 100
        lines.append(f"[bold]WACI reduction:[/bold]   {reduction:.1f}%")

    console.print(Panel("\n".join(lines), title="Rebalance Summary", expand=False))


def _print_weights_table(df) -> None:
    table = Table(title="Optimized Constituents", show_lines=False)
    for col, style in [
        ("ticker", "cyan"),
        ("company_name", "white"),
        ("country", "white"),
        ("gics_industry_group", "white"),
        ("underlying_weight", "white"),
        ("optimized_weight", "green"),
        ("active_weight", "yellow"),
        ("carbon_intensity", "white"),
    ]:
        table.add_column(col, style=style)

    for _, row in df.iterrows():
        active = row["active_weight"]
        active_str = f"[green]+{active:.4%}[/green]" if active >= 0 else f"[red]{active:.4%}[/red]"
        ci = row["carbon_intensity"]
        table.add_row(
            str(row["ticker"]),
            str(row["company_name"])[:28],
            str(row["country"]),
            str(row["gics_industry_group"])[:26],
            f"{row['underlying_weight']:.4%}",
            f"{row['optimized_weight']:.4%}",
            active_str,
            f"{ci:.1f}" if ci == ci else "N/A",  # NaN check
        )

    console.print(table)


def _print_exclusions_table(excluded: dict[str, str]) -> None:
    table = Table(title="Excluded Securities", show_lines=False)
    table.add_column("Ticker", style="red")
    table.add_column("Reason", style="white")

    for ticker, reason in sorted(excluded.items()):
        table.add_row(ticker, reason)

    console.print(table)


if __name__ == "__main__":
    main()
