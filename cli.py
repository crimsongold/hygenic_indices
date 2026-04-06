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


# ---------------------------------------------------------------------------
# Root command group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Hygenic Indices — S&P methodology implementations."""


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
# Formatting helpers
# ---------------------------------------------------------------------------


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
