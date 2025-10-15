"""Main entry point for the counter-surveillance analysis pipeline."""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.orchestration.langchain_pipeline import create_pipeline
from src.config.pipeline_config import AnalysisScenario
from src.config.logger import logger

console = Console()


def parse_args():
    """
    Parse command-line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Agentic Counter-Surveillance Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py Berlin
  python main.py Athens --country GR --scenario full
  python main.py Hamburg --skip-analyze
  python main.py --data-path data.json --skip-scrape Berlin
        """,
    )

    # Positional arguments
    parser.add_argument("city", help="City name to analyze")

    # Optional arguments
    parser.add_argument(
        "--country",
        help="Country code (e.g., DE, GR, US)",
        default=None,
    )
    parser.add_argument(
        "--scenario",
        choices=[s.value for s in AnalysisScenario],
        default=AnalysisScenario.BASIC.value,
        help="Analysis scenario preset (default: basic)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for results",
        default=None,
    )
    parser.add_argument(
        "--data-path",
        help="Path to existing data file (skips scraping)",
        default=None,
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip data scraping step",
    )
    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip analysis step",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def display_results(results: dict):
    """
    Display pipeline results in a formatted table.

    :param results: Pipeline execution results
    """
    # Status panel
    status = results["status"]
    status_color = {
        "completed": "green",
        "partial": "yellow",
        "failed": "red",
    }.get(status, "white")

    console.print(
        Panel(
            f"[bold {status_color}]{status.upper()}[/bold {status_color}]",
            title="Pipeline Status",
            border_style=status_color,
        )
    )

    # Summary table
    table = Table(title="Execution Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("City", results.get("city", "N/A"))
    table.add_row("Country", results.get("country", "N/A"))
    table.add_row("Scenario", results.get("scenario", "N/A"))
    table.add_row("Duration", f"{results.get('duration_seconds', 0):.2f}s")

    # Scraping results
    if "scrape" in results:
        scrape = results["scrape"]
        if scrape.get("skipped"):
            table.add_row("Scraping", "[yellow]Skipped[/yellow]")
        elif scrape.get("success"):
            elements = scrape.get("elements_count", 0)
            cached = " (cached)" if scrape.get("from_cache") else ""
            table.add_row("Elements Scraped", f"[green]{elements}{cached}[/green]")
        else:
            table.add_row(
                "Scraping", f"[red]Failed: {scrape.get('error', 'Unknown')}[/red]"
            )

    # Analysis results
    if "analyze" in results:
        analyze = results["analyze"]
        if analyze.get("skipped"):
            table.add_row("Analysis", "[yellow]Skipped[/yellow]")
        elif analyze.get("success"):
            elements = analyze.get("element_count", 0)
            table.add_row("Elements Analyzed", f"[green]{elements}[/green]")
        else:
            table.add_row(
                "Analysis", f"[red]Failed: {analyze.get('error', 'Unknown')}[/red]"
            )

    console.print(table)

    # Output files
    if "analyze" in results and results["analyze"].get("success"):
        analyze = results["analyze"]
        files_table = Table(title="Generated Files", show_header=False)
        files_table.add_column("Type", style="cyan")
        files_table.add_column("Path", style="green")

        if analyze.get("output_path"):
            files_table.add_row("Enriched Data", str(analyze["output_path"]))
        if analyze.get("geojson_path"):
            files_table.add_row("GeoJSON", str(analyze["geojson_path"]))
        if analyze.get("heatmap_path"):
            files_table.add_row("Heatmap", str(analyze["heatmap_path"]))
        if analyze.get("hotspots_path"):
            files_table.add_row("Hotspots Data", str(analyze["hotspots_path"]))
        if analyze.get("plot_hotspots"):
            files_table.add_row("Hotspots Plot", str(analyze["plot_hotspots"]))
        if analyze.get("chart_path"):
            files_table.add_row("Statistics Chart", str(analyze["chart_path"]))

        console.print(files_table)

    # Errors
    if "errors" in results and results["errors"]:
        console.print("\n[bold red]Errors:[/bold red]")
        for error in results["errors"]:
            console.print(f"  - {error}")


def main():
    """
    Main entry point for the surveillance pipeline.
    """
    args = parse_args()

    # Configure logging
    if args.verbose:
        logger.enable("src")
    else:
        logger.disable("src")

    # Display header
    console.print(
        Panel.fit(
            "[bold cyan]Agentic Counter-Surveillance Analysis[/bold cyan]\n"
            "Multi-agent pipeline for surveillance data analysis",
            border_style="cyan",
        )
    )

    # Create pipeline configuration
    scenario = AnalysisScenario(args.scenario)
    config_kwargs = {}

    if args.output_dir:
        config_kwargs["output_dir"] = args.output_dir
    if args.skip_scrape:
        config_kwargs["scrape_enabled"] = False
    if args.skip_analyze:
        config_kwargs["analyze_enabled"] = False

    # Create and run pipeline
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Initializing pipeline...", total=None)

            pipeline = create_pipeline(scenario=scenario, **config_kwargs)
            progress.update(
                task, description=f"[cyan]Running pipeline for {args.city}..."
            )

            # Prepare kwargs
            run_kwargs = {}
            if args.country:
                run_kwargs["country"] = args.country
            if args.output_dir:
                run_kwargs["output_dir"] = args.output_dir
            if args.data_path:
                run_kwargs["data_path"] = args.data_path

            results = pipeline.run(args.city, **run_kwargs)
            progress.update(task, description="[green]Pipeline completed")

        # Display results
        console.print()  # Blank line
        display_results(results)

        # Exit with appropriate status
        if results.get("success"):
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
