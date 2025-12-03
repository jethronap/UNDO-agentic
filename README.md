# An Agentic System for Researching Surveillance Infrastructure

A multi-agent system for analyzing surveillance infrastructure and computing privacy-preserving walking routes in urban environments using OpenStreetMap data. The system operates completely locally without external APIs.

## Overview

The pipeline consists of three main agents:

- **Scraper Agent**: Downloads surveillance camera data from OpenStreetMap via Overpass API
- **Analyzer Agent**: Enriches data using local LLM analysis and generates visualizations
- **Route Finder Agent**: Computes low-surveillance walking routes using k-shortest paths and spatial analysis

**Key Features:**
- **Privacy-focused routing**: Find walking routes that minimize camera exposure
- **Local LLM processing**: No external API calls - complete privacy
- **Intelligent caching**: Agent memory stores results to avoid redundant computation
- **Multiple analysis scenarios**: Configurable presets (basic, full, quick, report, mapping)
- **Rich CLI interface**: Progress tracking and formatted result displays
- **Comprehensive visualizations**: Heatmaps, hotspots, route maps, and statistical charts
- **Spatial optimization**: Efficient GeoDataFrame indexing for large camera datasets

# Installation

## Prerequisites

- Python 3.11
- `uv` package manager

### Install Python 3.11

- For macOS
  - Use HomeBrew package manager. Install HomeBrew following these [instructions](https://brew.sh).
  
    ```commandline
    brew install python@3.11
    ```
- For Ubuntu
  - You can utilize the [Deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa).
    
    ```commandline
    sudo add-apt-repository ppa:deadsnakes/ppa
    ```
  - Update the package list.
    
    ```commandline
    sudo apt update
    ```
  - Install Python 3.11.
    
    ```commandline
    sudo apt install python3.11
    ```
  - Verify the installation.
    
    ```commandline
    python3.11 --version
    ```
    
### Install `uv`

```commandline
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create and activate virtual environment

```commandline
uv venv --python 3.11
source .venv/bin/activate
```

### Add dependencies:

```commandline
uv add name-of-dependency
```

### Synchronize dependencies

```commandline
uv sync
```

## Testing:
In order to run the tests from the root project run:

```commandline
bash ./local_test_pipeline.sh
```


## Code formatting

This project uses [.pre-commit](https://pre-commit.com) hooks to ensure universal code formatting.

To install these use:

```commandline
pre-commit install
```

## Ollama client

The application uses Ollama for interacting with LLMs locally.

In order for this to work follow these steps:

1. Create `.env` file at the root of the project. See [`.env-sample`](.env-sample) for the exact naming and properties.

2. Download and install [Ollama](https://ollama.com/download).

3. Open your terminal and execute the following command:

- Download the model:

```commandline
ollama pull llama3:latest
```
- Start Ollama:

```commandline
ollama serve
```

## Usage

The system provides a rich CLI interface for running surveillance analysis:

### Basic Usage

```bash
# Analyze a city with basic settings
python main.py Berlin

# Specify country for disambiguation
python main.py Athens --country GR

# Use different analysis scenarios
python main.py Hamburg --scenario full
python main.py Munich --scenario quick
```

### Analysis Scenarios

- `basic` (default): Essential analysis producing key files
- `full`: Complete analysis with all visualizations and reports
- `quick`: Fast analysis with minimal processing
- `report`: Focus on statistical summaries and charts
- `mapping`: Emphasis on geospatial visualizations

### Low-Surveillance Routing

The system can compute privacy-preserving walking routes that minimize exposure to surveillance cameras. Routes are calculated using k-shortest paths algorithms and scored based on camera density within a configurable buffer radius.

**Basic Routing:**

```bash
# Compute a low-surveillance route between two coordinates
python main.py Lund \
  --country SE \
  --enable-routing \
  --start-lat 55.709400 \
  --start-lon 13.194381 \
  --end-lat 55.705962 \
  --end-lon 13.182304
```

**Using Existing Data:**

```bash
# Skip scraping and use cached camera data
python main.py Malmö \
  --country SE \
  --data-path overpass_data/malmö/malmö.json \
  --skip-scrape \
  --enable-routing \
  --start-lat 55.595650 \
  --start-lon 13.022659 \
  --end-lat 55.594801 \
  --end-lon 13.000557
```

**Routing Features:**
- **k-shortest paths**: Evaluates multiple candidate routes (default: 3)
- **Exposure scoring**: Cameras per kilometer metric for route comparison
- **Baseline comparison**: Shows how much safer the route is vs. shortest path
- **Interactive maps**: Folium-based HTML visualizations with route and cameras
- **Graph caching**: OSMnx pedestrian networks cached locally for fast re-computation
- **Result caching**: Routes cached in agent memory for identical requests

**Note:** First-time routing for a city will download the pedestrian network from OSM, which can take several minutes for large cities. Subsequent routes in the same city will be much faster.

### Advanced Options

```bash
# Skip scraping (use existing data)
python main.py Berlin --data-path overpass_data/berlin/berlin.json --skip-scrape

# Skip analysis (scraping only)
python main.py Hamburg --skip-analyze

# Custom output directory
python main.py Paris --output-dir /custom/path

# Verbose logging (helpful for debugging routing performance)
python main.py London --verbose

# Combine routing with full analysis
python main.py Berlin \
  --scenario full \
  --enable-routing \
  --start-lat 52.52 \
  --start-lon 13.40 \
  --end-lat 52.50 \
  --end-lon 13.42
```

### Output Files

The system generates files in `overpass_data/<city>/` organized by function:

**Analysis Outputs:**
- **Enriched JSON** (`<city>_enriched.json`): Original data enhanced with LLM analysis
- **GeoJSON** (`<city>_enriched.geojson`): Geographic data for mapping applications
- **Heatmap** (`<city>_heatmap.html`): Interactive spatial density visualization
- **Hotspots** (`hotspots_<city>.geojson`, `hotspot_plot_<city>.png`): DBSCAN clustering results
- **Statistics** (`stats_chart_<city>.png`): Summary charts and metrics

**Routing Outputs** (in `routes/` subdirectory):
- **Route GeoJSON** (`route_<hash>.geojson`): Route geometry with exposure metrics and nearby camera IDs
- **Route Map** (`route_<hash>.html`): Interactive Folium map with:
  - Low-surveillance route (blue line)
  - Start/end markers (green/red)
  - Camera coverage circles (semi-transparent red)
  - Route metrics tooltip (length, exposure score)

**Cache Files:**
- **OSM Graphs** (`.graph_cache/<hash>.graphml`): Cached pedestrian networks
- **Agent Memory** (`memory.db`): SQLite database storing route and query caches

## Architecture

### Agent-Based Design

The system follows a perceive-plan-act pattern with three specialized agents:

1. **SurveillanceDataCollector** (Scraper)
   - Perceives: City name and query parameters
   - Plans: Check cache, determine if scraping needed
   - Acts: Query Overpass API, save results, update cache

2. **SurveillanceAnalyzerAgent** (Analyzer)
   - Perceives: Raw surveillance data from scraper
   - Plans: Select analysis workflow based on scenario
   - Acts: Enrich with LLM, generate visualizations, compute statistics

3. **RouteFinderAgent**
   - Perceives: Start/end coordinates, enriched camera data
   - Plans: Check route cache, determine computation steps
   - Acts: Build graph, generate k-shortest paths, score exposure, select optimal route

### Routing Algorithm

The routing system uses a multi-step approach to find privacy-preserving routes:

1. **Graph Construction**: OSMnx downloads walkable street network from OpenStreetMap
2. **Node Snapping**: Start/end coordinates snapped to nearest graph nodes (500m threshold)
3. **Path Generation**: NetworkX k-shortest simple paths algorithm generates candidate routes
4. **Exposure Scoring**:
   - Buffer each route by configurable radius (default: 50m)
   - Use GeoDataFrame spatial join to count cameras within buffer
   - Calculate exposure as cameras/km
5. **Route Selection**: Choose path with minimum exposure score
6. **Baseline Comparison**: Compare against shortest path to quantify privacy gain

**Performance Optimizations:**
- OSM graphs cached to disk (avoiding repeated downloads)
- Camera GeoDataFrame built once and reused across all candidate paths
- Routes cached in agent memory by (city, coordinates, settings) hash

### Configuration

Route computation can be customized via `src/config/settings.py`:

```python
class RouteSettings:
    max_candidates: int = 3           # Number of alternative paths to evaluate
    buffer_radius_m: float = 50.0     # Camera detection radius in meters
    network_type: str = "walk"        # OSMnx network type
    snap_distance_threshold_m: float = 500.0  # Max distance to snap coordinates
```

## Troubleshooting

### Routing Performance

**Symptom:** First routing attempt for a city takes 10-30+ minutes

**Cause:** OSMnx is downloading the entire pedestrian network from OpenStreetMap

**Solution:**
- Use `--verbose` flag to confirm it's the graph download step
- Be patient - this only happens once per city (results are cached)
- For large cities like Malmö, consider testing with closer coordinates first

**Performance Tips:**
- Test with points 500m-1km apart before trying longer routes
- Use `--data-path` and `--skip-scrape` to skip analysis when testing routes
- Check `overpass_data/.graph_cache/` to see which cities are already cached

### Coordinate Snapping Errors

**Error:** `Cannot snap (lat, lon) to walkable network: nearest node is XXXm away`

**Cause:** Coordinates are not near any walkable paths (e.g., middle of water, private property)

**Solution:**
- Verify coordinates using OpenStreetMap
- Ensure coordinates are on or near streets/sidewalks
- Try coordinates closer to known roads

### Help

```bash
python main.py --help
```

## Contributing

This project uses:
- **uv** for dependency management
- **pytest** for testing
- **pre-commit** hooks for code formatting
- **ruff** for linting

Run tests with:
```bash
bash ./local_test_pipeline.sh
```
