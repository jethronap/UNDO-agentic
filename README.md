# Agentic Counter-Surveillance Analysis

A multi-agent system for analyzing surveillance infrastructure using OpenStreetMap data. The system operates completely locally without external APIs and uses LangChain-based agents that create memories of their actions.

## Overview

The pipeline consists of two main agents:

- **Scraper Agent**: Downloads surveillance data from OpenStreetMap via Overpass API
- **Analyzer Agent**: Enriches data using local LLM analysis and generates visualizations

**Key Features:**
- Local LLM processing (no external APIs)
- Persistent agent memory with SQLite
- Multiple analysis scenarios (basic, full, quick, report, mapping)
- Rich CLI interface with progress tracking
- Automatic caching to avoid re-downloading data
- Comprehensive visualizations (heatmaps, hotspots, charts)

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

### Advanced Options

```bash
# Skip scraping (use existing data)
python main.py Berlin --data-path overpass_data/berlin/berlin.json --skip-scrape

# Skip analysis (scraping only)
python main.py Hamburg --skip-analyze

# Custom output directory
python main.py Paris --output-dir /custom/path

# Verbose logging
python main.py London --verbose
```

### Output Files

The analysis generates several files in the output directory:

- **Enriched JSON**: Original data enhanced with LLM analysis
- **GeoJSON**: Geographic data for mapping applications
- **Heatmap**: Spatial density visualization
- **Hotspots**: DBSCAN clustering results and plots
- **Statistics**: Summary charts and metrics

### Help

```bash
python main.py --help
```
