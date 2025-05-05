# UNDO - Agentic counter-surveillance

The application is agentic system that downloads data from various online sources, analyzes it and presents the results.
Everything is executed locally and no-external APIs are needed. The agents used in this software create and store 
memories of their actions.

# Scraper agent

The Scraper agent's goal is to get all available data that exist on OpenStreet maps concerning surveillance for a given
city or municipality. The agent downloads data in json format and stores them locally on the filesystem.

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
ollama pull gemma2
```
- Start Ollama:

```commandline
ollama serve
```