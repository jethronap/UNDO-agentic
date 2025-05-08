from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Union

from src.config.logger import logger


def load_overpass_elements(path: Path | str) -> List[Dict[str, Any]]:
    """
    Read an Overpass dump and return its elements list.
    Agents can call this tool by the name "load_json".

    :param path: The Path object to the Overpass dump
    :return: The extracted list of elements
    """
    p = Path(path).expanduser().resolve()
    logger.debug(f"Loading {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("elements", [])


def save_enriched_elements(elements: List[Dict[str, Any]], path: Path | str) -> str:
    """
    Write the enriched elements list next to the source file.
    :param elements: The enriched elements
    :param path: The path of the source file
    :return: The absolute path to the new file
    """
    p = Path(path).expanduser().resolve()
    destination = p.with_name(p.stem + "_enriched.json")
    logger.debug(f"Saving {destination}")
    destination.write_text(json.dumps({"elements": elements}, indent=2), "utf-8")
    return str(destination)


def save_overpass_dump(
    data: Dict[str, Any], city: str, overpass_dir: Union[Path, str]
) -> Path:
    """
    Save the Overpass API response to a JSON file in a specified directory.

    :param data: The JSON data to write.
    :param city: The name of the city used to name the file.
    :param overpass_dir: The output directory where the file will be saved.
    :returns: The full path to the saved file.
    """
    try:
        out = Path(overpass_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / f"{city.lower().replace(' ', '_')}.json"
        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return filepath
    except Exception as e:
        raise RuntimeError(f"Failed to save JSON for city '{city}'") from e
