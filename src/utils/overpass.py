from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Union
import textwrap
import requests
from src.config.settings import OverpassSettings


def best_area_candidate(results: list[Dict[str, Any]]) -> tuple[int, str]:
    """
    Select the best boundary candidate from Nominatim search results.

    Priority:
      1. A relation with `admin_level` 8 or 9.
      2. Any relation.
      3. Fallback to the first way or node.

    :param results: A list of Nominatim result dictionaries.
    :returns: A tuple containing the OSM ID and type ('relation', 'way', or 'node').
    """
    for item in results:
        if item["osm_type"] == "relation" and item.get("extratags", {}).get(
            "admin_level"
        ) in {"8", "9"}:
            return int(item["osm_id"]), "relation"

    for item in results:
        if item["osm_type"] == "relation":
            return int(item["osm_id"]), "relation"

    first = results[0]
    return int(first["osm_id"]), first["osm_type"]


def nominatim_city(
    osm_query: str,
    settings: OverpassSettings = OverpassSettings(),
    country: str | None = None,
) -> tuple[int, str]:
    """
    Query Nominatim to find the best OSM boundary match for a city name.

    :param settings: The Overpass base settings.
    :param osm_query: The name of the city to search for.
    :param country: Optional 2-letter ISO country code to narrow the search.
    :returns: A tuple containing the OSM ID and type.
    :raises RuntimeError: If no results are returned from Nominatim.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": osm_query,
        "format": "jsonv2",
        "limit": 100,
        "polygon_geojson": 0,
        "extratags": 1,
    }
    if country:
        params["countrycodes"] = country.lower()

    try:
        r = requests.get(url, params=params, headers=settings.header, timeout=30)
        r.raise_for_status()
        results = r.json()
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Nominatim request failed for {osm_query!r}") from e

    if not results:
        raise RuntimeError(f"No Nominatim result for {osm_query!r}")

    return best_area_candidate(results)


def area_id(osm_id: int, osm_type: str) -> int:
    """
    Convert an OSM object ID into an Overpass area ID.

    :param osm_id: The OSM ID of the object.
    :param osm_type: The type of the OSM object ('node', 'way', or 'relation').
    :returns: The corresponding Overpass area ID.
    """
    base = {
        "node": 1_600_000_000,
        "way": 2_400_000_000,
        "relation": 3_600_000_000,
    }[osm_type]
    return base + osm_id


def build_query(
    city: str,
    *,
    country: str | None = None,
    settings: OverpassSettings = OverpassSettings(),
) -> str:
    """
    Build an Overpass QL query to find all `man_made=surveillance` features in a city.

    :param city: The name of the city to query.
    :param country: Optional ISO country code to disambiguate city name.
    :param settings: The Overpass base settings.
    :returns: A formatted Overpass QL query string.
    """
    try:
        osm_id, osm_type = nominatim_city(city, country=country, settings=settings)
        a_id = area_id(osm_id, osm_type)
    except Exception as e:
        raise RuntimeError(f"Failed to construct query for city '{city}': {e}") from e

    return textwrap.dedent(
        f"""
            [out:json][timeout:{settings.query_timeout}];
            area({a_id})->.searchArea;
            (
              nwr["man_made"="surveillance"](area.searchArea);
            );
            out geom;
            """
    ).strip()


def run_query(
    query: str, settings: OverpassSettings = OverpassSettings()
) -> Dict[str, Any]:
    """
    Submit an Overpass QL query and return the parsed JSON result.

    :param query: The Overpass query to execute.
    :param settings: The Overpass base settings.
    :returns: The JSON-decoded response from the Overpass API.
    :raises RuntimeError: If the Overpass API returns an error.
    """
    try:
        resp = requests.post(
            settings.endpoint,
            data=query.encode("utf-8"),
            timeout=60,
            headers=settings.header,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Overpass error {exc.response.status_code}: {exc.response.text.strip()}"
        ) from exc
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError("Failed to execute Overpass query") from e


def save_json(data: Dict[str, Any], city: str, overpass_dir: Union[Path, str]) -> Path:
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
