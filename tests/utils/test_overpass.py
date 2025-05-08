import json
import pytest

from src.utils.overpass import (
    best_area_candidate,
    area_id,
    build_query,
    nominatim_city,
)
from src.tools.io_tools import save_overpass_dump
from tests.conftest import _DummyResp


def test_best_area_candidate_priorities():
    rel_city = {"osm_type": "relation", "osm_id": 7, "extratags": {"admin_level": "8"}}
    rel_generic = {"osm_type": "relation", "osm_id": 99}
    node = {"osm_type": "node", "osm_id": 5}

    assert best_area_candidate([rel_city, rel_generic, node]) == (7, "relation")
    assert best_area_candidate([rel_generic, node]) == (99, "relation")
    assert best_area_candidate([node]) == (5, "node")


@pytest.mark.parametrize(
    ("osm_type", "expected"),
    [("node", 1_600_000_123), ("way", 2_400_000_123), ("relation", 3_600_000_123)],
)
def test_area_id(osm_type, expected):
    assert area_id(123, osm_type) == expected


def test_nominatim_city_success(patch_requests, ovp_settings):
    patch_requests["get"] = _DummyResp(
        status_code=200,
        _text=json.dumps(
            [{"osm_type": "relation", "osm_id": 42, "extratags": {"admin_level": "8"}}]
        ),
    )
    assert nominatim_city("Foo", settings=ovp_settings) == (42, "relation")


def test_nominatim_city_no_results(patch_requests, ovp_settings):
    patch_requests["get"] = _DummyResp(status_code=200, _text="[]")
    with pytest.raises(RuntimeError, match="No Nominatim result"):
        nominatim_city("Nowhere", settings=ovp_settings)


def test_build_query_contains_area(monkeypatch, ovp_settings):
    monkeypatch.setattr(
        "src.utils.overpass.nominatim_city", lambda *a, **k: (55, "relation")
    )
    q = build_query("Bar", settings=ovp_settings)
    assert "area(3600000055)" in q
    assert '"man_made"="surveillance"' in q


def test_run_query_success(patch_requests, ovp_settings):
    patch_requests["post"] = _DummyResp(
        status_code=200, _text='{"elements": [{"id": 1}]}'
    )

    import src.utils.overpass as ovp

    data = ovp.run_query("dummy", settings=ovp_settings)

    assert data == {"elements": [{"id": 1}]}


def test_save_json_roundtrip(tmp_path):
    fp = save_overpass_dump({"foo": 1}, "Foo City", tmp_path)
    assert fp.exists()
    assert fp.read_text() == json.dumps({"foo": 1}, indent=2)
