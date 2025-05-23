from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Callable


from src.agents.base_agent import Agent
from src.config.logger import logger
from src.config.models.surveillance_metadata import SurveillanceMetadata
from src.config.settings import PromptsSettings
from src.tools.llm_wrapper import LocalLLM
from src.utils.db import summarize, payload_hash
from src.memory.store import MemoryStore
from src.tools.io_tools import (
    load_overpass_elements as load_json,
    save_enriched_elements as save_json,
    to_geojson,
)
from src.tools.mapping_tools import to_heatmap
from src.tools.stat_tools import compute_statistics
from src.utils.decorators import log_action

Tool = Callable[..., Any]


class AnalyzerAgent(Agent):
    """
    Loads an Overpass JSON dump, enriches each element via a local LLM,
    and writes an <city>_enriched.json file.
    """

    def __init__(
        self,
        name: str,
        memory: MemoryStore,
        llm: LocalLLM | None = None,
        tools: Dict[str, Tool] | None = None,
    ):
        default_tools: Dict[str, Tool] = {
            "load_json": load_json,
            "enrich": self._enrich_element,
            "save_json": save_json,
            "to_geojson": to_geojson,
            "to_heatmap": to_heatmap,
            "report": compute_statistics,
        }
        super().__init__(name, tools or default_tools, memory)
        self.llm = llm or LocalLLM()
        self._elements: List[Dict[str, Any]] = []
        self._enriched: List[Dict[str, Any]] = []

    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(input_data["path"]).expanduser().resolve()
        if not path.exists():
            logger.error(f"FileNotFound:{path}")
            raise FileNotFoundError(path)
        generate_geojson = input_data.get("generate_geojson", True)
        generate_heatmap = input_data.get("generate_heatmap", False)
        compute_stats = input_data.get("compute_stats", True)
        return {
            "path": path,
            "generate_geojson": generate_geojson,
            "generate_heatmap": generate_heatmap,
            "compute_stats": compute_stats,
        }

    def plan(self, observation: Dict[str, Any]) -> List[str]:
        steps = ["load_json", "enrich", "save_json"]
        if observation["generate_geojson"]:
            steps.append("to_geojson")
        if observation["generate_heatmap"]:
            steps.append("to_heatmap")
        if observation["compute_stats"]:
            steps.append("report")
        return steps

    @log_action
    def act(self, action: str, context: Dict[str, Any]) -> Any:
        if action not in self.tools:
            raise ValueError(f"No tool named '{action}' found.")

        if action == "load_json":
            raw_path = context["path"]
            enriched_path = raw_path.with_name(f"{raw_path.stem}_enriched.json")
            geojson_path = enriched_path.with_suffix(".geojson")

            elems = self.tools["load_json"](raw_path)
            context["elements"] = elems
            raw_hash = payload_hash({"elements": elems})
            context["raw_hash"] = raw_hash

            # Check filesystem
            context["enriched_exists"] = enriched_path.exists()
            context["geojson_exists"] = geojson_path.exists()

            if enriched_path.exists():
                context["output_path"] = str(enriched_path)
            if geojson_path.exists():
                context["geojson_path"] = str(geojson_path)

            # Look for enriched/geojson cache
            for m in self.memory.load(self.name):
                if m.step == "enriched_cache" and m.content.startswith(raw_hash):
                    _, enriched_s, geojson_s = m.content.split("|")
                    enriched_f = Path(enriched_s)
                    geojson_f = Path(geojson_s)
                    if enriched_f.exists() and geojson_f.exists():
                        context.update(
                            {
                                "output_path": str(enriched_f),
                                "geojson_path": str(geojson_f),
                                "cache_hit": True,
                            }
                        )
                        return elems
            context["cache_hit"] = False
            return elems

        if action == "enrich":
            if context["cache_hit"] or context.get("enriched_exists"):
                # skip LLM reload enriched json file
                enriched = json.loads(Path(context["output_path"]).read_text())
                context["enriched"] = enriched["elements"]
                return context["enriched"]
            enriched = [self.tools["enrich"](el) for el in context["elements"]]
            self._enriched = enriched
            return enriched

        if action == "save_json":
            if context["cache_hit"] or context.get("enriched_exists"):
                return context["output_path"]

            enriched_path: Path = self.tools["save_json"](
                context["enriched"], context["path"]
            )
            # stash path to enriched JSON for downstream steps
            context["output_path"] = str(enriched_path)
            return str(enriched_path)

        if action == "to_geojson":
            if context["cache_hit"] or context.get("geojson_exists"):
                return context["geojson_path"]
            enriched_path: Path = Path(context["output_path"])
            # derive .geojson filename alongside enriched JSON
            geojson_path = enriched_path.with_suffix(".geojson")
            self.tools["to_geojson"](enriched_path, geojson_path)
            context["geojson_path"] = str(geojson_path)
            # record the memory
            cache_value = (
                f"{context['raw_hash']}|{context['output_path']}|{geojson_path}"
            )
            self.remember("enriched_cache", cache_value)
            return str(geojson_path)

        if action == "to_heatmap":
            geojson_path = Path(context["geojson_path"])
            html_path = geojson_path.with_suffix(".html")
            self.tools["to_heatmap"](geojson_path, html_path)
            context["heatmap_path"] = str(html_path)
            self.remember("heatmap_cache", f"{context['raw_hash']}|{html_path}")
            return str(html_path)

        if action == "report":
            stats: Dict[str, Any] = self.tools["report"](context["enriched"])
            self.remember("report", json.dumps(stats))
            return stats

        logger.error(f"Unhandled action: {action}")
        raise NotImplementedError(f"Unhandled action: {action}")

    def achieve_goal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        observation = self.perceive(input_data)
        plan_steps = self.plan(observation)
        context: Dict[str, Any] = {**observation}

        for step in plan_steps:
            result = self.act(step, context)
            self.remember(step, summarize(result))
            if step == "load_json":
                context["elements"] = result
            elif step == "enrich":
                context["enriched"] = result
            elif step == "save_json":
                context["output_path"] = result
            elif step == "to_geojson":
                context["geojson_path"] = result
            elif step == "to_heatmap":
                context["heatmap_path"] = result
            elif step == "report":
                context["stats"] = result

        return context

    def _enrich_element(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load the prompt template once, fill it with the element's tag dict,
        call the LLM and merge the structured response back.
        """
        template = self._load_template()  # str (cached)
        filled_prompt = template.replace(
            "{{ tags }}",
            json.dumps(element.get("tags", {}), ensure_ascii=False, indent=2),
        )

        raw: str = self.llm.generate_response(filled_prompt, expect_json=True)

        # strip markdown fences if present
        raw = raw.strip()
        raw = raw.strip()
        if raw.startswith("```"):
            # drop first line (```json or ``` )
            raw = raw.split("\n", 1)[1]
            # drop trailing ``` fence
            raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

        try:
            enriched_fields: Dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned non‑JSON: {e}")
            enriched_fields = {}

        try:
            # validate schema
            meta = SurveillanceMetadata.from_raw(element, enriched_fields)
            analysis: Dict[str, Any] = meta.model_dump(exclude_none=True)
        except Exception as e:
            logger.warning(f"Schema validation failed for element {element['id']}: {e}")
            analysis = {"schema_errors": str(e), **enriched_fields}

        merged = {**element, "analysis": analysis}
        self._enriched.append(merged)
        return merged

    # helper – cached template loader
    @lru_cache(
        maxsize=1,
    )
    def _load_template(self, settings: PromptsSettings = PromptsSettings()) -> str:
        path = settings.template_dir / settings.template_file
        if not path.exists():
            logger.error(f"Prompt template not found: {path}")
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")
