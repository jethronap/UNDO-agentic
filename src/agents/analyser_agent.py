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
from src.utils.db import summarize
from src.memory.store import MemoryStore
from src.tools.io_tools import (
    load_overpass_elements as load_json,
    save_enriched_elements as save_json,
)


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
        }
        super().__init__(name, tools or default_tools, memory)
        self.llm = llm or LocalLLM()
        self._elements: List[Dict[str, Any]] = []
        self._enriched: List[Dict[str, Any]] = []

    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(input_data["path"]).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return {"path": path}

    def plan(self, observation: Dict[str, Any]) -> List[str]:
        # 1. load file 2. enrich each element 3. save file
        return ["load_json", "enrich", "save_json"]

    def act(self, action: str, context: Dict[str, Any]) -> Any:
        if action not in self.tools:
            raise ValueError(f"No tool named '{action}' found.")
        if action == "enrich":
            return [self.tools["enrich"](el) for el in context["elements"]]
        return (
            self.tools[action](**context)
            if action == "load_json"
            else self.tools[action](self._enriched, context["path"])
        )

    def achieve_goal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        observation = self.perceive(input_data)
        plan_steps = self.plan(observation)
        context: Dict[str, Any] = {**observation}

        for step in plan_steps:
            result = self.act(step, context)
            self.remember(step, summarize(result))
            if step == "load_json":
                context.update({"elements": result})
            elif step == "save_json":
                context["output_path"] = result
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
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8")
