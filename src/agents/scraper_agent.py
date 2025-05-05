from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Callable
from src.agents.base_agent import Agent
from src.config.logger import logger
from src.config.settings import OverpassSettings
from src.utils.db import summarize, query_hash, payload_hash
from src.utils.overpass import build_query, run_query, save_json
from src.memory.store import MemoryStore

Tool = Callable[..., Any]


class ScraperAgent(Agent):
    """
    Agent that fetches `man_made=surveillance` objects from OpenStreetMap via the Overpass API and remembers every step.
    """

    def __init__(
        self, name: str, memory: MemoryStore, tools: Dict[str, Tool] | None = None
    ) -> None:
        """
        Constructor.
        :param name: The Agent name.
        :param memory: The memory store.
        :param tools: The tools to use.
        """
        default_tools: Dict[str, Tool] = {
            "run_query": run_query,
            "save_json": save_json,
        }
        super().__init__(name=name, tools=tools or default_tools, memory=memory)

    def perceive(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expect query params and build query text.
        :param input_data: The query params. Eg. `{"city": "Malmö", "overpass_dir": "data"}`
        :return: An enriched observation for subsequent stages
        """

        city = input_data["city"]
        country = input_data.get("country")  # new, optional
        query = build_query(city, country=country)
        return {
            "city": city,
            "country": country,
            "query": query,
            "overpass_dir": input_data.get("overpass_dir", "overpass_data"),
        }

    def plan(self, observation: Dict[str, Any]) -> List[str]:
        """
        Very simple two-step plan: (1) fetch → (2) persist.
        :param observation:
        :return: The available steps.
        """
        return ["run_query", "save_json"]

    def act(self, action: str, context: Dict[str, Any]) -> Any:
        """
        Map action name to the corresponding tool and return its result.
        :param action: The name of the action.
        :param context: The data from `perceive()`.
        :return: The actions result.
        """

        if action not in self.tools:
            raise ValueError(f"No tool named '{action}' found.")

        if action == "run_query":
            q_hash = query_hash(context["query"])
            # Look for a cache entry
            if self.memory:
                for m in self.memory.load(self.name):
                    if m.step == "cache" and m.content.startswith(q_hash):
                        _, fp, p_hash = m.content.split("|")
                        filepath = Path(fp)
                        if filepath.exists():
                            # cache hit
                            with filepath.open(encoding="utf-8") as f:
                                data = json.load(f)
                            # double-check integrity
                            if payload_hash(data) == p_hash:
                                elements = len(data.get("elements", []))
                                # make sure steps down the line have what they need
                                context.update(
                                    {
                                        "cache_hit": True,
                                        "data": data,
                                        "cached_path": str(filepath),
                                        "elements_count": elements,
                                        "empty": elements == 0,
                                    }
                                )
                                return data
            # otherwise run the query
            data = self.tools[action](context["query"])
            elements = len(data.get("elements", []))
            context.update(
                {
                    "cache_hit": False,
                    "data": data,
                    "elements_count": elements,
                    "empty": elements == 0,
                }
            )
            return data

        if action == "save_json":
            # skip if the query returns empty
            if context.get("empty", False):
                # remember so that we don't re-fetch
                self.remember(
                    "empty",
                    f"{context['city']}|{context.get('country')}|{query_hash(context['query'])}",
                )
                return "NO_DATA"
            # skip if served from cache
            if context.get("cache_hit"):
                return context["cached_path"]
            overpass_dir = OverpassSettings().dir
            path = self.tools[action](context["data"], context["city"], overpass_dir)
            q_hash = query_hash(context["query"])
            p_hash = payload_hash(context["data"])
            self.remember("cache", f"{q_hash}|{path}|{p_hash}")
            return str(path)

        raise NotImplementedError(action)

    def achieve_goal(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrates the entire agent life-cycle while keeping every intermediate result
        alive, inspectable, and persisted. Overridden to keep the whole `context` alive between steps.
        :param input_data: The user raw input.
        :return: The final context dictionary produced by the run.
        """
        observation = self.perceive(input_data)
        plan_steps = self.plan(observation)
        # A shallow copy of observation i.e. the original object is not mutated.
        # From here on context is shared and every stage can read or extend.
        context: Dict[str, Any] = {**observation}

        for step in plan_steps:
            result = self.act(step, context)
            self.remember(step, summarize(result))
            context[step] = result

        if context.get("empty"):
            logger.warning(
                f"[ScraperAgent] WARNING: 0 surveillance objects found for "
                f"{context['city']} ({context.get('country', 'no country')})"
            )
        return context
