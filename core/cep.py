# -*- coding: utf-8 -*-
"""
Cognitive Enhancement Plugin (CEP) — Per-Agent Strategy Database.

Each heterogeneous agent (Retail, Institutional, Arbitrageur) maintains
its own strategy library  D_{s,k} keyed by (agent_role, state_type).
The plugin handles strategy CRUD (insert, update, retrieve, get_by_id)
and stores/loads strategies to disk per agent.
"""
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='market_tom_cep.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM.CEP')

COLOR_TITLE = "\033[1;36m"
COLOR_SUCCESS = "\033[1;32m"
COLOR_WARNING = "\033[1;33m"
COLOR_ERROR = "\033[1;31m"
COLOR_INFO = "\033[0;34m"
COLOR_VALUE = "\033[1;35m"
COLOR_DEBUG = "\033[0;90m"
COLOR_RESET = "\033[0m"

DEFAULT_AGENT_ROLES = ["Retail", "Institutional", "Arbitrageur"]


@dataclass
class StrategyData:
    """Strategy data structure with agent role affiliation."""
    level: str
    states_scenario: Dict[str, str]
    strategy: str
    timestamp: str
    agent_role: str = "_global"
    version: int = 1
    id: Optional[str] = None
    similarity: float = 0.0

    STATE_RELATIONSHIPS = {
        "belief": ["environmental"],
        "emotion": ["belief", "environmental"],
        "intent": ["belief"]
    }

    def to_dict(self) -> dict:
        item_dict = {
            "id": self.id,
            "states_scenario": self.states_scenario,
            "strategy": self.strategy,
            "timestamp": self.timestamp,
            "version": self.version,
            "agent_role": self.agent_role
        }
        return {"level": self.level, "item": item_dict}

    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyData':
        level = data["level"]
        item_data = data["item"]

        if "states_scenario" in item_data:
            states_scenario = item_data["states_scenario"]
        else:
            states_scenario = {}
            dependencies = cls.STATE_RELATIONSHIPS.get(level, [])
            for dep in dependencies:
                for key_variant in [f"{dep}_states_scenario", f"{dep}_scenario", dep]:
                    if key_variant in item_data and item_data[key_variant]:
                        states_scenario[dep] = item_data[key_variant]
                        break

        return cls(
            level=level,
            states_scenario=states_scenario,
            strategy=item_data["strategy"],
            timestamp=item_data["timestamp"],
            version=item_data.get("version", 1),
            id=item_data.get("id"),
            agent_role=item_data.get("agent_role", "_global")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyData':
        data = json.loads(json_str)
        if "item" in data and "timestamp" not in data["item"]:
            data["item"]["timestamp"] = datetime.now().isoformat()
        return cls.from_dict(data)


class CognitiveEnhancementPlugin:
    """Per-agent CEP strategy database.
    
    Storage layout:
        strategy_db[agent_role][level] = [strategy_dict, ...]
    
    On disk:
        storage_path/<agent_role>/<level>_strategies.json
    
    Falls back to _global when agent_role is None.
    """

    VALID_LEVELS = {"belief", "intent", "emotion"}

    def __init__(self, storage_path: str, agent_roles: List[str] = None):
        self.storage_path = storage_path
        self.agent_roles = agent_roles or DEFAULT_AGENT_ROLES
        # strategy_db: {agent_role: {level: [strategy_dict]}}
        self.strategy_db: Dict[str, Dict[str, List[dict]]] = {}
        self._embedder = None
        self._strategy_counter = 0

        os.makedirs(self.storage_path, exist_ok=True)

        # Initialize per-agent storage
        for role in self.agent_roles + ["_global"]:
            role_dir = os.path.join(self.storage_path, role.lower())
            os.makedirs(role_dir, exist_ok=True)
            self.strategy_db[role] = {level: [] for level in self.VALID_LEVELS}

        self._load_strategies()
        logger.info(f"CEP initialized: roles={self.agent_roles}, path={self.storage_path}")

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedder

    def _resolve_role(self, agent_role: Optional[str]) -> str:
        if agent_role and agent_role in self.agent_roles:
            return agent_role
        return "_global"

    def is_empty(self, agent_role: str = None) -> bool:
        role = self._resolve_role(agent_role)
        role_db = self.strategy_db.get(role, {})
        return all(len(strategies) == 0 for strategies in role_db.values())

    def get_strategies_by_level(self, level: str, agent_role: str = None) -> List[dict]:
        role = self._resolve_role(agent_role)
        return self.strategy_db.get(role, {}).get(level, [])

    def _get_storage_path(self, level: str, agent_role: str = None) -> str:
        role = self._resolve_role(agent_role)
        role_dir = os.path.join(self.storage_path, role.lower())
        os.makedirs(role_dir, exist_ok=True)
        return os.path.join(role_dir, f"{level}_strategies.json")

    def _load_strategies(self):
        """Load strategies from disk for all agent roles."""
        # Load per-agent strategies
        for role in self.agent_roles + ["_global"]:
            for level in self.VALID_LEVELS:
                filepath = self._get_storage_path(level, role)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                self.strategy_db[role][level] = data
                            elif isinstance(data, dict) and "strategies" in data:
                                self.strategy_db[role][level] = data["strategies"]
                            logger.info(
                                f"Loaded {len(self.strategy_db[role][level])} "
                                f"{level} strategies for {role}"
                            )
                    except Exception as e:
                        logger.error(f"Error loading {filepath}: {e}")
                        self.strategy_db[role][level] = []

        # Legacy: load from root strategy_database dir (old format)
        for level in self.VALID_LEVELS:
            legacy_path = os.path.join(self.storage_path, f"{level}_strategies.json")
            if os.path.exists(legacy_path):
                try:
                    with open(legacy_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        strategies = data if isinstance(data, list) else data.get("strategies", [])
                        if strategies and not self.strategy_db["_global"][level]:
                            self.strategy_db["_global"][level] = strategies
                            logger.info(
                                f"Loaded {len(strategies)} legacy {level} strategies → _global"
                            )
                except Exception as e:
                    logger.error(f"Error loading legacy {legacy_path}: {e}")

    def _save_strategies(self, level: str, agent_role: str = None):
        """Save strategies to disk for a specific agent/level."""
        role = self._resolve_role(agent_role)
        filepath = self._get_storage_path(level, role)
        strategies = self.strategy_db.get(role, {}).get(level, [])
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(strategies, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(strategies)} {level} strategies for {role}")
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")

    def _generate_strategy_id(self, level: str, agent_role: str = None) -> str:
        role = self._resolve_role(agent_role)
        role_prefix = role[:3].lower() if role != "_global" else "glb"
        self._strategy_counter += 1
        return f"{role_prefix}_{level}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._strategy_counter}"

    def get_strategy_by_id(self, level: str, strategy_id: str,
                           agent_role: str = None) -> Optional[dict]:
        """Retrieve a strategy by its ID."""
        if not strategy_id:
            return None

        role = self._resolve_role(agent_role)
        strategies = self.strategy_db.get(role, {}).get(level, [])
        for s in strategies:
            if s.get("id") == strategy_id:
                return {"level": level, "item": s}

        # Fallback: search _global
        if role != "_global":
            for s in self.strategy_db.get("_global", {}).get(level, []):
                if s.get("id") == strategy_id:
                    return {"level": level, "item": s}

        # Fallback: search all roles
        for r in self.agent_roles:
            for s in self.strategy_db.get(r, {}).get(level, []):
                if s.get("id") == strategy_id:
                    return {"level": level, "item": s}

        return None

    def insert_strategy(self, level: str, states_scenario: Dict[str, str],
                        strategy_content: str, agent_role: str = None) -> Optional[str]:
        """Insert a new strategy for a specific agent and level."""
        if level not in self.VALID_LEVELS:
            logger.error(f"Invalid level: {level}")
            return None

        role = self._resolve_role(agent_role)
        strategy_id = self._generate_strategy_id(level, role)

        new_strategy = {
            "id": strategy_id,
            "states_scenario": states_scenario,
            "strategy": strategy_content,
            "timestamp": datetime.now().isoformat(),
            "version": 1,
            "agent_role": role
        }

        if role not in self.strategy_db:
            self.strategy_db[role] = {l: [] for l in self.VALID_LEVELS}
        self.strategy_db[role][level].append(new_strategy)
        self._save_strategies(level, role)

        logger.info(f"Inserted strategy {strategy_id} for {role}/{level}")
        return strategy_id

    def update_strategy(self, level: str, strategy_id: str,
                        new_content: str, agent_role: str = None) -> Optional[str]:
        """Update an existing strategy's content."""
        role = self._resolve_role(agent_role)
        strategies = self.strategy_db.get(role, {}).get(level, [])

        for s in strategies:
            if s.get("id") == strategy_id:
                s["strategy"] = new_content
                s["timestamp"] = datetime.now().isoformat()
                s["version"] = s.get("version", 1) + 1
                self._save_strategies(level, role)
                logger.info(f"Updated strategy {strategy_id} for {role}/{level}")
                return strategy_id

        # Fallback: search _global
        if role != "_global":
            for s in self.strategy_db.get("_global", {}).get(level, []):
                if s.get("id") == strategy_id:
                    s["strategy"] = new_content
                    s["timestamp"] = datetime.now().isoformat()
                    s["version"] = s.get("version", 1) + 1
                    self._save_strategies(level, "_global")
                    return strategy_id

        logger.warning(f"Strategy {strategy_id} not found for update")
        return None

    def retrieve_strategies(self, level: str, query_scenario: Dict[str, str],
                            agent_role: str = None,
                            top_k: int = 1,
                            similarity_threshold: float = 0.1) -> List[dict]:
        """Retrieve top-K most relevant strategies by semantic similarity."""
        if level not in self.VALID_LEVELS:
            return []

        role = self._resolve_role(agent_role)
        candidates = list(self.strategy_db.get(role, {}).get(level, []))

        # Also include _global strategies as fallback
        if role != "_global":
            candidates.extend(self.strategy_db.get("_global", {}).get(level, []))

        if not candidates:
            return []

        # Build query text from scenario
        query_parts = []
        for key in sorted(query_scenario.keys()):
            query_parts.append(f"{key}: {query_scenario[key]}")
        query_text = " | ".join(query_parts)

        if not query_text.strip():
            return candidates[:top_k]

        try:
            query_emb = self._get_embedding(query_text)
            scored = []
            for s in candidates:
                scenario = s.get("states_scenario", {})
                scenario_parts = []
                for key in sorted(scenario.keys()):
                    scenario_parts.append(f"{key}: {scenario[key]}")
                candidate_text = " | ".join(scenario_parts)
                if not candidate_text.strip():
                    candidate_text = s.get("strategy", "")

                candidate_emb = self._get_embedding(candidate_text)
                sim = cosine_similarity(query_emb.reshape(1, -1),
                                        candidate_emb.reshape(1, -1))[0][0]
                if sim >= similarity_threshold:
                    scored.append((s, float(sim)))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = []
            for s, sim in scored[:top_k]:
                entry = dict(s)
                entry["similarity"] = sim
                results.append(entry)
            return results

        except Exception as e:
            logger.error(f"Strategy retrieval error: {e}\n{traceback.format_exc()}")
            return candidates[:top_k]

    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, show_progress_bar=False)
