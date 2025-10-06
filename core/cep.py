import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
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

COLOR_TITLE = "\033[1;36m"     # Cyan bold (titles)
COLOR_SUCCESS = "\033[1;32m"   # Green bold (success)
COLOR_WARNING = "\033[1;33m"   # Yellow bold (warnings)
COLOR_ERROR = "\033[1;31m"     # Red bold (errors)
COLOR_INFO = "\033[0;34m"      # Blue (info)
COLOR_VALUE = "\033[1;35m"     # Magenta bold (values)
COLOR_DEBUG = "\033[0;90m"     # Gray (debug)
COLOR_RESET = "\033[0m"        # Reset color

@dataclass
class StrategyData:
    """Strategy data structure that maintains causal relationships"""
    level: str
    states_scenario: Dict[str, str]
    strategy: str
    timestamp: str
    version: int = 1
    id: Optional[str] = None
    similarity: float = 0.0

    STATE_RELATIONSHIPS = {
        "belief": ["environmental_states"],
        "emotion": ["belief_states", "environmental_states"],
        "intent": ["belief_states"]
    }
    logger.info(f"StrategyData.STATE_RELATIONSHIPS loaded: {STATE_RELATIONSHIPS}")

    def to_dict(self) -> dict:
        """Convert to dictionary maintaining states_scenario structure"""
        item_dict = {
            "id": self.id,
            "states_scenario": self.states_scenario,
            "strategy": self.strategy,
            "timestamp": self.timestamp,
            "version": self.version
        }
        
        return {
            "level": self.level,
            "item": item_dict
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyData':
        """Create from dictionary handling both formats"""
        level = data["level"]
        item_data = data["item"]
        
        if "states_scenario" in item_data:
            states_scenario = item_data["states_scenario"]
        else:
            states_scenario = {}
            dependencies = cls.STATE_RELATIONSHIPS.get(level, [])
            for dep in dependencies:
                state_type = dep.replace('_states', '')
                scenario_key = f"{dep}_scenario"
                if scenario_key in item_data and item_data[scenario_key]:
                    states_scenario[state_type] = item_data[scenario_key]
        
        return cls(
            level=level,
            states_scenario=states_scenario,
            strategy=item_data["strategy"],
            timestamp=item_data["timestamp"],
            version=item_data.get("version", 1),
            id=item_data.get("id")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'StrategyData':
        data = json.loads(json_str)
        if "item" in data and "timestamp" not in data["item"]:
            data["item"]["timestamp"] = datetime.now().isoformat()
        return cls.from_dict(data)


class CognitiveEnhancementPlugin:
    """Enhanced Cognitive Enhancement Plugin with explicit causal relationships"""
    
    VALID_LEVELS = {"belief", "intent", "emotion"}
    
    def __init__(self, storage_path: str):
        """
        Initializes the CognitiveEnhancementPlugin.
        The `storage_path` should be a fully resolved, unambiguous path provided by `run.py` or another caller.
        CEP will not try to guess paths internally.
        """
        logger.info(f"Initializing CEP with storage_path: '{storage_path}'")
        print(f"{COLOR_INFO}Initializing strategy database...{COLOR_RESET}")
        
        abs_storage_path = os.path.abspath(storage_path)
        self.storage_path = os.path.normpath(abs_storage_path)

        logger.info(f"Using normalized absolute storage_path: '{self.storage_path}'")
        
        self._ensure_storage_exists()
        
        self.strategy_db: Dict[str, List[dict]] = {
            level: [] for level in self.VALID_LEVELS
        }
        
        self._load_strategies()
        
        try:
            model_name = 'all-MiniLM-L6-v2'
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
            print(f"{COLOR_SUCCESS}✓ Strategy retrieval system ready{COLOR_RESET}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            print(f"{COLOR_ERROR}⚠ Strategy embedding model not available{COLOR_RESET}")
            self.embedding_model = None
        
    def is_empty(self) -> bool:
        for level in ["belief", "emotion", "intent"]:
            if self.get_strategies_by_level(level):
                return False
        return True
    
    def get_strategies_by_level(self, level: str) -> List[dict]:
        if level not in self.VALID_LEVELS:
            logger.warning(f"Invalid strategy level requested: {level}")
            return []
            
        return self.strategy_db.get(level, [])
        
    def _ensure_storage_exists(self):
        """Ensure storage directory exists"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            logger.info(f"Created strategy storage directory: {self.storage_path}")
    
    def _get_storage_path(self, level: str) -> str:
        """Get storage file path for specific level"""
        return os.path.join(self.storage_path, f"{level}_strategies.json")
    
    def _load_strategies(self):
        """Load all strategies from storage"""
        logger.info(f"Loading strategies from storage: {self.storage_path}")
        print(f"{COLOR_TITLE}STEP 1: LOADING STRATEGY DATABASE{COLOR_RESET}")
        
        if not os.path.isdir(self.storage_path):
            logger.error(f"Strategy database path is not a directory: '{self.storage_path}'")
            print(f"{COLOR_ERROR}✗ Strategy database path not found{COLOR_RESET}")
            return

        total_strategies = 0
        for level in self.VALID_LEVELS:
            file_path_to_load = self._get_storage_path(level)
            
            abs_file_path = os.path.abspath(file_path_to_load)
            logger.info(f"Attempting to load {level} strategies from: {abs_file_path}")
            
            if os.path.isfile(abs_file_path):
                try:
                    with open(abs_file_path, 'r', encoding='utf-8') as f:
                        content_peek = f.read(10)
                        if not content_peek.strip():
                            logger.warning(f"File for {level} is empty or contains only whitespace")
                            self.strategy_db[level] = []
                            continue
                        f.seek(0)
                        
                        data = json.load(f)
                        if not isinstance(data, list):
                            logger.error(f"JSON content in '{abs_file_path}' is not a list")
                            self.strategy_db[level] = []
                            continue

                        loaded_strategies = []
                        for item_idx, item_content in enumerate(data):
                            try:
                                loaded_strategies.append(StrategyData.from_dict(item_content).to_dict())
                            except Exception as item_e:
                                logger.error(f"Failed to parse strategy #{item_idx} in {level} file: {str(item_e)}")
                        
                        self.strategy_db[level] = loaded_strategies
                        logger.info(f"Loaded {len(loaded_strategies)} {level} strategies")
                        total_strategies += len(loaded_strategies)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON decode error in {abs_file_path}: {json_err}")
                    self.strategy_db[level] = []
                except Exception as e:
                    logger.error(f"Failed to load {level} strategies: {e}")
                    self.strategy_db[level] = []
            else:
                logger.info(f"No strategy file found for {level}. Initializing empty list.")
                self.strategy_db[level] = []
        
        if total_strategies > 0:
            print(f"{COLOR_SUCCESS}✓ Loaded {total_strategies} strategies{COLOR_RESET}")
        else:
            print(f"{COLOR_WARNING}! No strategies found in database{COLOR_RESET}")
        
        logger.info(f"Strategy loading complete, {total_strategies} total strategies loaded")
    
    def _save_strategies(self, level: str):
        """Save strategies for specific level to file"""
        file_path = self._get_storage_path(level)
        logger.info(f"Saving {level} strategies to file: {file_path}")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.strategy_db[level], f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully saved {len(self.strategy_db[level])} {level} strategies")
            print(f"{COLOR_SUCCESS}✓ Saved {level} strategies to database{COLOR_RESET}")
        except Exception as e:
            logger.error(f"Error saving {level} strategies: {str(e)}")
            print(f"{COLOR_ERROR}✗ Failed to save {level} strategies{COLOR_RESET}")
            
    def get_strategy_by_id(
        self, 
        level: str, 
        strategy_id: str
    ) -> Optional[dict]:
        """
        Retrieve a specific strategy by level and id
        
        Args:
            level (str): The strategy level
            strategy_id (str): The strategy id

        Returns:
            Optional[dict]: The strategy if found, None otherwise
        """
        if level not in self.VALID_LEVELS:
            logger.warning(f"Invalid level in get_strategy_by_id: {level}")
            return None
            
        try:
            # Search for strategy with matching ID
            for strategy in self.strategy_db[level]:
                if strategy["item"].get("id") == strategy_id:
                    return strategy
                    
            logger.info(f"No strategy found with id {strategy_id} in level {level}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving strategy: {str(e)}")
            return None
            
    def insert_strategy(
            self,
            level: str,
            states_scenario: Dict[str, str],
            strategy: str
        ) -> Optional[str]:
            """
            Insert a new strategy with automatically generated ID, timestamp and version.
            
            Args:
                level: The cognitive level (belief, emotion, intent, action)
                states_scenario: Dict mapping state type to scenario description
                            e.g. {"belief": "Market trend up", "environmental": "High volume"}
                strategy: The strategy description
                
            Returns:
                Generated strategy ID if successful, None if failed
            """
            try:
                if level not in self.VALID_LEVELS:
                    logger.error(f"Invalid strategy level: {level}, valid levels: {self.VALID_LEVELS}")
                    print(f"{COLOR_ERROR}Invalid strategy level: {level}{COLOR_RESET}")
                    raise ValueError(f"Invalid level '{level}'")
                    
                required_states = [dep.replace('_states','') for dep in StrategyData.STATE_RELATIONSHIPS.get(level, [])]
                logger.info(f"Required states for {level}: {required_states}")
                logger.info(f"Provided states: {list(states_scenario.keys())}")
                
                missing_states = []
                for state in required_states:
                    if state not in states_scenario:
                        missing_states.append(state)
                
                if missing_states:
                    logger.warning(f"Missing required states for {level}: {missing_states}")
                    
                    auto_fixed = True
                    for state in missing_states:
                        placeholder = f"[AUTO-GENERATED {state.upper()} STATE]"
                        states_scenario[state] = placeholder
                        logger.info(f"Auto-added state placeholder: {state} = {placeholder}")
                    
                    if auto_fixed:
                        logger.info("Successfully auto-filled all missing states")
                    else:
                        logger.error("Could not auto-fill all missing states")
                        return None
                
                new_id = f"{level}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.strategy_db[level])+1:04d}"
                
                strategy_data = StrategyData(
                    level=level,
                    states_scenario=states_scenario,
                    strategy=strategy,
                    timestamp=datetime.now().isoformat(),
                    version=1,
                    id=new_id
                )
                
                strategy_dict = strategy_data.to_dict()
                self.strategy_db[level].append(strategy_dict)
                self._save_strategies(level)
                
                print(f"{COLOR_SUCCESS}✓ Created new {level} strategy: {new_id}{COLOR_RESET}")
                return new_id
                
            except Exception as e:
                logger.error(f"Error creating strategy: {str(e)}")
                logger.error(traceback.format_exc())
                print(f"{COLOR_ERROR}✗ Failed to create strategy{COLOR_RESET}")
                return None

    def update_strategy(
            self,
            level: str,
            strategy_id: str,
            strategy: str
        ) -> Optional[str]:
            """
            Update an existing strategy with new strategy text and incremented version.
            
            Args:
                level: The cognitive level of the strategy
                strategy_id: The ID of the strategy to update
                strategy: The new strategy text
                
            Returns:
                Updated strategy ID if successful, None if failed
            """
            try:
                if level not in self.VALID_LEVELS:
                    logger.error(f"Invalid strategy level in update: {level}")
                    raise ValueError(f"Invalid level '{level}'")
                
                for i, existing_strategy in enumerate(self.strategy_db[level]):
                    if existing_strategy["item"].get("id") == strategy_id:
                        old_data = StrategyData.from_dict(existing_strategy)
                        
                        new_version = old_data.version + 1
                        new_id = f"{level}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{new_version:04d}"
                        
                        updated_data = StrategyData(
                            level=level,
                            states_scenario=old_data.states_scenario,
                            strategy=strategy,
                            timestamp=datetime.now().isoformat(),
                            version=new_version,
                            id=new_id
                        )
                        
                        strategy_dict = updated_data.to_dict()
                        self.strategy_db[level][i] = strategy_dict
                        self._save_strategies(level)
                        
                        logger.info(f"Updated strategy {strategy_id} to new version {new_id}")
                        print(f"{COLOR_SUCCESS}✓ Updated {level} strategy to version {new_version}{COLOR_RESET}")
                        return new_id
                        
                logger.warning(f"Strategy with ID {strategy_id} not found in level {level}")
                print(f"{COLOR_WARNING}! Strategy not found: {strategy_id}{COLOR_RESET}")
                raise ValueError(f"Strategy with ID {strategy_id} not found in level {level}")
                
            except Exception as e:
                logger.error(f"Error updating strategy: {str(e)}")
                print(f"{COLOR_ERROR}✗ Failed to update strategy{COLOR_RESET}")
                return None
        
    def retrieve_strategies(
            self,
            level: str,
            scenarios: Dict[str, str],
            top_k: int = 5,
            similarity_threshold: float = 0.1
        ) -> List[dict]:
        """
        Retrieve strategies most relevant to the given scenario
        """
        if level not in self.VALID_LEVELS:
            logger.error(f"Invalid strategy level in retrieval: {level}")
            return []
        
        strategies = self.strategy_db[level]
        
        if not strategies:
            logger.warning(f"No strategies found for level: {level}")
            return []
        
        if not scenarios:
            logger.warning("No scenarios provided for retrieval")
            return []
        
        try:
            query_texts = []
            for scenario_type, content in scenarios.items():
                if content and content.strip():
                    query_texts.append(f"{scenario_type}: {content}")
            
            if not query_texts:
                logger.warning("Scenario content is empty")
                return []
                
            query_text = " ".join(query_texts)
            query_embedding = self._get_embedding(query_text)
            
            results_with_scores = []
            
            for strategy in strategies:
                strategy_states_scenario = strategy["item"].get("states_scenario", {})
                if not strategy_states_scenario:
                    continue
                
                strategy_texts = []
                for state_type, content in strategy_states_scenario.items():
                    if content and content.strip():
                        strategy_texts.append(f"{state_type}: {content}")
                
                if not strategy_texts:
                    continue
                    
                strategy_text = " ".join(strategy_texts)
                strategy_embedding = self._get_embedding(strategy_text)
                
                similarity = cosine_similarity([query_embedding], [strategy_embedding])[0][0]
                
                if similarity >= similarity_threshold:
                    strategy_copy = dict(strategy)
                    if "item" in strategy_copy:
                        strategy_copy["similarity"] = float(similarity)
                    results_with_scores.append((strategy_copy, similarity))
            
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = [result for result, _ in results_with_scores[:top_k]]
            
            logger.info(f"Retrieved {len(top_results)} strategies for {level} (out of {len(strategies)} total)")
            return top_results
            
        except Exception as e:
            logger.error(f"Strategy retrieval error: {str(e)}")
            return []

    def _get_embedding(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            logger.error(f"Text embedding conversion failed: {str(e)}")
            return np.zeros(384)
