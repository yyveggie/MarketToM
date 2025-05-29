
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

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
        "intent": ["belief_states"],
        "action": ["emotion_states", "intent_states"]
    }

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
        
        # Handle states_scenario field
        if "states_scenario" in item_data:
            # New format with states_scenario dictionary
            states_scenario = item_data["states_scenario"]
        else:
            # Old format with *_states_scenario fields
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
        # Add timestamp if not present
        if "item" in data and "timestamp" not in data["item"]:
            data["item"]["timestamp"] = datetime.now().isoformat()
        return cls.from_dict(data)


class CognitiveEnhancementPlugin:
    """Enhanced Cognitive Enhancement Plugin with explicit causal relationships"""
    
    VALID_LEVELS = {"environmental", "belief", "intent", "emotion", "action"}
    
    def __init__(self, storage_path: str):
        """
        Initializes the CognitiveEnhancementPlugin.
        The `storage_path` should be a fully resolved, unambiguous path provided by `run.py` or another caller.
        CEP will not try to guess paths internally.
        """
        print(f"[CEP Init] Received storage_path: '{storage_path}'")

        abs_storage_path = os.path.abspath(storage_path)
        self.storage_path = os.path.normpath(abs_storage_path)

        print(f"[CEP Init] Using normalized absolute storage_path: '{self.storage_path}'")
        
        self._ensure_storage_exists()
        
        self.strategy_db: Dict[str, List[dict]] = {
            level: [] for level in self.VALID_LEVELS
        }
        
        self._load_strategies()
        
        try:
            model_name = 'all-MiniLM-L6-v2'
            print(f"[CEP Init] Attempting to load embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            print("[CEP Init] Embedding model loaded successfully.")
        except Exception as e:
            print(f"[CEP Init] WARNING: Failed to load embedding model: {e}")
            print(f"[CEP Init] Detailed error: {traceback.format_exc()}")
            self.embedding_model = None
        
    def is_empty(self) -> bool:
        for level in ["belief", "emotion", "intent"]:
            if self.get_strategies_by_level(level):
                return False
        return True
    
    def get_strategies_by_level(self, level: str) -> List[dict]:
        if level not in self.VALID_LEVELS:
            print(f"Invalid level: {level}")
            return []
            
        return self.strategy_db.get(level, [])
        
    def _ensure_storage_exists(self):
        """Ensure storage directory exists"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
    
    def _get_storage_path(self, level: str) -> str:
        """Get storage file path for specific level"""
        return os.path.join(self.storage_path, f"{level}_strategies.json")
    
    def _load_strategies(self):
        """Load all strategies from storage"""
        print("\n--- [CEP DEBUG] Starting to load strategies from storage ---")
        print(f"  [CEP DEBUG] Strategy database directory being used: '{self.storage_path}'")
        
        if not os.path.isdir(self.storage_path):
            print(f"  [CEP ERROR] The resolved strategy database path IS NOT A DIRECTORY: '{self.storage_path}'. Cannot load strategies.")
            return

        for level in self.VALID_LEVELS:
            file_path_to_load = self._get_storage_path(level)
            
            norm_file_path = os.path.normpath(file_path_to_load)
            abs_file_path = os.path.abspath(norm_file_path)

            print(f"\n  Attempting to load level '{level}'")
            print(f"    Constructed file path: '{file_path_to_load}'")
            print(f"    Normalized path: '{norm_file_path}'")
            print(f"    Absolute path for checking: '{abs_file_path}'")
            
            exists = os.path.exists(abs_file_path)
            isfile = os.path.isfile(abs_file_path)
            print(f"    os.path.exists() check: {exists}")
            print(f"    os.path.isfile() check: {isfile}")

            if isfile:
                try:
                    with open(abs_file_path, 'r', encoding='utf-8') as f:
                        content_peek = f.read(10)
                        if not content_peek.strip():
                            print(f"    WARNING: File '{abs_file_path}' for level '{level}' is empty or contains only whitespace. Skipping.")
                            self.strategy_db[level] = []
                            continue
                        f.seek(0)
                        
                        data = json.load(f)
                        if not isinstance(data, list):
                             print(f"    ERROR: JSON content in '{abs_file_path}' for level '{level}' is not a list. Skipping.")
                             self.strategy_db[level] = []
                             continue

                        loaded_strategies = []
                        for item_idx, item_content in enumerate(data):
                            try:
                                loaded_strategies.append(StrategyData.from_dict(item_content).to_dict())
                            except Exception as item_e:
                                print(f"    ERROR: Failed to parse item #{item_idx} in '{abs_file_path}' for level '{level}'. Item: {item_content}. Error: {item_e}")
                        
                        self.strategy_db[level] = loaded_strategies
                        print(f"    Successfully loaded {len(loaded_strategies)} strategies for level '{level}'.")
                except json.JSONDecodeError as json_err:
                    print(f"    !!!!!!!! ERROR decoding JSON from {abs_file_path} for level '{level}' !!!!!!!!")
                    print(f"    JSONDecodeError: {json_err}")
                    self.strategy_db[level] = []
                except Exception as e:
                    print(f"    !!!!!!!! ERROR loading {level} level strategies from {abs_file_path} !!!!!!!!")
                    print(f"    Exception: {e}")
                    print(f"    Traceback: {traceback.format_exc()}")
                    self.strategy_db[level] = []
            else:
                print(f"    File not found or is not a file for level '{level}' at '{abs_file_path}'. Initializing empty list for this level.")
                self.strategy_db[level] = []
        print("--- [CEP DEBUG] Finished loading strategies ---\n")
    
    def _save_strategies(self, level: str):
        """Save strategies for specific level to file"""
        file_path = self._get_storage_path(level)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.strategy_db[level], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving {level} level strategies: {str(e)}")
            
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
            print(f"Invalid level: {level}")
            return None
            
        try:
            # Search for strategy with matching ID
            for strategy in self.strategy_db[level]:
                if strategy["item"].get("id") == strategy_id:
                    return strategy
                    
            print(f"No strategy found with id {strategy_id} in level {level}")
            return None
            
        except Exception as e:
            print(f"Error retrieving strategy: {str(e)}")
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
                # Validate level
                if level not in self.VALID_LEVELS:
                    raise ValueError(f"Invalid level '{level}'")
                    
                # Validate required states based on STATE_RELATIONSHIPS
                required_states = [dep.replace('_states','') for dep in StrategyData.STATE_RELATIONSHIPS.get(level, [])]
                if not all(state in states_scenario for state in required_states):
                    missing = set(required_states) - set(states_scenario.keys())
                    raise ValueError(f"Missing required scenarios for states: {missing}")
                
                # Generate new ID
                new_id = f"{level}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.strategy_db[level])+1:04d}"
                
                # Create new strategy data
                strategy_data = StrategyData(
                    level=level,
                    states_scenario=states_scenario,
                    strategy=strategy,
                    timestamp=datetime.now().isoformat(),
                    version=1,
                    id=new_id
                )
                
                # Convert to dict and save
                strategy_dict = strategy_data.to_dict()
                self.strategy_db[level].append(strategy_dict)
                self._save_strategies(level)
                
                return new_id
                
            except Exception as e:
                print(f"Error inserting strategy: {str(e)}")
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
                # Validate level
                if level not in self.VALID_LEVELS:
                    raise ValueError(f"Invalid level '{level}'")
                
                # Find the strategy to update
                for i, existing_strategy in enumerate(self.strategy_db[level]):
                    if existing_strategy["item"].get("id") == strategy_id:
                        # Create updated strategy data
                        old_data = StrategyData.from_dict(existing_strategy)
                        
                        # Generate new ID with incremented version
                        new_version = old_data.version + 1
                        new_id = f"{level}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{new_version:04d}"
                        
                        # Create updated strategy
                        updated_data = StrategyData(
                            level=level,
                            states_scenario=old_data.states_scenario,
                            strategy=strategy,
                            timestamp=datetime.now().isoformat(),
                            version=new_version,
                            id=new_id
                        )
                        
                        # Update in database
                        strategy_dict = updated_data.to_dict()
                        self.strategy_db[level][i] = strategy_dict
                        self._save_strategies(level)
                        
                        return new_id
                        
                raise ValueError(f"Strategy with ID {strategy_id} not found in level {level}")
                
            except Exception as e:
                print(f"Error updating strategy: {str(e)}")
                return None
        
    def retrieve_strategies(
            self,
            level: str,
            scenarios: Dict[str, str],
            top_k: int = 5,
            similarity_threshold: float = 0.1
        ) -> List[dict]:
            """
            Retrieve strategies based on semantic similarity using sentence embeddings.

            Args:
                level (str): The strategy level (e.g., "belief").
                scenarios (Dict[str, str]): Dictionary of current state scenarios
                                            (e.g., {"environmental": "...", "belief": "..."})
                                            used for querying.
                top_k (int): The maximum number of strategies to return.
                similarity_threshold (float): The minimum similarity score to consider a match.

            Returns:
                List[dict]: A list of relevant strategies sorted by similarity.
            """
            print(f"\\n--- [CEP DEBUG] Retrieving strategies for level: '{level}' ---")
            print(f"  Input Scenarios Keys: {list(scenarios.keys())}")
            # print(f"  Input Scenarios Content (first 100 chars): {[v[:100] + '...' for v in scenarios.values()]}") # Avoid printing huge env_state
            print(f"  Configured top_k: {top_k}, similarity_threshold: {similarity_threshold}")

            if level not in self.VALID_LEVELS:
                print(f"  [CEP DEBUG] Invalid level '{level}' provided.")
                return []

            relevant_strategies = []
            all_loaded_strategies = self.strategy_db.get(level, [])
            print(f"  Total strategies loaded for level '{level}': {len(all_loaded_strategies)}")

            if not all_loaded_strategies:
                 print(f"  [CEP DEBUG] No strategies loaded for level '{level}'. Returning empty list.")
                 return []

            if not scenarios:
                 print(f"  [CEP DEBUG] Input scenarios dictionary is empty. Cannot perform similarity search.")
                 return []

            # Prepare query texts from input scenarios
            query_texts = [text for text in scenarios.values() if text] # Filter out empty strings
            if not query_texts:
                 print(f"  [CEP DEBUG] No valid text found in input scenarios. Cannot perform similarity search.")
                 return []

            print(f"  Encoding {len(query_texts)} input scenario text(s)...")
            try:
                query_embeddings = self.embedding_model.encode(query_texts)
            except Exception as e:
                 print(f"  [CEP ERROR] Failed to encode query texts: {e}")
                 return []

            print(f"  Comparing against {len(all_loaded_strategies)} loaded strategies...")
            for i, strategy_data_dict in enumerate(all_loaded_strategies):
                try:
                    strategy_id = strategy_data_dict.get("item", {}).get("id", f"Unknown_ID_{i}")
                    strategy_scenario_dict = strategy_data_dict.get("item", {}).get("states_scenario", {})

                    # print(f"\\n  Comparing with Strategy ID: {strategy_id}")
                    # print(f"    Strategy states_scenario keys: {list(strategy_scenario_dict.keys())}")

                    strategy_texts = [text for text in strategy_scenario_dict.values() if text] # Filter out empty strings

                    if not strategy_texts:
                        # print(f"    Skipping strategy {strategy_id}: No valid text found in states_scenario.")
                        continue

                    # Encode strategy texts
                    strategy_embeddings = self.embedding_model.encode(strategy_texts)

                    # Calculate cosine similarity
                    # Shape: (num_query_texts, num_strategy_texts)
                    similarity_matrix = cosine_similarity(query_embeddings, strategy_embeddings)

                    # Calculate average similarity
                    # This simple average might not be ideal if dimensions mismatch,
                    # but follows the original logic for now.
                    if similarity_matrix.size == 0:
                         avg_similarity = 0.0
                    else:
                         avg_similarity = np.mean(similarity_matrix)


                    print(f"    - Strategy ID: {strategy_id} | Avg Similarity: {avg_similarity:.4f}") # Log similarity

                    if avg_similarity >= similarity_threshold:
                        print(f"      -> PASSED threshold ({similarity_threshold:.4f})")
                        strategy_data_dict['item']['similarity'] = float(avg_similarity) # Store similarity
                        relevant_strategies.append(strategy_data_dict)
                    # else:
                    #    print(f"      -> FAILED threshold ({similarity_threshold:.4f})")

                except Exception as e:
                     print(f"  [CEP ERROR] Error processing strategy {strategy_id}: {e}")
                     continue # Skip to next strategy on error

            # Sort by similarity score in descending order
            relevant_strategies.sort(key=lambda s: s['item']['similarity'], reverse=True)

            print(f"  Found {len(relevant_strategies)} strategies passing the threshold.")
            print(f"  Returning top {min(top_k, len(relevant_strategies))} strategies.")
            print(f"--- [CEP DEBUG] Retrieval for level '{level}' finished ---\\n")

            return relevant_strategies[:top_k]


# Usage example
if __name__ == "__main__":
    # Initialize plugin with absolute path to ensure correct location
    # Use the absolute path that was shown in the terminal output
    storage_path = ".MarketToM1/strategy_database"
    cep = CognitiveEnhancementPlugin(storage_path)
    
    # Example strategy insertion using JSON string
    # # 插入新策略
    # belief_id = cep.insert_strategy(
    #     level="belief",
    #     states_scenario={
    #         "environmental": "Market sentiment optimistic, volume increased"
    #     },
    #     strategy="Maintain bullish outlook based on volume analysis"
    # )

    # # 更新已有策略
    # updated_id = cep.update_strategy(
    #     level="belief",
    #     strategy_id="belief_20240326100000_0001",
    #     strategy="Adjust bullish outlook with stricter risk management"
    # )

    # 插入多状态策略
    # emotion_id = cep.insert_strategy(
    #     level="emotion",
    #     states_scenario={
    #         "belief": "Market trend is bullish",
    #         "environmental": "High trading volume"
    #     },
    #     strategy="Maintain confident but cautious emotional state"
    # )
    
    # 先列出所有belief级别的策略ID，看看有哪些可用
    print("\n可用的belief策略:")
    belief_strategies = cep.get_strategies_by_level("belief")
    if belief_strategies:
        for strategy in belief_strategies:
            print(f"ID: {strategy['item']['id']}")
    else:
        print("未找到任何belief策略")
        
    # 根据 level 和 id 获取特定策略
    # 修改为您在上面列表中看到的一个实际存在的ID
    strategy_id_to_find = "belief_20241105083648_0002"  # 这只是一个示例，请用您实际存在的ID替换
    strategy = cep.get_strategy_by_id(
        level="belief",
        strategy_id=strategy_id_to_find
    )

    # 检查结果
    if strategy:
        print(f"\nFound strategy:")
        print(f"ID: {strategy['item']['id']}")
        print(f"Strategy: {strategy['item']['strategy']}")
        print(f"Timestamp: {strategy['item']['timestamp']}")
    else:
        print(f"\nStrategy with ID {strategy_id_to_find} not found")