
import json
import os
import time
import logging
from typing import Dict, Optional, Any, List
import traceback
from openai import OpenAI
import random
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, field_validator
from core.cep import CognitiveEnhancementPlugin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='market_tom_analysis.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM.ErrorAnalysis')

COLOR_TITLE = "\033[1;36m"     # Cyan bold (titles)
COLOR_SUCCESS = "\033[1;32m"   # Green bold (success)
COLOR_WARNING = "\033[1;33m"   # Yellow bold (warnings)
COLOR_ERROR = "\033[1;31m"     # Red bold (errors)
COLOR_INFO = "\033[0;34m"      # Blue (info)
COLOR_VALUE = "\033[1;35m"     # Magenta bold (values)
COLOR_DEBUG = "\033[0;90m"     # Gray (debug)
COLOR_PHASE = "\033[1;94m"     # Light blue bold (phases)
COLOR_BELIEF = "\033[1;35m"    # Magenta bold (belief)
COLOR_INTENT = "\033[1;33m"    # Yellow bold (intent)
COLOR_EMOTION = "\033[1;36m"   # Cyan bold (emotion)
COLOR_CREATE = "\033[1;32m"    # Green bold (create)
COLOR_MODIFY = "\033[1;34m"    # Blue bold (modify)
COLOR_RESET = "\033[0m"        # Reset color
COLOR_STATES = "\033[0;33m"    # Yellow (states)
COLOR_CONTENT = "\033[0;36m"   # Cyan (content)


class StrategyUpdate(BaseModel):
    level: str = Field(..., description="Strategy level: belief, intent, or emotion")
    decision_type: str = Field(..., description="Operation type: CREATE or MODIFY")
    original_id: Optional[str] = Field(None, description="Original strategy ID (required only for MODIFY operations)")
    content: str = Field(..., description="Strategy content")
    
    @field_validator('level')
    def validate_level(cls, v):
        if v.lower() not in ['belief', 'intent', 'emotion']:
            raise ValueError('Level must be belief, intent, or emotion')
        return v.lower()
    
    @field_validator('decision_type')
    def validate_decision_type(cls, v):
        if v.upper() not in ['CREATE', 'MODIFY']:
            raise ValueError('Decision type must be CREATE or MODIFY')
        return v.upper()
    
    @field_validator('original_id')
    def validate_original_id(cls, v, info):
        values = {}
        for field_name, field in cls.model_fields.items():
            if field_name in info.data:
                values[field_name] = info.data[field_name]
        
        if values.get('decision_type') == 'MODIFY' and (v is None or v.strip() == ''):
            raise ValueError('MODIFY operation requires an original_id')
        
        if v is not None and v.strip() != '':
            parts = v.split('_')
            if len(parts) != 3:
                raise ValueError(f'Invalid strategy ID format: {v}, should be <level>_<timestamp>_<id>')
            
            level = values.get('level', '').lower()
            if parts[0] != level:
                raise ValueError(f'Strategy ID level part ({parts[0]}) does not match the specified level ({level})')
            
            if not (len(parts[1]) == 14 and parts[1].isdigit()):
                raise ValueError(f'Strategy ID timestamp part ({parts[1]}) is invalid, should be 14 digits')
                
            if not (len(parts[2]) == 4 and parts[2].isdigit()):
                raise ValueError(f'Strategy ID number part ({parts[2]}) is invalid, should be 4 digits')
        
        return v


class BackwardInferenceResponse(BaseModel):
    strategy_updates: List[StrategyUpdate] = Field(default_factory=list)


last_api_request_time = datetime.now() - timedelta(seconds=10)
MIN_REQUEST_INTERVAL = 20.
DEFAULT_COOLDOWN = 20.0 
MAX_JITTER = 1.0


def rate_limit_api_call(func):
    def wrapper(*args, **kwargs):
        global last_api_request_time
        
        now = datetime.now()
        time_since_last_request = (now - last_api_request_time).total_seconds()
        
        if time_since_last_request < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last_request + random.uniform(0, MAX_JITTER)
            logger.info(f"Rate limiting: Waiting {wait_time:.2f}s before next API call")
            time.sleep(wait_time)
        
        last_api_request_time = datetime.now()
        
        result = func(*args, **kwargs)

        cooldown = DEFAULT_COOLDOWN + random.uniform(0, MAX_JITTER)
        logger.info(f"API call completed. Cooling down for {cooldown:.2f}s")
        time.sleep(cooldown)
        
        return result
    return wrapper


class BackwardInference:
    def __init__(self, 
                cep: CognitiveEnhancementPlugin, 
                llm_client: OpenAI,
                llm_model: str,
                backward_template_abs_path: str,
                inference_logs_abs_path: str,
                max_retries: int,
                base_delay_seconds: float,
                llm_temperature: float,
                llm_max_tokens: int):
        
        self.cep = cep
        self.llm_client = llm_client
        self.llm_model = llm_model 
        self.backward_template_abs_path = backward_template_abs_path
        self.inference_logs_abs_path = inference_logs_abs_path 
        self.max_retries = max_retries
        self.base_delay = base_delay_seconds
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        
        # 创建反向推理日志目录
        self.backward_logs_dir = os.path.join(os.path.dirname(inference_logs_abs_path), "backward_inference_logs")
        os.makedirs(self.backward_logs_dir, exist_ok=True)
        
        logger.info("Backward inference system initialized")
        logger.info(f"Using template: {self.backward_template_abs_path}")
        logger.info(f"Using inference logs path: {self.inference_logs_abs_path}")
        logger.info(f"Backward inference logs will be saved to: {self.backward_logs_dir}")

    def _save_backward_inference_log(self, 
                                    timestamp: str,
                                    predicted_action: str,
                                    actual_action: str,
                                    inference_filename: str,
                                    strategy_updates: List[Dict],
                                    analysis_result: str) -> str:
        """
        保存反向推理日志
        
        Args:
            timestamp: 推理时间戳
            predicted_action: 预测动作
            actual_action: 实际动作
            inference_filename: 原始推理文件名
            strategy_updates: 策略更新列表
            analysis_result: LLM分析结果
            
        Returns:
            保存的日志文件路径
        """
        log_entry = {
            "timestamp": timestamp,
            "prediction_error": {
                "predicted_action": predicted_action,
                "actual_action": actual_action
            },
            "original_inference_file": inference_filename,
            "strategy_updates": strategy_updates,
            "llm_analysis": analysis_result,
            "backward_inference_timestamp": datetime.now().isoformat()
        }
        
        # 生成文件名：backward_YYYYMMDD_HHMMSS.json
        dt = datetime.now()
        filename = f"backward_{dt.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.backward_logs_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved backward inference log to: {filepath}")
        return filepath

    def _load_inference_result(self, filename: str) -> Dict[str, Any]:
        filepath = os.path.join(self.inference_logs_abs_path, filename)
        logger.info(f"Loading inference log from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key in ['mental_states', 'strategies_used', 'environmental_state']:
                if key not in data:
                    logger.warning(f"Log file {filename} missing expected key '{key}'")
            return data
        except FileNotFoundError:
            logger.error(f"Inference result file not found at {filepath}")
            raise FileNotFoundError(f"Inference result file not found at {filepath}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in inference result file: {filepath}")
            raise 
        except Exception as e:
            logger.error(f"Error loading inference file {filepath}: {str(e)}")
            raise

    def _load_prompt_template(self) -> str:
        logger.info(f"Loading backward template from: {self.backward_template_abs_path}")
        try:
            with open(self.backward_template_abs_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {self.backward_template_abs_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading prompt template file {self.backward_template_abs_path}: {str(e)}")
            raise

    def _get_strategy_text(self, level: str, strategy_id: Optional[str]) -> str:
        """Helper to get strategy text by ID, returns placeholder if not found."""
        if not strategy_id:
            return f"No {level} strategy used or ID missing"
        
        strategy_data = self.cep.get_strategy_by_id(level, strategy_id)
        if strategy_data and "item" in strategy_data and "strategy" in strategy_data["item"]:
            return strategy_data["item"]["strategy"]
        else:
            logger.warning(f"Strategy text not found for ID: {strategy_id} in level: {level}")
            return f"Strategy text not found for {level} ID: {strategy_id}"

    def _rebuild_states_scenario(self, level: str, inference_data: Dict) -> Dict[str, str]:
        """Rebuild state scenario needed for strategies"""
        scenario = {}
        mental_states = inference_data.get('mental_states', {})
        env_state = inference_data.get('environmental_state', "[ENVIRONMENTAL STATE NOT FOUND]")
        # Define dependencies based on the CBN structure described in the prompt
        # Environmental States -> Belief
        # Belief -> Intent
        # Belief + Environmental States -> Emotion
        try:
            logger.debug(f"Rebuilding state scenario for {level} level")
            if level == "belief":
                scenario["environmental"] = env_state
            elif level == "intent":
                scenario["belief"] = mental_states.get('belief', "[BELIEF NOT FOUND]")
            elif level == "emotion":
                scenario["belief"] = mental_states.get('belief', "[BELIEF NOT FOUND]")
                scenario["environmental"] = env_state
            else:
                logger.warning(f"Unknown level '{level}', returning empty scenario")

        except KeyError as e:
             logger.warning(f"Missing key '{e}' in mental_states when rebuilding scenario for '{level}' level")
             if level == "emotion":
                  if "belief" not in scenario: scenario["belief"] = "[BELIEF NOT FOUND]"
                  if "environmental" not in scenario: scenario["environmental"] = "[ENVIRONMENTAL STATE NOT FOUND]" 
        except Exception as e:
            logger.error(f"Error rebuilding scenario for level '{level}': {str(e)}")

        try:
            from core.cep import StrategyData
            required_states = [dep.replace('_states','') for dep in StrategyData.STATE_RELATIONSHIPS.get(level, [])]
            missing_states = [state for state in required_states if state not in scenario]
            if missing_states:
                logger.warning(f"{level} scenario missing required states: {missing_states}")
                logger.info("This may cause strategy creation to fail")
                for state in missing_states:
                    if state == "environmental":
                        scenario[state] = env_state
                    elif state in mental_states:
                        scenario[state] = mental_states.get(state)
                    else:
                        placeholder = f"[Auto-generated {state.upper()} state description]"
                        scenario[state] = placeholder
                        logger.debug(f"Added placeholder: {state} = {placeholder}")
        except Exception as e:
            logger.debug(f"Error checking required states: {str(e)}")

        keys = list(scenario.keys())
        logger.debug(f"Rebuilt scenario for {level} contains keys: {keys}")
        return scenario

    @rate_limit_api_call
    def _call_backward_llm(self, prompt_input: Dict[str, str]) -> Optional[str]:
        try:
            template = self._load_prompt_template()
            formatted_prompt = template
            for key, value in prompt_input.items():
                    str_value = str(value) if value is not None else f"[{key} VALUE MISSING]"
                    placeholder = f"[{key}]"
                    formatted_prompt = formatted_prompt.replace(placeholder, str_value)
        except Exception as e:
            logger.warning(f"Failed to format backward inference template: {str(e)}")
            return None

        for attempt in range(self.max_retries):
            try:
                print(f"\n{COLOR_PHASE}STEP 2: ANALYZING PREDICTION ERROR{COLOR_RESET}")
                logger.info(f"Calling LLM for backward inference analysis (attempt {attempt+1}/{self.max_retries})")
                logger.info(f"Using model: {self.llm_model}, temperature: {self.llm_temperature}")
                
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model, 
                    messages=[
                        {"role": "system", "content": formatted_prompt}
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens
                )
                logger.info("LLM call successful")
                print(f"{COLOR_SUCCESS}✓ Analysis completed{COLOR_RESET}")
                llm_content = response.choices[0].message.content.strip()
                
                try:
                    json_data = json.loads(llm_content)
                    has_strategy_updates = "strategy_updates" in json_data
                    
                    if has_strategy_updates:
                        strategy_count = len(json_data.get("strategy_updates", []))
                        logger.info(f"Received {strategy_count} strategy update suggestions")
                                
                except json.JSONDecodeError:
                    logger.warning("Response is not valid JSON format")
                
                return llm_content
            except Exception as e:
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5) 
                logger.warning(f"LLM API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                logger.info(f"Retrying in {delay:.2f} seconds")
                time.sleep(delay)
        
        return None
        
    def _process_llm_backward_response(self, llm_response: str, inference_data: Dict) -> List[Dict]:
        """Process LLM analysis result using Pydantic, create or update strategies
        
        Returns:
            List of strategy updates that were processed
        """
        inserted_count = 0
        updated_count = 0
        
        belief_color = COLOR_BELIEF
        intent_color = COLOR_INTENT
        emotion_color = COLOR_EMOTION
        create_color = COLOR_CREATE
        modify_color = COLOR_MODIFY
        
        modified_strategies = {
            "belief": {"create": [], "modify": []},
            "intent": {"create": [], "modify": []},
            "emotion": {"create": [], "modify": []}
        }
        
        try:
            json_data = None
            
            try:
                try:
                    json_data = json.loads(llm_response)
                    logger.debug("Successfully parsed complete JSON response")
                except json.JSONDecodeError:
                    pass
                
                if not json_data and "```json" in llm_response:
                    json_start = llm_response.find("```json") + 7
                    json_end = llm_response.find("```", json_start)
                    if json_end > json_start:
                        json_text = llm_response[json_start:json_end].strip()
                        try:
                            json_data = json.loads(json_text)
                            logger.debug("Successfully extracted JSON from code block")
                        except json.JSONDecodeError:
                            pass
                
                if not json_data and "```" in llm_response:
                    json_start = llm_response.find("```") + 3
                    if "json" in llm_response[json_start:json_start+10]:
                        json_start = llm_response.find("\n", json_start) + 1
                    json_end = llm_response.find("```", json_start)
                    if json_end > json_start:
                        json_text = llm_response[json_start:json_end].strip()
                        try:
                            json_data = json.loads(json_text)
                            logger.debug("Successfully extracted JSON from code block")
                        except json.JSONDecodeError:
                            pass
                
                if not json_data:
                    first_brace = llm_response.find("{")
                    last_brace = llm_response.rfind("}")
                    if first_brace != -1 and last_brace > first_brace:
                        json_text = llm_response[first_brace:last_brace + 1]
                        try:
                            json_data = json.loads(json_text)
                            logger.debug("Successfully extracted JSON from braces")
                        except json.JSONDecodeError:
                            pass
                
                if not json_data:
                    logger.warning("Unable to extract valid JSON from LLM response")
                    return
            except Exception as e:
                logger.warning(f"Error extracting JSON: {str(e)}")
                return
            
            try:
                if "strategy_updates" in json_data and isinstance(json_data["strategy_updates"], list):
                    filtered_updates = []
                    for i, strategy in enumerate(json_data["strategy_updates"]):
                        level = strategy.get("level", "").lower()
                        if level == "action":
                            logger.warning(f"Skipping strategy #{i+1}: 'action' is not a valid update level")
                            continue
                        else:
                            filtered_updates.append(strategy)
                        
                    if len(filtered_updates) != len(json_data["strategy_updates"]):
                        logger.debug(f"Filtered out {len(json_data['strategy_updates']) - len(filtered_updates)} invalid level strategy updates")
                        json_data["strategy_updates"] = filtered_updates
                
                try:
                    response_obj = BackwardInferenceResponse.model_validate(json_data)
                    strategy_updates = response_obj.strategy_updates
                except Exception as first_parse_error:
                    logger.warning(f"Failed to parse complete response: {str(first_parse_error)}")
                    logger.debug("Attempting partial parsing...")
                    
                    strategy_updates = []
                    if "strategy_updates" in json_data and isinstance(json_data["strategy_updates"], list):
                        for i, strategy_data in enumerate(json_data["strategy_updates"]):
                            try:
                                strategy = StrategyUpdate(**strategy_data)
                                strategy_updates.append(strategy)
                            except Exception as e:
                                logger.warning(f"Strategy #{i+1} parsing failed: {str(e)}")
                    else:
                        logger.warning("No valid strategy update list found")
                
                print(f"\n{COLOR_PHASE}STEP 3: IMPLEMENTING IMPROVEMENTS{COLOR_RESET}")
                logger.info(f"Parsed {len(strategy_updates)} strategy updates")
                
                if len(strategy_updates) > 0:
                    print(f"{COLOR_INFO}Found {len(strategy_updates)} possible strategy improvements{COLOR_RESET}")
                
                for i, update in enumerate(strategy_updates):
                    if update.decision_type == "MODIFY" and update.original_id:
                        strategy_exists = self.cep.get_strategy_by_id(update.level, update.original_id)
                        if not strategy_exists:
                            logger.warning(f"Strategy ID '{update.original_id}' not found in '{update.level}' level")
                            logger.info("Changing operation from MODIFY to CREATE")
                            strategy_updates[i].decision_type = "CREATE"
                            strategy_updates[i].original_id = None
                
                success_count = 0
                total_count = len(strategy_updates)
                
                for update in strategy_updates:
                    level = update.level
                    decision_type = update.decision_type
                    content = update.content
                    
                    level_color = belief_color
                    if level == "intent":
                        level_color = intent_color
                    elif level == "emotion":
                        level_color = emotion_color
                    
                    op_color = create_color if decision_type == 'CREATE' else modify_color
                    op_text = "Creating new" if decision_type == 'CREATE' else "Updating"
                    
                    if decision_type == 'CREATE':
                        logger.info(f"Attempting to create new {level} strategy")
                        scenario = self._rebuild_states_scenario(level, inference_data)
                        if not scenario:
                            logger.error(f"Failed to rebuild scenario for {level} strategy")
                            continue
                        
                        try:
                            logger.debug(f"Strategy content: {content[:100]}...")
                            logger.debug(f"Scenario content: {scenario}")
                            
                            try:
                                from core.cep import StrategyData
                                required_states = [dep.replace('_states','') for dep in StrategyData.STATE_RELATIONSHIPS.get(level, [])]
                                missing = [state for state in required_states if state not in scenario]
                                if missing:
                                    logger.warning(f"{level} scenario missing required states: {missing}")
                                    logger.info("Attempting to add default values...")
                                    for state in missing:
                                        scenario[state] = f"[Auto-filled {state.upper()} state]"
                            except Exception as e:
                                logger.debug(f"Error checking required states: {str(e)}")
                            
                            new_id = self.cep.insert_strategy(
                                level=level,
                                states_scenario=scenario,
                                strategy=content
                            )
                            if new_id:
                                print(f"{op_color}✓ Created new {level_color}{level}{COLOR_RESET} strategy{COLOR_RESET}")
                                print(f"\n{COLOR_TITLE}┌─ NEW {level.upper()} STRATEGY ─────────────────────{COLOR_RESET}")
                                print(f"{level_color}{content}{COLOR_RESET}")
                                print(f"{COLOR_TITLE}└────────────────────────────────────────────{COLOR_RESET}\n")
                                inserted_count += 1
                                modified_strategies[level]["create"].append(new_id)
                                success_count += 1
                                logger.info(f"Created new {level} strategy with ID: {new_id}")
                            else:
                                logger.error(f"Failed to insert {level} strategy, CEP returned None")
                                logger.info("Possible causes: 1. Invalid scenario data 2. Invalid strategy content 3. CEP internal validation failure")
                        except Exception as e:
                            logger.error(f"Error creating {level} strategy: {str(e)}")
                            logger.error(traceback.format_exc())
                            
                    elif decision_type == 'MODIFY':
                        original_id = update.original_id
                        if not original_id:
                            logger.warning("Missing original strategy ID, cannot perform modification")
                            continue
                        
                        strategy_data = self.cep.get_strategy_by_id(level, original_id)
                        if not strategy_data:
                            logger.warning(f"Strategy ID '{original_id}' not found in '{level}' level, cannot modify")
                            logger.info("Converting operation from MODIFY to CREATE")
                            
                            scenario = self._rebuild_states_scenario(level, inference_data)
                            if not scenario:
                                logger.error(f"Failed to rebuild scenario for {level} strategy")
                                continue
                                
                            try:
                                logger.debug(f"Strategy content: {content[:100]}...")
                                logger.debug(f"Scenario content: {scenario}")

                                try:
                                    from core.cep import StrategyData
                                    required_states = [dep.replace('_states','') for dep in StrategyData.STATE_RELATIONSHIPS.get(level, [])]
                                    missing = [state for state in required_states if state not in scenario]
                                    if missing:
                                        logger.warning(f"{level} scenario missing required states: {missing}")
                                        logger.info("Attempting to add default values...")
                                        for state in missing:
                                            scenario[state] = f"[Auto-filled {state.upper()} state]"
                                except Exception as e:
                                    logger.debug(f"Error checking required states: {str(e)}")
                                
                                new_id = self.cep.insert_strategy(
                                    level=level,
                                    states_scenario=scenario,
                                    strategy=content
                                )
                                if new_id:
                                    print(f"{create_color}✓ Created new {level_color}{level}{COLOR_RESET} strategy{COLOR_RESET}")
                                    print(f"\n{COLOR_TITLE}┌─ NEW {level.upper()} STRATEGY ─────────────────────{COLOR_RESET}")
                                    print(f"{level_color}{content}{COLOR_RESET}")
                                    print(f"{COLOR_TITLE}└────────────────────────────────────────────{COLOR_RESET}\n")
                                    logger.info(f"Created new {level} strategy (originally planned to modify {original_id}): {new_id}")
                                    inserted_count += 1
                                    modified_strategies[level]["create"].append(new_id)
                                    success_count += 1
                                else:
                                    logger.error(f"Failed to insert {level} strategy, CEP returned None")
                            except Exception as e:
                                logger.error(f"Error creating {level} strategy: {str(e)}")
                                logger.error(traceback.format_exc())
                            continue
                        
                        try:
                            logger.debug(f"Attempting to update {level} strategy with ID: {original_id}")
                            
                            original_content = None
                            if strategy_data and "item" in strategy_data and "strategy" in strategy_data["item"]:
                                original_content = strategy_data["item"]["strategy"]
                            
                            updated_id = self.cep.update_strategy(
                                level=level,
                                strategy_id=original_id,
                                strategy=content
                            )
                            if updated_id:
                                print(f"{op_color}✓ Updated {level_color}{level}{COLOR_RESET} strategy{COLOR_RESET}")
                                print(f"\n{COLOR_TITLE}┌─ UPDATED {level.upper()} STRATEGY ─────────────────────{COLOR_RESET}")
                                print(f"{level_color}{content}{COLOR_RESET}")
                                print(f"{COLOR_TITLE}└────────────────────────────────────────────{COLOR_RESET}\n")
                                updated_count += 1
                                modified_strategies[level]["modify"].append(updated_id)
                                success_count += 1
                                logger.info(f"Updated {level} strategy: {updated_id}")
                            else:
                                logger.warning(f"Failed to update {level} strategy, CEP returned None")
                        except Exception as e:
                            logger.warning(f"Error updating {level} strategy: {str(e)}")
                            logger.error(traceback.format_exc())
            
            except Exception as e:
                logger.error(f"Error parsing JSON with Pydantic: {str(e)}")
                logger.error(traceback.format_exc())
                return
                
            any_changes = False
            for level, actions in modified_strategies.items():
                if actions["create"] or actions["modify"]:
                    any_changes = True
            
            if any_changes:
                print(f"\n{COLOR_SUCCESS}✓ Successfully implemented {success_count} improvements{COLOR_RESET}")
            
            # 🆕 返回按level分组的策略更新（用于前端显示）
            grouped_updates = {}
            for level, actions in modified_strategies.items():
                level_updates = []
                
                # 创建的策略
                for strategy_id in actions["create"]:
                    strategy_data = self.cep.retrieve_strategy_by_id(level, strategy_id)
                    if strategy_data and "item" in strategy_data:
                        level_updates.append({
                            "type": "创建",
                            "id": strategy_id,
                            "content": strategy_data["item"].get("strategy", "N/A")
                        })
                
                # 修改的策略
                for strategy_id in actions["modify"]:
                    strategy_data = self.cep.retrieve_strategy_by_id(level, strategy_id)
                    if strategy_data and "item" in strategy_data:
                        level_updates.append({
                            "type": "修改",
                            "id": strategy_id,
                            "content": strategy_data["item"].get("strategy", "N/A")
                        })
                
                if level_updates:
                    grouped_updates[level] = level_updates
            
            return grouped_updates
                    
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def perform_backward_inference(self, filename: str, predicted_action: str, actual_action: str) -> Optional[str]:
        """Perform backward inference analysis on prediction errors"""
        try:
            print(f"\n{COLOR_TITLE}=== ANALYZING PREDICTION ERROR ==={COLOR_RESET}")
            print(f"{COLOR_PHASE}STEP 1: GATHERING INFORMATION{COLOR_RESET}")
            
            log_data = self._load_inference_result(filename)
            mental_states = log_data.get("mental_states", {})
            strategies_used = log_data.get("strategies_used", {})
            env_state = log_data.get("environmental_state", "")
            
            logger.debug(f"Found strategy IDs: Belief:{strategies_used.get('belief', [])} Intent:{strategies_used.get('intent', [])} Emotion:{strategies_used.get('emotion', [])}")
            
            belief_state = mental_states.get("belief", "")
            intent_state = mental_states.get("intent", "")
            emotion_state = mental_states.get("emotion", "")

            belief_ids = strategies_used.get('belief', [])
            intent_ids = strategies_used.get('intent', [])
            emotion_ids = strategies_used.get('emotion', [])

            belief_strategy_text = self._get_strategy_text('belief', belief_ids[0]) if belief_ids else "No strategy available."
            intent_strategy_text = self._get_strategy_text('intent', intent_ids[0]) if intent_ids else "No strategy available."
            emotion_strategy_text = self._get_strategy_text('emotion', emotion_ids[0]) if emotion_ids else "No strategy available."

            prompt_input = {
                "ENVIRONMENTAL_STATE": env_state,
                "BELIEF_STATE": belief_state,
                "INTENTION_STATE": intent_state,
                "EMOTION_STATE": emotion_state,
                "PREDICTED_ACTION": predicted_action,
                "ACTUAL_ACTION": actual_action,
                "BELIEF_STRATEGY": belief_strategy_text,
                "INTENTION_STRATEGY": intent_strategy_text,
                "EMOTION_STRATEGY": emotion_strategy_text
            }

            if belief_ids:
                prompt_input["BELIEF_STRATEGY_ID"] = belief_ids[0]
            if intent_ids:
                prompt_input["INTENTION_STRATEGY_ID"] = intent_ids[0]
            if emotion_ids:
                prompt_input["EMOTION_STRATEGY_ID"] = emotion_ids[0]

            print(f"{COLOR_INFO}Analyzing why the prediction was incorrect...{COLOR_RESET}")
            llm_analysis_text = self._call_backward_llm(prompt_input)

            if not llm_analysis_text:
                 logger.error("Analysis failed")
                 print(f"{COLOR_ERROR}✗ Analysis failed{COLOR_RESET}")
                 return None

            # 处理LLM响应并提取策略更新
            strategy_updates = self._process_llm_backward_response(llm_analysis_text, log_data)
            
            # 保存反向推理日志
            timestamp = log_data.get("timestamp", datetime.now().isoformat())
            self._save_backward_inference_log(
                timestamp=timestamp,
                predicted_action=predicted_action,
                actual_action=actual_action,
                inference_filename=filename,
                strategy_updates=strategy_updates,
                analysis_result=llm_analysis_text
            )
            
            # 🆕 返回包含策略更新的完整结果
            return {
                'analysis': llm_analysis_text,
                '策略库更新': strategy_updates or {}
            }

        except Exception as e:
             logger.error(f"Backward inference error: {str(e)}")
             print(f"{COLOR_ERROR}✗ Error during analysis{COLOR_RESET}")
             return None
