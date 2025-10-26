from typing import List, Dict, Any, Tuple, Optional
import json
import os
import numpy as np
from pydantic import BaseModel, Field, field_validator
import openai
import time
import re
import logging
from core.cep import CognitiveEnhancementPlugin
from datetime import datetime, timedelta
import random
import traceback
from core.expert_perspectives import get_random_perspectives

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='market_tom_probability.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM.Probability')

COLOR_TITLE = "\033[1;36m"     # Cyan bold (titles)
COLOR_SUCCESS = "\033[1;32m"   # Green bold (success)
COLOR_WARNING = "\033[1;33m"   # Yellow bold (warnings)
COLOR_ERROR = "\033[1;31m"     # Red bold (errors)
COLOR_INFO = "\033[0;34m"      # Blue (info)
COLOR_VALUE = "\033[1;35m"     # Magenta bold (values)
COLOR_DEBUG = "\033[0;90m"     # Gray (debug)
COLOR_PHASE = "\033[1;94m"     # Light blue bold (phases)
COLOR_RESET = "\033[0m"        # Reset color
COLOR_EXPERT = "\033[1;33m"    # Yellow bold (expert)
COLOR_REASONING = "\033[0;36m" # Cyan (reasoning)

last_api_request_time = datetime.now() - timedelta(seconds=10)
MIN_REQUEST_INTERVAL = 20.0
DEFAULT_COOLDOWN = 20.0
MAX_JITTER = 2.0

def rate_limit_api_call(func):
    """Decorator: Control API call frequency to prevent rate limits"""
    def wrapper(*args, **kwargs):
        global last_api_request_time
        now = datetime.now()
        time_since_last_request = (now - last_api_request_time).total_seconds()
        if time_since_last_request < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last_request + random.uniform(0, MAX_JITTER)
            logger.info(f"Rate limiting: Waiting {wait_time:.2f}s before next API call")
            time.sleep(wait_time)
        last_api_request_time = datetime.now()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            cooldown = DEFAULT_COOLDOWN + random.uniform(0, MAX_JITTER)
            logger.info(f"API call failed or errored. Cooling down for {cooldown:.2f}s")
            time.sleep(cooldown)
            raise e
        else:
            cooldown = DEFAULT_COOLDOWN + random.uniform(0, MAX_JITTER)
            logger.info(f"API call completed. Cooling down for {cooldown:.2f}s")
            time.sleep(cooldown)
            return result
    return wrapper

class ProbabilitySample(BaseModel):
    """Individual probability sample model"""
    value: float = Field(..., description="Probability value (between 0-1)")
    log_confidence: float = Field(..., description="Confidence based on token log-probability")
    normalized_weight: float = Field(0.0, description="Normalized weight")
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Probability value must be between 0 and 1")
        return v

class ProbabilityResult(BaseModel):
    """Final probability calculation result model"""
    probability: float = Field(..., description="Weighted aggregated upward probability")
    samples: List[ProbabilitySample] = Field(..., description="List of probability samples")
    strategy_ids: List[str] = Field(default_factory=list, description="List of used strategy IDs")
    inference_id: str = Field("", description="Inference ID")
    timestamp: str = Field("", description="Timestamp")
    environmental_state: str = Field("", description="Environmental state")
    expert_details: List[dict] = Field(default_factory=list, description="Expert roles and reasoning")

class ProbabilityResponse(BaseModel):
    """Pydantic model for parsing probability value list from LLM response"""
    probabilities: List[float] = Field(..., description="List of probability estimates, each representing the probability of market going up")
    
    @field_validator('probabilities')
    def validate_probabilities(cls, v):
        for prob in v:
            if not (0 <= prob <= 1):
                raise ValueError(f"Probability values must be between 0-1, got: {prob}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"probabilities": [0.65, 0.72, 0.58, 0.61, 0.69]}
            ]
        }
    }

class ExpertProbabilityResponse(BaseModel):
    """Single expert's probability assessment and reasoning"""
    reasoning: str = Field(..., description="Expert's analysis and reasoning")
    probability: float = Field(..., description="Expert's estimated upward probability")
    
    @field_validator('probability')
    def validate_probability(cls, v):
        if not (0 <= v <= 1):
            raise ValueError(f"Probability values must be between 0-1, got: {v}")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "reasoning": "From a technical analysis perspective, the market has broken through important resistance levels with increasing volume, indicating strengthening bullish momentum.",
                    "probability": 0.75
                }
            ]
        }
    }

class ExpertInfo:
    """Store expert information including role, reasoning, and probability sample"""
    def __init__(self, role: str, reasoning: str, probability_sample: ProbabilitySample):
        self.role = role
        self.reasoning = reasoning
        self.probability_sample = probability_sample

class ActionProbabilityCalculator:
    """Market action probability calculator using Log-Confidence Weighting algorithm"""
    def __init__(self,
                 cep: CognitiveEnhancementPlugin,
                 llm_client: openai.OpenAI,
                 llm_model: str,
                 action_prob_template_abs_path: str,
                 inference_logs_abs_path: str,
                 action_prob_top_k: int,
                 num_probs_to_generate: int,
                 max_retries_list: int,
                 base_delay_list_seconds: float,
                 llm_temperature: float = 0.7,
                 expert_prob_method: bool = False,
                 expert_template_abs_path: str = None,
                 **kwargs):
        """Initialize the market action probability calculator"""
        self.cep = cep
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.action_prob_template_abs_path = action_prob_template_abs_path
        self.inference_logs_abs_path = inference_logs_abs_path
        self.action_prob_top_k = action_prob_top_k
        self.num_probs_to_generate = num_probs_to_generate
        self.max_retries = max_retries_list
        self.base_delay = base_delay_list_seconds
        self.llm_temperature = llm_temperature
        self.expert_prob_method = expert_prob_method
        self.expert_template_abs_path = expert_template_abs_path
        
        method_name = "Expert Perspective" if self.expert_prob_method else "Log-Confidence Weighting"
        print(f"{COLOR_INFO}Probability calculator initialized with {COLOR_VALUE}{method_name}{COLOR_RESET} algorithm")
        logger.info(f"Initialized with {method_name} algorithm")
        
        if self.expert_prob_method and not self.expert_template_abs_path:
            print(f"{COLOR_WARNING}Warning: Expert method enabled but no expert template path provided!{COLOR_RESET}")
            logger.warning("Expert method enabled but no expert template path provided")

    def _load_prompt_template(self) -> str:
        """Load action probability prompt template from absolute path"""
        template_path = self.expert_template_abs_path if self.expert_prob_method else self.action_prob_template_abs_path
        
        if self.expert_prob_method:
            logger.debug(f"Loading expert template from: {template_path}")
            if not os.path.exists(template_path):
                print(f"{COLOR_ERROR}ERROR: Expert template file not found!{COLOR_RESET}")
                if os.path.exists(self.action_prob_template_abs_path):
                    print(f"{COLOR_WARNING}Falling back to standard template{COLOR_RESET}")
                    template_path = self.action_prob_template_abs_path
        else:
            logger.debug(f"Loading standard template from: {template_path}")
            
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                first_lines = "\n".join(template_content.split("\n")[:3]) + "..."
                logger.debug(f"Template content (first few lines): {first_lines}")
                return template_content
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {template_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading prompt template file: {str(e)}")
            raise

    def load_inference_log(self, filename: str) -> Dict[str, Any]:
        """Load inference log"""
        filepath = os.path.join(self.inference_logs_abs_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'mental_states' not in data:
                raise ValueError(f"File {filepath} is missing 'mental_states' field.")
            if 'intent' not in data['mental_states'] or 'emotion' not in data['mental_states']:
                raise ValueError(f"File {filepath} 'mental_states' is missing 'intent' or 'emotion'.")
            return data
        except Exception as e:
            logger.error(f"Failed to read inference log file {filepath}: {str(e)}")
            raise

    def calculate_probability_from_file(self, filename: str) -> ProbabilityResult:
        """Calculate action probability from inference log file"""
        print(f"\n{COLOR_TITLE}=== CALCULATING MARKET MOVEMENT PROBABILITY ===={COLOR_RESET}")
        logger.info(f"Calculating probability from file: {filename}")
        
        print(f"{COLOR_DEBUG}Using {'Expert Perspective' if self.expert_prob_method else 'Log-Confidence Weighting'} method{COLOR_RESET}")
        
        try:
            data = self.load_inference_log(filename)
            mental_states = data.get('mental_states', {})
            intent_desc = mental_states.get('intent', '')
            emotion_desc = mental_states.get('emotion', '')
            env_state = data.get('environmental_state', '')

            strategy_objects = []
            logger.info("Calculating probability directly from intent and emotion, no strategy retrieval")
            
            strategy_ids = []
            
            strategies_text = "Based on the market's current intent and emotion, determine the probability of an upward trend."
            
            if self.expert_prob_method:
                print(f"{COLOR_DEBUG}Entering expert perspective method{COLOR_RESET}")
                result = self._calculate_probability_expert(intent_desc, emotion_desc)
                print(f"{COLOR_DEBUG}Completed expert perspective method{COLOR_RESET}")
            else:
                print(f"{COLOR_DEBUG}Entering log-confidence weighting method{COLOR_RESET}")
                result = self._calculate_probability(intent_desc, emotion_desc, strategies_text, strategy_ids)
                print(f"{COLOR_DEBUG}Completed log-confidence weighting method{COLOR_RESET}")

            result.inference_id = filename.replace('.json', '')
            result.timestamp = data.get('timestamp', datetime.now().isoformat())
            result.environmental_state = env_state
            
            return result

        except Exception as e:
            logger.error(f"Error calculating probability: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _format_strategies_text(self, strategy_objects: List[Dict]) -> str:
        """Format strategy objects into text"""
        if not strategy_objects:
            return "No relevant strategies retrieved."
            
        strategy_texts = []
        for i, strat_obj in enumerate(strategy_objects, 1):
            try:
                item = strat_obj.get('item', {})
                strategy_text = item.get('strategy', 'Strategy text not available')
                scenario = item.get('states_scenario', {})
                strategy_texts.append(f"Strategy {i}:\n{strategy_text}\n\nScenario: {json.dumps(scenario, ensure_ascii=False)}")
            except Exception as e:
                strategy_texts.append(f"Strategy {i}: Error formatting strategy - {str(e)}")
                
        return "\n\n".join(strategy_texts)

    def _calculate_probability(self, intent_desc: str, emotion_desc: str, strategies_text: str, strategy_ids: List[str]) -> ProbabilityResult:
        """Implement Log-Confidence Weighting algorithm"""
        print(f"\n{COLOR_PHASE}STEP 1: GATHERING MULTIPLE PROBABILITY ESTIMATES{COLOR_RESET}")
        logger.info(f"Generating {self.num_probs_to_generate} probability samples")
        probability_samples = self._generate_probability_samples(intent_desc, emotion_desc, strategies_text)
        
        if not probability_samples:
            logger.warning("No valid probability samples generated, returning default value 0.5")
            print(f"{COLOR_WARNING}Unable to generate valid estimates, using default value.{COLOR_RESET}")
            return ProbabilityResult(
                probability=0.5,
                samples=[ProbabilitySample(value=0.5, log_confidence=-100.0, normalized_weight=1.0)],
                strategy_ids=strategy_ids
            )
        
        print(f"\n{COLOR_PHASE}STEP 2: CALCULATING WEIGHTED PROBABILITY{COLOR_RESET}")
        log_confidences = [sample.log_confidence for sample in probability_samples]
        probabilities = [sample.value for sample in probability_samples]
        
        logger.debug(f"Raw probability values: {probabilities}")
        logger.debug(f"Raw log-confidence values: {log_confidences}")
        
        weights = self._softmax(log_confidences)
        
        logger.debug(f"Softmax weights: {[f'{w:.4f}' for w in weights]}")
        
        weighted_contributions = []
        for i, (prob, weight) in enumerate(zip(probabilities, weights)):
            contribution = prob * weight
            weighted_contributions.append(contribution)
            logger.debug(f"Sample {i+1}: Probability={prob:.4f} × Weight={weight:.4f} = Contribution {contribution:.4f}")
        
        weighted_sum = sum(weighted_contributions)
        logger.info(f"Weighted probability sum: {weighted_sum:.4f}")
        
        for i, weight in enumerate(weights):
            probability_samples[i].normalized_weight = weight
            
        result = ProbabilityResult(
            probability=weighted_sum,
            samples=probability_samples,
            strategy_ids=strategy_ids
        )
        
        print(f"\n{COLOR_SUCCESS}Market movement probability: {COLOR_VALUE}{weighted_sum:.4f}{COLOR_RESET}")
        
        if weighted_sum > 0.5:
            trend = "UP 📈"
            trend_color = COLOR_SUCCESS
        else:
            trend = "DOWN 📉" 
            trend_color = COLOR_ERROR
        print(f"{COLOR_INFO}Prediction: {trend_color}{trend}{COLOR_RESET} (confidence: {COLOR_VALUE}{abs(weighted_sum-0.5)*2:.2f}{COLOR_RESET})")
        
        return result
    
    def _calculate_probability_expert(self, intent_desc: str, emotion_desc: str) -> ProbabilityResult:
        """Use multi-expert perspectives to calculate market action probability"""
        print(f"\n{COLOR_PHASE}STEP 1: CONSULTING MARKET EXPERTS{COLOR_RESET}")
        logger.info(f"Gathering predictions from {self.num_probs_to_generate} expert perspectives")
        
        expert_roles = get_random_perspectives(self.num_probs_to_generate)
        
        valid_samples = []
        expert_infos = []
        print(f"{COLOR_INFO}Consulting {self.num_probs_to_generate} market experts...{COLOR_RESET}")
        
        for i, role in enumerate(expert_roles, 1):
            print(f"{COLOR_EXPERT}• Expert {i}/{self.num_probs_to_generate} analyzing...{COLOR_RESET}")
            sample, reasoning = self._generate_expert_probability(intent_desc, emotion_desc, role)
            if sample:
                valid_samples.append(sample)
                expert_infos.append(ExpertInfo(role, reasoning, sample))
        
        logger.info(f"Successfully obtained {len(valid_samples)}/{self.num_probs_to_generate} expert predictions")
        print(f"{COLOR_SUCCESS}✓ Analysis complete: {len(valid_samples)} expert opinions considered{COLOR_RESET}")
        
        if not valid_samples:
            logger.warning("No valid expert predictions obtained, returning default value 0.5")
            print(f"{COLOR_WARNING}Unable to gather expert opinions, using default value.{COLOR_RESET}")
            return ProbabilityResult(
                probability=0.5,
                samples=[ProbabilitySample(value=0.5, log_confidence=-100.0, normalized_weight=1.0)],
                strategy_ids=[]
            )
        
        log_confidences = [sample.log_confidence for sample in valid_samples]
        probabilities = [sample.value for sample in valid_samples]
        
        logger.debug(f"Expert probability values: {probabilities}")
        logger.debug(f"Expert log-confidence values: {log_confidences}")
        
        weights = self._softmax(log_confidences)
        
        logger.debug(f"Softmax weights: {[f'{w:.4f}' for w in weights]}")
        
        weighted_contributions = []
        for i, (prob, weight) in enumerate(zip(probabilities, weights)):
            contribution = prob * weight
            weighted_contributions.append(contribution)
            logger.debug(f"Expert {i+1}: Probability={prob:.4f} × Weight={weight:.4f} = Contribution {contribution:.4f}")
        
        weighted_sum = sum(weighted_contributions)
        logger.info(f"Expert consensus weighted probability: {weighted_sum:.4f}")
        
        for i, weight in enumerate(weights):
            valid_samples[i].normalized_weight = weight
            if i < len(expert_infos):
                expert_infos[i].probability_sample.normalized_weight = weight
        
        # 🆕 构建专家详细信息列表
        expert_details_list = [
            {
                'role': expert_info.role,
                'reasoning': expert_info.reasoning,
                'probability': expert_info.probability_sample.value,
                'log_confidence': expert_info.probability_sample.log_confidence,
                'normalized_weight': expert_info.probability_sample.normalized_weight
            }
            for expert_info in expert_infos
        ]
        
        result = ProbabilityResult(
            probability=weighted_sum,
            samples=valid_samples,
            strategy_ids=[],
            expert_details=expert_details_list
        )
        
        print(f"\n{COLOR_TITLE}┌───────────────────────────────────────────────────{COLOR_RESET}")
        print(f"{COLOR_TITLE}│ {COLOR_SUCCESS}CONSENSUS FORECAST{COLOR_RESET}")
        print(f"{COLOR_TITLE}└───────────────────────────────────────────────────{COLOR_RESET}")
        print(f"{COLOR_INFO}Expert consensus probability: {COLOR_VALUE}{weighted_sum:.4f}{COLOR_RESET}")
        
        if weighted_sum > 0.5:
            trend = "UP 📈"
            trend_color = COLOR_SUCCESS
        else:
            trend = "DOWN 📉" 
            trend_color = COLOR_ERROR
        print(f"{COLOR_INFO}Market direction: {trend_color}{trend}{COLOR_RESET} (confidence: {COLOR_VALUE}{abs(weighted_sum-0.5)*2:.2f}{COLOR_RESET})")
        
        return result
    
    @staticmethod
    def _softmax(x: List[float]) -> List[float]:
        """Calculate softmax function"""
        x_array = np.array(x)
        x_shifted = x_array - np.max(x_array)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)

    def _parse_probability_text(self, text: str) -> List[float]:
        """Parse probability value list from text"""
        if not text:
            return []
            
        match = re.search(r"<ProbabilityValues>\s*(.*?)\s*</ProbabilityValues>", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            content = content.strip('[]')
            values = []
            for item in re.split(r'[,\s\n]+', content):
                item = item.strip()
                if item:
                    try:
                        value = float(item)
                        if 0 <= value <= 1:
                            values.append(value)
                    except ValueError:
                        pass
            return values
        
        values = []
        for match in re.finditer(r"0?\.\d+|0(?!\d)|1(?!\d)", text):
            try:
                value = float(match.group())
                if 0 <= value <= 1:
                    values.append(value)
            except ValueError:
                pass
        
        return values

    def _find_probability_tokens(self, completion_text: str, probability_value: float, logprobs_content) -> List[dict]:
        """
        Precisely find tokens and their logprobs corresponding to a probability value
        
        Args:
            completion_text: Complete response text
            probability_value: Probability value to find
            logprobs_content: List containing token and logprob information
            
        Returns:
            List of token information dictionaries matching the probability value
        """
        prob_str = str(probability_value)
        
        positions = []
        start_idx = 0
        while True:
            pos = completion_text.find(prob_str, start_idx)
            if pos == -1:
                break
            positions.append((pos, pos + len(prob_str)))
            start_idx = pos + 1
        
        if not positions and '.' in prob_str:
            base, decimal = prob_str.split('.')
            if decimal.endswith('0'):
                alt_prob = f"{base}.{decimal.rstrip('0')}"
                return self._find_probability_tokens(completion_text, float(alt_prob), logprobs_content)
            else:
                for i in range(1, 4):
                    alt_prob = f"{base}.{decimal}{'0' * i}"
                    alt_positions = []
                    start_idx = 0
                    while True:
                        pos = completion_text.find(alt_prob, start_idx)
                        if pos == -1:
                            break
                        alt_positions.append((pos, pos + len(alt_prob)))
                        start_idx = pos + 1
                    if alt_positions:
                        positions = alt_positions
                        prob_str = alt_prob
                        break
        
        if not positions:
            logger.warning(f"Unable to find probability value {probability_value} in text")
            return []
        
        relevant_tokens = []
        for start_pos, end_pos in positions:
            tokens_with_positions = []
            pos = 0
            for token_info in logprobs_content:
                token_text = token_info.token
                token_pos = completion_text.find(token_text, pos)
                if token_pos != -1:
                    tokens_with_positions.append({
                        'token': token_text,
                        'logprob': token_info.logprob,
                        'start': token_pos,
                        'end': token_pos + len(token_text)
                    })
                    pos = token_pos + 1

            for token_info in tokens_with_positions:
                token_start, token_end = token_info['start'], token_info['end']
                if not (token_end <= start_pos or token_start >= end_pos):
                    relevant_tokens.append(token_info)

        unique_tokens = {}
        for token_info in relevant_tokens:
            pos_key = f"{token_info['start']}:{token_info['end']}"
            if pos_key not in unique_tokens:
                unique_tokens[pos_key] = token_info
        
        return list(unique_tokens.values())

    def _generate_expert_probability_samples(self, intent_desc: str, emotion_desc: str) -> Tuple[List[ProbabilitySample], List[ExpertInfo]]:
        """Get probability samples from multiple expert perspectives"""
        expert_roles = get_random_perspectives(self.num_probs_to_generate)

        valid_samples = []
        expert_infos = []
        print(f"{COLOR_INFO}Consulting {self.num_probs_to_generate} market experts...{COLOR_RESET}")
        for i, role in enumerate(expert_roles, 1):
            sample, reasoning = self._generate_expert_probability(intent_desc, emotion_desc, role)
            if sample:
                valid_samples.append(sample)
                expert_infos.append(ExpertInfo(role, reasoning, sample))
        
        logger.info(f"Successfully obtained {len(valid_samples)}/{self.num_probs_to_generate} expert predictions")
        print(f"{COLOR_SUCCESS}✓ Analysis complete: {len(valid_samples)} expert opinions gathered{COLOR_RESET}")
        return valid_samples, expert_infos

    @rate_limit_api_call
    def _generate_expert_probability(self, intent_desc: str, emotion_desc: str, expert_role: str) -> Tuple[Optional[ProbabilitySample], str]:
        """Generate a single expert's probability prediction and reasoning"""
        logger.info(f"Consulting expert: {expert_role[:50]}...")

        template = self._load_prompt_template()

        template = template.replace('[EXPERT_ROLE_DESCRIPTION]', expert_role)
        template = template.replace('[DESCRIPTION OF THE INFERRED MARKET INTENTION]', intent_desc)
        template = template.replace('[DESCRIPTION OF THE INFERRED MARKET EMOTION]', emotion_desc)
        
        user_prompt = "According to the system prompt word output the probability of market rise."
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": template},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.llm_temperature,
                max_tokens=500,
                response_format={"type": "json_object"},
                logprobs=True,
                top_logprobs=1
            )
            
            completion_text = response.choices[0].message.content.strip()
            logprobs_content = response.choices[0].logprobs.content

            try:
                json_data = json.loads(completion_text)
                expert_response = ExpertProbabilityResponse.model_validate(json_data)
                
                # Calculate log confidence
                probability_tokens = self._find_probability_tokens(
                    completion_text, 
                    expert_response.probability, 
                    logprobs_content
                )
                
                if probability_tokens:
                    log_confidence_sum = sum(token['logprob'] for token in probability_tokens)
                else:
                    log_confidence_sum = -100.0
                    logger.warning("Unable to find tokens corresponding to probability value")

                sample = ProbabilitySample(
                    value=expert_response.probability,
                    log_confidence=log_confidence_sum,
                    normalized_weight=0.0
                )

                return sample, expert_response.reasoning
                
            except Exception as e:
                logger.warning(f"JSON parsing failed: {str(e)}")
                print(f"{COLOR_WARNING}Failed to parse expert response: {str(e)}{COLOR_RESET}")
                return None, ""
                
        except Exception as e:
            logger.error(f"Error generating expert probability: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"{COLOR_ERROR}Error consulting expert: {str(e)}{COLOR_RESET}")
            return None, ""
    
    @rate_limit_api_call
    def _generate_probability_samples(self, intent_desc: str, emotion_desc: str, strategies_text: str) -> List[ProbabilitySample]:
        """Generate multiple probability samples with confidence values"""
        print(f"{COLOR_INFO}Generating {self.num_probs_to_generate} probability estimates...{COLOR_RESET}")
                
        template = self._load_prompt_template()
        template = template.replace('{num_probabilities}', str(self.num_probs_to_generate))
        template = template.replace('[DESCRIPTION OF THE INFERRED MARKET INTENTION]', intent_desc)
        template = template.replace('[DESCRIPTION OF THE INFERRED MARKET EMOTION]', emotion_desc)
        
        user_prompt = f"Please provide {self.num_probs_to_generate} independent market upward probability estimates. Return as JSON."
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": template},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.llm_temperature,
                max_tokens=300,
                response_format={"type": "json_object"},
                logprobs=True,
                top_logprobs=1
            )
            
            completion_text = response.choices[0].message.content.strip()
            logprobs_content = response.choices[0].logprobs.content
            
            logger.debug(f"LLM response text: {completion_text[:100]}...")
            logger.debug(f"Received logprobs for {len(logprobs_content)} tokens")

            try:
                json_data = json.loads(completion_text)
                probability_response = ProbabilityResponse.model_validate(json_data)
                probability_values = probability_response.probabilities

                if len(probability_values) < self.num_probs_to_generate:
                    logger.warning(f"Not enough probability values in JSON ({len(probability_values)}/{self.num_probs_to_generate}), trying text parsing")
                    backup_values = self._parse_probability_text(completion_text)
                    if len(backup_values) > len(probability_values):
                        probability_values = backup_values[:self.num_probs_to_generate]
            except Exception as e:
                logger.warning(f"JSON parsing failed: {str(e)}, trying text parsing")
                probability_values = self._parse_probability_text(completion_text)

            if probability_values and len(probability_values) > 0:
                if len(probability_values) > self.num_probs_to_generate:
                    probability_values = probability_values[:self.num_probs_to_generate]
                
                logger.info(f"Parsed probability values: {probability_values}")

                valid_samples = []
                for value in probability_values:
                    probability_tokens = self._find_probability_tokens(
                        completion_text, 
                        value, 
                        logprobs_content
                    )

                    if probability_tokens:
                        log_confidence_sum = sum(token['logprob'] for token in probability_tokens)
                        logger.debug(f"Probability {value}: found {len(probability_tokens)} relevant tokens, log confidence: {log_confidence_sum:.4f}")
                    else:
                        log_confidence_sum = -100.0
                        logger.warning(f"Unable to find tokens for probability {value}")

                    valid_samples.append(ProbabilitySample(
                        value=value,
                        log_confidence=log_confidence_sum,
                        normalized_weight=0.0
                    ))
                
                print(f"{COLOR_SUCCESS}✓ Generated {len(valid_samples)} probability estimates{COLOR_RESET}")
                return valid_samples
            else:
                logger.error("Unable to parse valid probability values")
                return []
        except Exception as e:
            logger.error(f"Error generating probability samples: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def _find_probability_positions(self, text: str, prob_str: str) -> List[Tuple[int, int]]:
        """Find all positions of a probability value in text"""
        positions = []
        current_pos = 0

        pattern = r'\b' + re.escape(prob_str) + r'\b'
        for match in re.finditer(pattern, text):
            start, end = match.span()
            positions.append((start, end))

        if not positions and '.' in prob_str:
            base, decimal = prob_str.split('.')
            # Handle trailing zeros
            if decimal.endswith('0'):
                alt_prob = f"{base}.{decimal.rstrip('0')}"
                alt_pattern = r'\b' + re.escape(alt_prob) + r'\b'
                for match in re.finditer(alt_pattern, text):
                    start, end = match.span()
                    positions.append((start, end))
            else:
                alt_prob = f"{base}.{decimal}0"
                alt_pattern = r'\b' + re.escape(alt_prob) + r'\b'
                for match in re.finditer(alt_pattern, text):
                    start, end = match.span()
                    positions.append((start, end))
        
        return positions
