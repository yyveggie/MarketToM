# -*- coding: utf-8 -*-
"""
Stance Classifier: Market mental state stance classifier
Uses LLM probe to determine market stance (UP/DOWN) for each mental state
"""

import os
import json
import logging
import time
import random
from typing import Tuple
from datetime import datetime, timedelta
import openai
from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='market_tom_stance.log',
    filemode='a'
)
logger = logging.getLogger('MarketToM.StanceClassifier')

COLOR_TITLE = "\033[1;36m"
COLOR_SUCCESS = "\033[1;32m"
COLOR_WARNING = "\033[1;33m"
COLOR_ERROR = "\033[1;31m"
COLOR_INFO = "\033[0;34m"
COLOR_VALUE = "\033[1;35m"
COLOR_DEBUG = "\033[0;90m"
COLOR_RESET = "\033[0m"

last_api_request_time = datetime.now() - timedelta(seconds=10)
MIN_REQUEST_INTERVAL = 15.0
DEFAULT_COOLDOWN = 15.0
MAX_JITTER = 1.0


class StanceResponse(BaseModel):
    """LLM stance classification response model"""
    stance: str = Field(..., description="Market stance: UP or DOWN")
    confidence: float = Field(..., description="Confidence level (0-1)")
    reasoning: str = Field(..., description="Brief reasoning for the stance")
    
    def validate_stance(self):
        if self.stance.upper() not in ['UP', 'DOWN']:
            raise ValueError(f"Invalid stance: {self.stance}, must be UP or DOWN")
        self.stance = self.stance.upper()


def rate_limit_api_call(func):
    """API rate limiting decorator"""
    def wrapper(*args, **kwargs):
        global last_api_request_time
        now = datetime.now()
        time_since_last = (now - last_api_request_time).total_seconds()
        
        if time_since_last < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last + random.uniform(0, MAX_JITTER)
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        last_api_request_time = datetime.now()
        result = func(*args, **kwargs)
        
        cooldown = DEFAULT_COOLDOWN + random.uniform(0, MAX_JITTER)
        logger.debug(f"Cooling down for {cooldown:.2f}s")
        time.sleep(cooldown)
        
        return result
    return wrapper


class StanceClassifier:
    """Mental state stance classifier"""
    
    def __init__(self, llm_client: openai.OpenAI, llm_model: str, temperature: float = 0.3):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.temperature = temperature
        
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'templates',
            'stance_classification_prompt_template.xml'
        )
        
        with open(template_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        tree = ET.fromstring(xml_content)
        
        role = tree.find('Role').text.strip()
        task = tree.find('Task').text.strip()
        
        perspective = tree.find('.//Perspective').text.strip()
        belief_rule = tree.find('.//Belief').text.strip()
        intention_rule = tree.find('.//Intention').text.strip()
        emotion_rule = tree.find('.//Emotion').text.strip()
        
        output_format = tree.find('OutputFormat').text.strip()
        
        self.STANCE_PROMPT_TEMPLATE = f"""{role}

{task}

Market {{state_type}} Description:
{{description}}

ANALYSIS PERSPECTIVE:
{perspective}

RULES:
- BELIEF: {belief_rule}
- INTENTION: {intention_rule}  
- EMOTION: {emotion_rule}

{output_format}
"""
            
        logger.info(f"StanceClassifier initialized with model {llm_model}, temperature {temperature}")
        logger.info(f"Loaded prompt template from {template_path}")

    @rate_limit_api_call
    def classify_stance(self, description: str, state_type: str) -> Tuple[str, float, str]:
        logger.info(f"Classifying stance for {state_type}")
        
        prompt = self.STANCE_PROMPT_TEMPLATE.format(
            state_type=state_type.upper(),
            description=description
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Classify the market stance."}
                    ],
                    temperature=self.temperature,
                    max_tokens=300,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                logger.debug(f"LLM response: {content[:100]}...")
                
                data = json.loads(content)
                stance_obj = StanceResponse.model_validate(data)
                stance_obj.validate_stance()
                
                logger.info(f"Stance classified: {stance_obj.stance} (confidence: {stance_obj.confidence:.2f})")
                return stance_obj.stance, stance_obj.confidence, stance_obj.reasoning
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to classify stance after {max_retries} attempts")
                    return "UNKNOWN", 0.5, "Classification failed"
                time.sleep(2 ** attempt)
        
        return "UNKNOWN", 0.5, "Classification failed"
    
    def classify_all_states(self, mental_states: dict, verbose: bool = False) -> dict:
        if verbose:
            print(f"\n{COLOR_TITLE}┌─ MENTAL STATE STANCE CLASSIFICATION ─────────────────┐{COLOR_RESET}")
        
        stances = {}
        
        state_icons = {
            'belief': '🧠',
            'intent': '🎯',
            'emotion': '💭'
        }
        
        for idx, state_type in enumerate(['belief', 'intent', 'emotion'], 1):
            if state_type not in mental_states:
                logger.warning(f"Missing {state_type} in mental_states")
                continue
            
            description = mental_states[state_type]
            
            if verbose:
                print(f"\n{COLOR_INFO}{state_icons[state_type]} [{idx}/3] Analyzing {state_type.upper()}:{COLOR_RESET}")
                print(f"{COLOR_DEBUG}├─ Description:{COLOR_RESET}")
                desc_lines = description.split('\n')
                for line in desc_lines[:3]:
                    print(f"{COLOR_DEBUG}│  {line[:70]}{COLOR_RESET}")
                if len(desc_lines) > 3:
                    print(f"{COLOR_DEBUG}│  ... ({len(desc_lines)-3} more lines){COLOR_RESET}")
                print(f"{COLOR_DEBUG}├─ Probing LLM...{COLOR_RESET}")
            else:
                print(f"{COLOR_INFO}📊 Analyzing {state_type.upper()} stance...{COLOR_RESET}")
            
            stance, confidence, reasoning = self.classify_stance(description, state_type)
            
            stances[state_type] = {
                "stance": stance,
                "confidence": confidence,
                "reasoning": reasoning,
                "description": description
            }
            
            if verbose:
                stance_color = COLOR_SUCCESS if stance == "UP" else COLOR_ERROR
                stance_arrow = "↗" if stance == "UP" else "↘"
                confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
                
                print(f"{COLOR_DEBUG}└─ LLM Response:{COLOR_RESET}")
                print(f"   {COLOR_VALUE}Stance:{COLOR_RESET} {stance_color}{stance} {stance_arrow}{COLOR_RESET}")
                print(f"   {COLOR_VALUE}Confidence:{COLOR_RESET} {confidence:.2f} [{confidence_bar}]")
                print(f"   {COLOR_VALUE}Reasoning:{COLOR_RESET} {COLOR_DEBUG}{reasoning}{COLOR_RESET}")
            else:
                stance_color = COLOR_SUCCESS if stance == "UP" else COLOR_ERROR
                print(f"{COLOR_VALUE}  {state_type.upper()}: {stance_color}{stance}{COLOR_RESET} (confidence: {confidence:.2f})")
        
        if verbose:
            print(f"\n{COLOR_TITLE}└────────────────────────────────────────────────────────┘{COLOR_RESET}")
        
        logger.info(f"All stances classified: {[(k, v['stance']) for k, v in stances.items()]}")
        return stances


def add_stances_to_inference_log(log_filepath: str, stances: dict) -> None:
    try:
        with open(log_filepath, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        log_data['mental_state_stances'] = stances
        log_data['stance_classification_timestamp'] = datetime.now().isoformat()
        
        with open(log_filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Stances added to log: {log_filepath}")
        
    except Exception as e:
        logger.error(f"Failed to add stances to log: {str(e)}")
        raise
