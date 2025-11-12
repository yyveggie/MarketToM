# -*- coding: utf-8 -*-
"""
Integrated Stance Classification
Automatically add stance classification after each inference
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.stance_classifier import StanceClassifier, add_stances_to_inference_log
import logging

logger = logging.getLogger('MarketToM.Integration')


class IntegratedStanceClassifier:
    """Integrated stance classifier for run.py"""
    
    def __init__(self, llm_client, llm_model: str, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self.classifier = StanceClassifier(llm_client, llm_model, temperature=0.3)
            logger.info("Integrated stance classifier enabled")
        else:
            self.classifier = None
            logger.info("Integrated stance classifier disabled")
    
    def classify_and_save(self, mental_states: dict, log_filepath: str) -> dict:
        if not self.enabled or self.classifier is None:
            logger.debug("Stance classification skipped (disabled)")
            return {}
        
        try:
            stances = self.classifier.classify_all_states(mental_states)
            add_stances_to_inference_log(log_filepath, stances)
            
            logger.info(f"Stances classified and saved to {log_filepath}")
            return stances
            
        except Exception as e:
            logger.error(f"Failed to classify stances: {str(e)}")
            return {}

