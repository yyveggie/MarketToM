# -*- coding: utf-8 -*-
"""
集成姿态分类到前向推理流程
在每次推理后自动添加姿态分类
"""

import os
import sys

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.stance_classifier import StanceClassifier, add_stances_to_inference_log
import logging

logger = logging.getLogger('MarketToM.Integration')


class IntegratedStanceClassifier:
    """集成的姿态分类器（用于run.py）"""
    
    def __init__(self, llm_client, llm_model: str, enabled: bool = True):
        """
        初始化集成分类器
        
        Args:
            llm_client: OpenAI客户端
            llm_model: LLM模型名称
            enabled: 是否启用姿态分类
        """
        self.enabled = enabled
        if enabled:
            self.classifier = StanceClassifier(llm_client, llm_model, temperature=0.3)
            logger.info("Integrated stance classifier enabled")
        else:
            self.classifier = None
            logger.info("Integrated stance classifier disabled")
    
    def classify_and_save(self, mental_states: dict, log_filepath: str) -> dict:
        """
        分类心智状态姿态并保存到日志
        
        Args:
            mental_states: 心智状态字典
            log_filepath: 推理日志文件路径
        
        Returns:
            姿态分类结果
        """
        if not self.enabled or self.classifier is None:
            logger.debug("Stance classification skipped (disabled)")
            return {}
        
        try:
            # 分类姿态
            stances = self.classifier.classify_all_states(mental_states)
            
            # 保存到日志
            add_stances_to_inference_log(log_filepath, stances)
            
            logger.info(f"Stances classified and saved to {log_filepath}")
            return stances
            
        except Exception as e:
            logger.error(f"Failed to classify stances: {str(e)}")
            return {}
