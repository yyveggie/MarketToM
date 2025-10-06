#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用文本脱敏规则
=============
定义了适用于所有股票的通用替换规则，如日期、年份等
"""

# 日期和年份替换模式
DATE_PATTERNS = [
    # 月日年格式
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}",
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}",
    r"\d{1,2}/\d{1,2}/\d{4}",  # MM/DD/YYYY or DD/MM/YYYY
    
    # 季度表达式
    r"Q[1-4]\s+\d{4}",  # Q1 2023
    r"[1-4]Q\s+\d{4}",  # 1Q 2023
    r"[1-4]Q\d{2}",     # 1Q23
    
    # 财年表达式
    r"FY\s*\d{4}",      # FY 2023, FY2023
    r"FY\s*\d{2}",      # FY 23, FY23
]

YEAR_PATTERN = r"\b(19|20)\d{2}\b"  # 1900-2099年 