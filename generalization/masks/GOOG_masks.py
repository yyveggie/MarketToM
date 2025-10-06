#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOOG股票数据脱敏规则
=================
定义了Google/Alphabet(GOOG)相关的公司名称、产品、CEO、高管等的替换规则
"""

# Google/Alphabet(GOOG)相关规则
GOOG_COMPANIES = [
    "Google", "Google Inc", "Google Inc.", "Google LLC", "Alphabet", "Alphabet Inc",
    "Alphabet Inc.", "GOOG", "GOOGL", "YouTube", "YouTube LLC", "Waymo", "Waymo LLC",
    "Verily", "Verily Life Sciences", "Calico", "Calico LLC", "DeepMind", 
    "DeepMind Technologies", "Google Fiber", "Nest", "Nest Labs", "Baidu", 
    "Microsoft", "Yahoo", "DuckDuckGo"
]

GOOG_PRODUCTS = [
    # 搜索和广告
    {"names": ["Google Search", "Google", "Google Ads", "AdWords", "AdSense", "Google Analytics"], 
     "category": "PRODUCT_SEARCH"},
    
    # 操作系统和设备
    {"names": ["Android", "Android\\s*\\d+", "Chrome OS", "ChromeOS"], 
     "category": "PRODUCT_OS"},
    {"names": ["Pixel", "Google Pixel", "Pixel\\s*\\d", "Pixel\\s*\\d\\s*Pro", "Pixel\\s*\\d+a",
              "Pixel Fold", "Pixel Tablet", "Chromebook", "Pixelbook"], 
     "category": "PRODUCT_DEVICE"},
    
    # 应用和服务
    {"names": ["Chrome", "Google Chrome", "Gmail", "Google Maps", "Maps", "YouTube", "Google Drive",
              "Drive", "Google Photos", "Photos", "Google Docs", "Docs", "Google Sheets", "Sheets",
              "Google Slides", "Slides", "Google Calendar", "Calendar"], 
     "category": "PRODUCT_APP"},
    {"names": ["Google Play", "Play Store", "Google Cloud", "Google Cloud Platform", "GCP",
              "Google Workspace", "G Suite", "Google One"], 
     "category": "PRODUCT_SERVICE"},
    
    # 智能家居和AI
    {"names": ["Google Assistant", "Assistant", "Google Home", "Google Nest", "Nest",
              "Nest Hub", "Nest Mini", "Nest Audio", "Nest Cam", "Nest Doorbell",
              "Nest Thermostat", "Nest Protect", "Nest WiFi", "Nest Secure"], 
     "category": "PRODUCT_HOME"},
    {"names": ["Waymo", "Waymo One", "Waymo Driver", "Waymo Via"], 
     "category": "PRODUCT_AUTO"},
    
    # 开发者工具
    {"names": ["TensorFlow", "Flutter", "Dart", "Firebase", "Google Cloud", "Google API"], 
     "category": "PRODUCT_DEV"}
]

GOOG_CEOS = ["Sundar Pichai", "sundar pichai", "Larry Page", "larry page", 
             "Sergey Brin", "sergey brin", "Eric Schmidt", "eric schmidt"]
GOOG_EXECS = ["Ruth Porat", "ruth porat", "Prabhakar Raghavan", "prabhakar raghavan", 
             "Philipp Schindler", "philipp schindler", "Kent Walker", "kent walker",
             "Rick Osterloh", "rick osterloh", "Thomas Kurian", "thomas kurian", 
             "Susan Wojcicki", "susan wojcicki", "Neal Mohan", "neal mohan"]

GOOG_PEERS = {
    "companies": ["Microsoft", "Apple", "Meta", "Facebook", "Amazon", "Baidu", 
                 "Bing", "Yahoo", "DuckDuckGo", "TikTok", "Twitter", "X"],
    "products": ["Bing", "Edge", "Azure", "Office", "iOS", "Safari", "Siri",
                "Alexa", "AWS", "Facebook", "Instagram", "TikTok"]
} 