#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FB股票数据脱敏规则
=================
定义了Facebook/Meta(FB)相关的公司名称、产品、CEO、高管等的替换规则
"""

# Facebook/Meta(FB)相关规则
FB_COMPANIES = [
    "Facebook", "Facebook Inc", "Facebook Inc.", "FB", "Meta", "Meta Platforms",
    "Meta Platforms Inc", "Meta Platforms Inc.", "Instagram", "WhatsApp", 
    "Oculus", "Oculus VR", "ByteDance", "Snap", "Snapchat", "Snap Inc", "Twitter",
    "X Corp", "LinkedIn", "Pinterest", "Reddit"
]

FB_PRODUCTS = [
    # 社交平台
    {"names": ["Facebook", "FB", "Meta", "Instagram", "IG", "WhatsApp", "Messenger", 
              "Facebook Messenger"], 
     "category": "PRODUCT_SOCIAL"},
    
    # VR/AR产品
    {"names": ["Oculus", "Oculus Rift", "Oculus Quest", "Quest", "Quest [0-9]", "Quest Pro", 
              "Meta Quest", "Meta Quest [0-9]", "Meta Quest Pro", "Ray-Ban Stories",
              "Horizon Worlds", "Horizon Workrooms"], 
     "category": "PRODUCT_VR"},
    
    # 其他服务
    {"names": ["Facebook Watch", "Facebook Gaming", "Facebook Marketplace", "Facebook Pay",
              "Facebook Portal", "Portal", "Portal\\+", "Portal TV", "Portal Go", "Portal Mini"], 
     "category": "PRODUCT_SERVICE"},
    
    # 技术平台
    {"names": ["React", "ReactJS", "React Native", "PyTorch", "FAIR", 
              "Facebook AI Research", "Meta AI"], 
     "category": "PRODUCT_TECH"}
]

FB_CEOS = ["Mark Zuckerberg", "mark zuckerberg", "Zuckerberg", "zuckerberg"]
FB_EXECS = ["Sheryl Sandberg", "sheryl sandberg", "David Wehner", "david wehner", 
            "Mike Schroepfer", "mike schroepfer", "Chris Cox", "chris cox", 
            "Javier Olivan", "javier olivan", "Andrew Bosworth", "andrew bosworth", 
            "Marne Levine", "marne levine", "Nick Clegg", "nick clegg", 
            "Jennifer Newstead", "jennifer newstead", "Susan Li", "susan li"]

FB_PEERS = {
    "companies": ["Twitter", "X", "Snap", "Snapchat", "TikTok", "LinkedIn", "Pinterest", 
                 "YouTube", "Discord", "Telegram", "WeChat", "Weibo"],
    "products": ["Twitter", "X", "Snapchat", "TikTok", "Douyin", "LinkedIn", "Pinterest",
                "YouTube", "Discord", "Telegram", "WeChat", "Threads"]
} 