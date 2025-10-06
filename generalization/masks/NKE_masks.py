#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKE股票数据脱敏规则
===============
定义了Nike(NKE)相关的公司名称、产品、CEO、高管等的替换规则
"""

# Nike(NKE)相关规则
NKE_COMPANIES = [
    "Nike", "Nike Inc", "Nike Inc.", "Nike, Inc.", "Nike, Inc", "NIKE", 
    "NKE", "Blue Ribbon Sports", "Converse", "Converse Inc.", 
    "Jordan Brand", "Hurley International", "Nike By You", "Nike+",
    "Nike Digital", "Nike Direct"
]

NKE_PRODUCTS = [
    # 鞋类产品
    {"names": ["Air Force 1", "Air Force One", "Air Jordan", "Jordan", "Jordan [0-9]+",
              "Air Max", "Air Max [0-9]+", "Air Max [0-9]+X", "VaporMax", "Nike Air",
              "Dunk", "Nike Dunk", "SB Dunk", "React", "Nike React", "Epic React",
              "Infinity React", "Flyknit", "Nike Flyknit", "Pegasus", "Nike Pegasus", 
              "Zoom Pegasus", "ZoomX", "Zoom", "Nike Zoom", "AlphaFly", "Vaporfly",
              "Metcon", "Free Run", "Nike Free", "Huarache", "Nike Blazer",
              "Cortez", "Nike Cortez", "Waffle", "Nike Waffle"], 
     "category": "PRODUCT_FOOTWEAR"},
    
    # 服装产品
    {"names": ["Nike Pro", "Dri-FIT", "Tech Fleece", "Nike Tech Fleece", "Therma-FIT",
              "Storm-FIT", "Nike ACG", "ACG", "Nike Sportswear", "Nike SB", 
              "Nike Basketball", "Nike Training", "Nike Running", "FIT ADV"], 
     "category": "PRODUCT_APPAREL"},
    
    # 装备产品
    {"names": ["Nike Elite", "Swoosh", "Nike Swoosh", "FuelBand", "Nike FuelBand"], 
     "category": "PRODUCT_EQUIPMENT"},
    
    # 数字产品和应用
    {"names": ["SNKRS", "Nike SNKRS", "Nike App", "Nike Training Club", "NTC",
              "Nike Run Club", "NRC", "Nike Fit", "Nike Adapt"], 
     "category": "PRODUCT_DIGITAL"},
    
    # 合作系列
    {"names": ["Nike x Off-White", "Nike x Supreme", "Nike x Sacai", "Nike x Fragment",
              "Nike x Travis Scott", "Nike x Comme des Garcons", "Nike x Undercover", 
              "Nike x Virgil Abloh", "Nike SB x", "Nike ACG x"], 
     "category": "PRODUCT_COLLABORATION"}
]

NKE_CEOS = ["John Donahoe", "john donahoe", "Mark Parker", "mark parker", 
           "William Perez", "william perez", "Phil Knight", "phil knight", 
           "Philip Knight", "philip knight"]
NKE_EXECS = ["Matthew Friend", "matthew friend", "Elliott Hill", "elliott hill",
            "Heidi O'Neill", "heidi o'neill", "Andy Campion", "andy campion", 
            "Hilary Krane", "hilary krane", "Monique Matheson", "monique matheson",
            "Ann Miller", "ann miller", "Tom Clarke", "tom clarke", 
            "Michael Spillane", "michael spillane"]

NKE_PEERS = {
    "companies": ["Adidas", "Under Armour", "Puma", "Reebok", "New Balance", "Asics",
                 "Skechers", "Lululemon", "Fila", "On Running", "Brooks", "Hoka",
                 "Vans", "Columbia", "The North Face", "Patagonia"],
    "products": ["Ultraboost", "NMD", "Yeezy", "Superstar", "Stan Smith", 
                "Curry", "UA HOVR", "RS-X", "Clyde", "Suede", "990", "Fresh Foam", 
                "Gel-Kayano", "Go Run", "Alphafly"]
} 