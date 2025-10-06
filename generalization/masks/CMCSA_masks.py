#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMCSA股票数据脱敏规则
=================
定义了Comcast(CMCSA)相关的公司名称、产品、CEO、高管等的替换规则
"""

# Comcast(CMCSA)相关规则
CMCSA_COMPANIES = [
    "Comcast", "Comcast Corporation", "CMCSA", "Comcast Cable", 
    "NBCUniversal", "NBC Universal", "NBCU", "Universal Studios",
    "Universal Pictures", "Sky Group", "Sky Limited", "Xfinity"
]

CMCSA_PRODUCTS = [
    # 视频和电视服务
    {"names": ["Xfinity TV", "Xfinity Stream", "X1", "Xfinity X1", "Xfinity Flex", 
              "Peacock", "Peacock Premium", "Peacock Plus", "Sky Q", "Now TV"], 
     "category": "PRODUCT_VIDEO"},
    
    # 互联网服务
    {"names": ["Xfinity Internet", "Xfinity WiFi", "xFi", "Xfinity xFi", "Xfinity xFi Advanced Gateway",
              "Xfinity xFi Complete", "Sky Broadband", "Sky Fiber"], 
     "category": "PRODUCT_INTERNET"},
    
    # 电话服务
    {"names": ["Xfinity Voice", "Xfinity Mobile", "Sky Talk", "Sky Mobile"], 
     "category": "PRODUCT_PHONE"},
    
    # 内容和媒体
    {"names": ["NBC", "MSNBC", "CNBC", "Telemundo", "Universal Studios", "Universal Pictures",
              "Dreamworks Animation", "Dreamworks", "Sky News", "Sky Sports", "Sky Cinema", 
              "USA Network", "Syfy", "E!", "Bravo", "Oxygen", "Golf Channel"], 
     "category": "PRODUCT_MEDIA"},
    
    # 主题公园
    {"names": ["Universal Studios Hollywood", "Universal Orlando Resort", "Universal Studios Florida",
              "Islands of Adventure", "Universal's Volcano Bay", "Universal Studios Japan", 
              "Universal Studios Singapore", "Universal Beijing Resort"], 
     "category": "PRODUCT_THEME_PARK"},
    
    # 智能家居和安全
    {"names": ["Xfinity Home", "Xfinity Home Security", "Sky Q Hub"], 
     "category": "PRODUCT_SMART_HOME"}
]

CMCSA_CEOS = ["Brian Roberts", "brian roberts", "Brian L. Roberts", "brian l. roberts", 
             "Ralph Roberts", "ralph roberts"]
CMCSA_EXECS = ["Michael Cavanagh", "michael cavanagh", "Dave Watson", "dave watson",
              "Dana Strong", "dana strong", "Jeff Shell", "jeff shell", 
              "Karen Dougherty Buchholz", "karen dougherty buchholz", "Adam Miller", "adam miller",
              "Tom Reid", "tom reid", "Mitch Rose", "mitch rose"]

CMCSA_PEERS = {
    "companies": ["AT&T", "Verizon", "Charter Communications", "Charter", "Spectrum",
                 "Disney", "Walt Disney Company", "Warner Bros. Discovery", "Netflix",
                 "Amazon", "Hulu", "Apple", "Paramount", "Paramount Global", "ViacomCBS",
                 "Sony", "CBS", "Fox", "Fox Corporation", "Dish Network", "DISH",
                 "T-Mobile", "Cox Communications", "Cox", "Altice USA"],
    "products": ["DIRECTV", "DirecTV", "Spectrum TV", "Spectrum Internet", "Fios", 
                "Verizon Fios", "Disney+", "Netflix", "Amazon Prime Video", "Hulu",
                "Apple TV+", "HBO Max", "Max", "Paramount+", "Sling TV", "YouTube TV"]
} 