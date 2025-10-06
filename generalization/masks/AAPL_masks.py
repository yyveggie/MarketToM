#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AAPL股票数据脱敏规则
==================
定义了Apple/AAPL相关的公司名称、产品、CEO、高管等的替换规则
"""

# 苹果(AAPL)相关规则
AAPL_COMPANIES = [
    "Apple", "Apple Inc", "Apple Inc.", "AAPL", 
    "Beats Electronics", "Beats", "Claris", "Claris International",
    "FileMaker", "Apple Energy", "Foxconn", "Pegatron", "Wistron",
    "TSMC", "Taiwan Semiconductor", "Corning", "Jabil", "Jabil Circuit",
    "Broadcom", "Qualcomm", "Cirrus Logic", "Skyworks", "Skyworks Solutions",
    "Qorvo", "Lam Research", "Applied Materials", "Honeywell"
]

AAPL_PRODUCTS = [
    # 手机和平板
    {"names": ["iPhone", "iPhones", "iPhone\\s*\\d+", "iPhone\\s*\\d+[sS]", "iPhone\\s*\\d+\\s*Pro", 
              "iPhone\\s*\\d+\\s*Pro\\s*Max", "iPhone\\s*\\d+\\s*mini", "iPhone\\s*SE"], 
     "category": "PRODUCT_PHONE"},
    {"names": ["iPad", "iPads", "iPad\\s*Pro", "iPad\\s*Air", "iPad\\s*mini", "iPad\\s*\\d"], 
     "category": "PRODUCT_TABLET"},
    
    # 电脑
    {"names": ["Mac", "Macs", "MacBook", "MacBook\\s*Pro", "MacBook\\s*Air", "iMac", "Mac\\s*mini", 
              "Mac\\s*Pro", "Mac\\s*Studio"], 
     "category": "PRODUCT_COMPUTER"},
    
    # 手表和配件
    {"names": ["Apple Watch", "Apple\\s*Watch", "Watch\\s*Series\\s*\\d", "Apple Watch\\s*Series\\s*\\d", 
              "Watch\\s*Ultra", "Apple\\s*Watch\\s*Ultra"], 
     "category": "PRODUCT_WATCH"},
    {"names": ["AirPods", "AirPods\\s*Pro", "AirPods\\s*Max", "Beats", "Beats\\s*Studio", 
              "Beats\\s*Solo", "Beats\\s*Powerbeats"], 
     "category": "PRODUCT_HEADPHONE"},
    
    # 服务和软件
    {"names": ["iOS", "iOS\\s*\\d+", "iPadOS", "macOS", "watchOS", "tvOS", "macOS\\s*[A-Za-z\\s]+", 
              "Mac OS X", "OS X"], 
     "category": "PRODUCT_OS"},
    {"names": ["Safari", "Apple Music", "iTunes", "App Store", "iCloud", 
              "Apple TV\\+", "Apple Arcade", "Apple Pay", "Apple Card", "Apple News\\+", 
              "Apple Fitness\\+", "Apple One"], 
     "category": "PRODUCT_SERVICE"},
    
    # 其他设备
    {"names": ["Apple TV", "HomePod", "HomePod mini"], 
     "category": "PRODUCT_HOME"},
    {"names": ["Final Cut Pro", "Logic Pro", "GarageBand", "iWork", "Pages", 
              "Numbers", "Keynote"], 
     "category": "PRODUCT_SOFTWARE"}
]

AAPL_CEOS = ["Tim Cook", "Timothy Cook", "tim cook", "timothy cook", "Steve Jobs", "Steven Jobs", "steve jobs", "steven jobs", "John Sculley", "Gil Amelio", "Mike Markkula", "john sculley", "gil amelio", "mike markkula"]
AAPL_EXECS = ["Craig Federighi", "craig federighi", "Eddy Cue", "eddy cue", "Jeff Williams", "jeff williams", "Katherine Adams", "katherine adams", "Luca Maestri", "luca maestri", 
              "Phil Schiller", "phil schiller", "Johny Srouji", "johny srouji", "John Giannandrea", "john giannandrea", "Deirdre O'Brien", "deirdre o'brien", "John Ternus", "john ternus",
              "Scott Forstall", "scott forstall", "Jony Ive", "jony ive", "Jonathan Ive", "jonathan ive", "Tony Fadell", "tony fadell", "Ron Johnson", "ron johnson"]

AAPL_PEERS = {
    "companies": ["Samsung", "Xiaomi", "Huawei", "Google", "Microsoft", "Sony", "Dell", "HP", 
                 "Lenovo", "OPPO", "Vivo", "OnePlus", "LG", "HTC"],
    "products": ["Galaxy", "Galaxy S[0-9]+", "Galaxy Note", "Pixel", "Surface", "Windows", 
                "Android", "Xperia", "MatePad", "MateBook", "ThinkPad", "XPS"]
} 