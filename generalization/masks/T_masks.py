#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T股票数据脱敏规则
===============
定义了AT&T(T)相关的公司名称、产品、CEO、高管等的替换规则
"""

# AT&T(T)相关规则
T_COMPANIES = [
    "AT&T", "AT and T", "AT & T", "American Telephone and Telegraph Company", 
    "T", "AT&T Inc", "AT&T Inc.", "WarnerMedia", "DirecTV", "Cricket Wireless", 
    "AT&T Communications", "Verizon", "Verizon Communications", "T-Mobile", 
    "T-Mobile US", "Sprint", "Comcast", "Charter Communications", "Dish Network", 
    "DISH", "Vodafone", "Liberty Broadband", "LBRDA", "Warner Bros. Discovery", "WBD"
]

T_PRODUCTS = [
    # 电信服务
    {"names": ["AT&T Wireless", "AT&T Internet", "AT&T Fiber", "AT&T TV", "AT&T Phone",
              "DIRECTV", "DirecTV", "DIRECTV STREAM", "DirecTV Stream", "Cricket Wireless",
              "FirstNet"], 
     "category": "PRODUCT_TELECOM"},
    
    # 流媒体服务
    {"names": ["HBO Max", "HBO", "Max", "Warner Bros.", "Warner Brothers", "CNN", 
              "TNT", "TBS", "Cartoon Network", "Adult Swim", "DC", "DC Comics"], 
     "category": "PRODUCT_MEDIA"},
    
    # 网络技术
    {"names": ["5G", "5G\\+", "5G Evolution", "LTE", "4G LTE", "FirstNet", "AT&T Business"], 
     "category": "PRODUCT_NETWORK"}
]

T_CEOS = ["John Stankey", "john stankey", "Randall Stephenson", "randall stephenson", 
          "Edward Whitacre Jr.", "edward whitacre jr", "Ed Whitacre", "ed whitacre"]
T_EXECS = ["Pascal Desroches", "pascal desroches", "Jeff McElfresh", "jeff mcelfresh", 
          "Lori Lee", "lori lee", "David Huntley", "david huntley", "Jeremy Legg", "jeremy legg", 
          "Angela Santone", "angela santone", "David McAtee", "david mcatee"]

T_PEERS = {
    "companies": ["Verizon", "T-Mobile", "Sprint", "Comcast", "Charter", "Spectrum", 
                 "Dish Network", "DISH", "Altice", "Vodafone", "Rogers", "Bell"],
    "products": ["Verizon Fios", "Xfinity", "Spectrum", "Dish", "Netflix", "Disney\\+", 
                "Hulu", "Peacock", "Paramount\\+"]
} 