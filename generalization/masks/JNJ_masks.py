#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JNJ股票数据脱敏规则
===============
定义了Johnson & Johnson(JNJ)相关的公司名称、产品、CEO、高管等的替换规则
"""

# Johnson & Johnson(JNJ)相关规则
JNJ_COMPANIES = [
    "Johnson & Johnson", "Johnson and Johnson", "J&J", "JNJ", 
    "Janssen", "Janssen Pharmaceuticals", "Janssen Biotech", 
    "Ethicon", "Actelion", "McNeil Consumer Healthcare",
    "DePuy Synthes", "Biosense Webster", "Mentor Worldwide"
]

JNJ_PRODUCTS = [
    # 消费者健康产品
    {"names": ["Tylenol", "Motrin", "Benadryl", "Zyrtec", "Sudafed", "Pepcid",
              "Band-Aid", "Listerine", "Neutrogena", "Aveeno", "Johnson's",
              "Johnson's Baby", "Clean & Clear", "Nicorette", "OGX", "Lubriderm",
              "Rogaine", "Neosporin", "Desitin"], 
     "category": "PRODUCT_CONSUMER_HEALTH"},
    
    # 医疗设备
    {"names": ["DePuy Synthes", "Ethicon", "Biosense Webster", "Mentor", "Acclarent",
              "Cerenovus", "ACUVUE", "ACUVUE OASYS", "ACUVUE VITA", "ACUVUE DEFINE",
              "OneTouch", "OneTouch Verio", "Thermoscan", "Stapler", "HARMONIC",
              "ECHELON", "ENSEAL"], 
     "category": "PRODUCT_MEDICAL_DEVICE"},
    
    # 药品
    {"names": ["Remicade", "Stelara", "Tremfya", "Simponi", "Zytiga", "Darzalex",
              "Imbruvica", "Xarelto", "Invega", "Invega Sustenna", "Risperdal",
              "Concerta", "Prezista", "Invokana", "Erleada", "Doxil", "Spravato",
              "Ponvory", "Uptravi", "Opsumit", "Evarrest", "Vermox"], 
     "category": "PRODUCT_PHARMACEUTICAL"},
    
    # 疫苗
    {"names": ["COVID-19 Vaccine", "Janssen COVID-19 Vaccine", "J&J Vaccine", 
              "J&J COVID Vaccine", "Johnson & Johnson Vaccine", "Ebola Vaccine"], 
     "category": "PRODUCT_VACCINE"}
]

JNJ_CEOS = ["Joaquin Duato", "joaquin duato", "Alex Gorsky", "alex gorsky", 
           "William Weldon", "william weldon", "Ralph Larsen", "ralph larsen"]
JNJ_EXECS = ["Joseph Wolk", "joseph wolk", "Vanessa Broadhurst", "vanessa broadhurst", 
            "Peter Fasolo", "peter fasolo", "Ashley McEvoy", "ashley mcevoy",
            "Thibaut Mongon", "thibaut mongon", "Jennifer Taubert", "jennifer taubert", 
            "Kathryn Wengel", "kathryn wengel", "Michael Ullmann", "michael ullmann"]

JNJ_PEERS = {
    "companies": ["Pfizer", "Merck", "Novartis", "Roche", "AbbVie", "Bristol Myers Squibb", 
                 "Eli Lilly", "Abbott", "AstraZeneca", "Amgen", "Gilead", "GlaxoSmithKline", "GSK",
                 "Bayer", "Biogen", "Regeneron", "Moderna", "Sanofi", "Takeda",
                 "Procter & Gamble", "P&G", "Unilever", "Colgate-Palmolive"],
    "products": ["Advil", "Robitussin", "Centrum", "Crest", "Pampers", "Huggies",
                "Sensodyne", "Dove", "Nivea", "Pantene", "Coppertone", "L'Oreal",
                "Humira", "Keytruda", "Eliquis", "Vyndamax", "Skyrizi", "Dupixent",
                "Ozempic", "Trulicity", "Botox"]
} 