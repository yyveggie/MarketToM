#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Mastercard(MA)相关规则
MA_COMPANIES = [
    "Mastercard", "Mastercard Inc", "Mastercard Inc.", "Mastercard Incorporated",
    "MA", "Master Card", "MasterCard", "MasterCard Worldwide"
]

MA_PRODUCTS = [
    # 支付卡产品
    {"names": ["Mastercard", "Master Card", "MasterCard Gold", "MasterCard Platinum",
              "MasterCard World", "MasterCard World Elite", "World Mastercard",
              "Platinum Mastercard", "Titanium Mastercard", "Black Card"], 
     "category": "PRODUCT_CARD"},
    
    # 数字支付产品
    {"names": ["Masterpass", "Mastercard Click to Pay", "Mastercard Digital Enablement Service",
              "MDES", "Mastercard Send", "Mastercard B2B Hub", "Pay by Bank app"], 
     "category": "PRODUCT_DIGITAL_PAYMENT"},
    
    # 安全产品
    {"names": ["SecureCode", "Mastercard SecureCode", "Mastercard Identity Check",
              "Identity Check", "Threat Scan", "Safety Net", "NuData Security"], 
     "category": "PRODUCT_SECURITY"},
    
    # 数据分析和咨询服务
    {"names": ["Mastercard Advisors", "Mastercard Data & Services", "Applied Predictive Technology",
              "APT", "Mastercard SpendingPulse", "Test & Learn", "Market Insights"], 
     "category": "PRODUCT_ANALYTICS"}
]

MA_CEOS = ["Michael Miebach", "michael miebach", "Ajay Banga", "ajay banga", 
           "Robert Selander", "robert selander"]
MA_EXECS = ["Sachin Mehra", "sachin mehra", "Craig Vosburg", "craig vosburg",
           "Michael Froman", "michael froman", "Tim Murphy", "tim murphy", 
           "Linda Kirkpatrick", "linda kirkpatrick", "Raj Seshadri", "raj seshadri",
           "Jorn Lambert", "jorn lambert", "Gilberto Caldart", "gilberto caldart"]

MA_PEERS = {
    "companies": ["Visa", "American Express", "AmEx", "Discover", "PayPal", "Square", 
                 "Stripe", "Adyen", "Worldpay", "Fiserv", "FIS", "Western Union", 
                 "JCB", "UnionPay", "China UnionPay"],
    "products": ["Visa Card", "Visa", "AmEx", "Amex Card", "American Express Card", 
                "Discover Card", "PayPal", "Cash App", "Venmo", "Alipay", "WeChat Pay"]
} 