#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Amazon(AMZN)相关规则
AMZN_COMPANIES = [
    "Amazon", "Amazon.com", "Amazon.com Inc", "Amazon.com Inc.", "AMZN",
    "Amazon Web Services", "AWS", "Whole Foods", "Whole Foods Market",
    "Twitch", "Twitch Interactive", "Zoox", "Zappos", "PillPack", 
    "Kiva Systems", "Ring", "Ring Inc.", "Audible", "Audible Inc.", "IMDb",
    "Walmart", "Alibaba", "eBay", "JD.com", "JD", "Target", "Shopify"
]

AMZN_PRODUCTS = [
    # 电子商务和会员服务
    {"names": ["Amazon Prime", "Prime", "Prime Video", "Prime Music", "Prime Reading",
              "Prime Gaming", "Prime Day", "Amazon Fresh", "Whole Foods", "Amazon Go",
              "Amazon 4-star", "Amazon Books", "Amazon Pharmacy"], 
     "category": "PRODUCT_ECOMMERCE"},
    
    # 设备
    {"names": ["Kindle", "Kindle Paperwhite", "Kindle Oasis", "Kindle Scribe", 
              "Fire TV", "Fire TV Stick", "Fire TV Cube", "Fire Tablet", "Fire\\s*\\d+",
              "Echo", "Echo Dot", "Echo Show", "Echo Studio", "Echo Auto", "Echo Frames"], 
     "category": "PRODUCT_DEVICE"},
    
    # 云服务
    {"names": ["AWS", "Amazon Web Services", "EC2", "S3", "Lambda", "DynamoDB", "RDS",
              "Redshift", "CloudFront", "Amazon EKS", "Amazon ECS"], 
     "category": "PRODUCT_CLOUD"},
    
    # 智能家居
    {"names": ["Alexa", "Ring", "Ring Doorbell", "Ring Camera", "Ring Alarm", "Blink",
              "Blink Camera", "Blink Video Doorbell", "Blink Outdoor", "eero"], 
     "category": "PRODUCT_HOME"},
    
    # 娱乐和内容
    {"names": ["Audible", "Twitch", "IMDb", "Comixology", "Goodreads", "Amazon Music",
              "Amazon Luna", "Amazon Studios"], 
     "category": "PRODUCT_CONTENT"},
    
    # 物流和自动化
    {"names": ["Amazon Robotics", "Amazon Logistics", "Amazon Air", "Amazon Scout",
              "Amazon Prime Air"], 
     "category": "PRODUCT_LOGISTICS"}
]

AMZN_CEOS = ["Andy Jassy", "andy jassy", "Jeff Bezos", "jeff bezos", "Jeffrey Bezos", "jeffrey bezos"]
AMZN_EXECS = ["Brian Olsavsky", "brian olsavsky", "David Zapolsky", "david zapolsky", 
             "Dave Clark", "dave clark", "Adam Selipsky", "adam selipsky", 
             "Doug Herrington", "doug herrington", "John Felton", "john felton", 
             "Beth Galetti", "beth galetti", "David Limp", "david limp"]

AMZN_PEERS = {
    "companies": ["Walmart", "Target", "Alibaba", "JD.com", "eBay", "Shopify", "Etsy",
                 "Best Buy", "Costco", "Kroger", "Wayfair"],
    "products": ["Walmart\\+", "Walmart Marketplace", "Costco Wholesale", 
                "Target Circle", "Alibaba Tmall", "JD.com", "eBay"]
} 