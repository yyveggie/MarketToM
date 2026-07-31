#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Starbucks(SBUX)相关规则
SBUX_COMPANIES = [
    "Starbucks", "Starbucks Corporation", "Starbucks Corp", "Starbucks Corp.",
    "SBUX", "Starbucks Coffee", "Starbucks Coffee Company", 
    "Seattle's Best Coffee", "Teavana", "Ethos Water", "Evolution Fresh",
    "La Boulange", "Tazo", "Torrefazione Italia"
]

SBUX_PRODUCTS = [
    # 咖啡饮料
    {"names": ["Frappuccino", "Starbucks Frappuccino", "Caramel Frappuccino", "Mocha Frappuccino",
              "Java Chip Frappuccino", "Latte", "Caffe Latte", "Cappuccino", "Americano", 
              "Caramel Macchiato", "Flat White", "Cold Brew", "Nitro Cold Brew", 
              "Espresso", "Starbucks Espresso", "Vanilla Latte", "White Chocolate Mocha", 
              "Caramel Brulée Latte", "Pumpkin Spice Latte", "PSL"], 
     "category": "PRODUCT_COFFEE"},
    
    # 茶饮料
    {"names": ["Teavana", "Chai Tea Latte", "Chai Latte", "Green Tea", "Black Tea", 
              "Matcha", "Matcha Latte", "Iced Tea", "Passion Tea", "Earl Grey", 
              "London Fog", "Shaken Iced Tea", "Refresher", "Pink Drink", "Dragon Drink"], 
     "category": "PRODUCT_TEA"},
    
    # 食品
    {"names": ["Cake Pop", "Cake Pops", "Scone", "Bagel", "Croissant", "Muffin", 
              "Danish", "Breakfast Sandwich", "Protein Box", "Sous Vide Egg Bites",
              "Egg Bites", "Oatmeal", "Cookie", "Brownie"], 
     "category": "PRODUCT_FOOD"},
    
    # 咖啡豆和包装产品
    {"names": ["Starbucks Reserve", "Pike Place Roast", "Pike Place", "Blonde Roast", 
              "Medium Roast", "Dark Roast", "Sumatra", "Veranda Blend", "Coffee Beans", 
              "Ground Coffee", "K-Cup", "K-Cups", "Via", "Starbucks Via", "Via Instant", 
              "Starbucks Pods", "Starbucks Capsules"], 
     "category": "PRODUCT_RETAIL_COFFEE"},
    
    # 商品和设备
    {"names": ["Tumbler", "Starbucks Tumbler", "Mug", "Starbucks Mug", "Gift Card",
              "Starbucks Card", "Starbucks Gift Card", "Cold Cup", "Hot Cup"], 
     "category": "PRODUCT_MERCHANDISE"},
    
    # 项目和服务
    {"names": ["Starbucks Rewards", "Rewards Program", "Stars", "Reward Stars", 
              "Starbucks App", "Mobile Order", "Mobile Pay", "Starbucks Delivery"], 
     "category": "PRODUCT_SERVICE"}
]

SBUX_CEOS = ["Laxman Narasimhan", "laxman narasimhan", "Howard Schultz", "howard schultz",
            "Kevin Johnson", "kevin johnson", "Orin Smith", "orin smith"]
SBUX_EXECS = ["Rachel Ruggeri", "rachel ruggeri", "John Culver", "john culver", 
             "Michael Conway", "michael conway", "Frank Britt", "frank britt", 
             "Sara Trilling", "sara trilling", "Molly Liu", "molly liu", 
             "George Dowdie", "george dowdie", "Rachel Gonzalez", "rachel gonzalez"]

SBUX_PEERS = {
    "companies": ["McDonald's", "Dunkin'", "Dunkin Donuts", "Costa Coffee", "Tim Hortons", 
                 "Peet's Coffee", "Dutch Bros", "Coffee Bean & Tea Leaf", "Caribou Coffee",
                 "Blue Bottle Coffee", "Lavazza", "Luckin Coffee", "Panera Bread",
                 "Yum! Brands", "Restaurant Brands International", "Wendy's", "Chipotle"],
    "products": ["McCafé", "Dunkin' Coffee", "Dunkin' Donuts", "Costa Latte", "Tims Coffee", 
                "Double Double", "Peet's Cold Brew", "Dutch Bros Rebel", "Blue Bottle Latte"]
} 