#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIN_CN Dataset Text Masking Tool
=====================
This script processes all text_data.json files for five stocks in the CMIN_CN dataset:
招商蛇口, 五粮液, 永辉超市, 中国平安, and 格力电器.
It replaces sensitive information such as company names, product names, people's names,
and dates according to defined rules.
"""

import os
import json
import re
import shutil
import logging

from masks.CMSK_masks import (
    CMSK_COMPANIES, CMSK_PRODUCTS, CMSK_CEOS, CMSK_EXECS, CMSK_PEERS
)
from masks.WLY_masks import (
    WLY_COMPANIES, WLY_PRODUCTS, WLY_CEOS, WLY_EXECS, WLY_PEERS
)
from masks.YH_masks import (
    YH_COMPANIES, YH_PRODUCTS, YH_CEOS, YH_EXECS, YH_PEERS
)
from masks.PAIC_masks import (
    PAIC_COMPANIES, PAIC_PRODUCTS, PAIC_CEOS, PAIC_EXECS, PAIC_PEERS
)
from masks.GREE_masks import (
    GREE_COMPANIES, GREE_PRODUCTS, GREE_CEOS, GREE_EXECS, GREE_PEERS
)
from masks.common_masks import DATE_PATTERNS, YEAR_PATTERN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cmin_cn_masking.log"),
        logging.StreamHandler()
    ]
)

BASE_PATH = "data/CMIN_CN"

STOCKS_MAP = {
    "招商蛇口": "CMSK",
    "五粮液": "WLY",
    "永辉超市": "YH",
    "中国平安": "PAIC",
    "格力电器": "GREE"
}

def backup_data(stock_name):
    """Create backup for specified stock data"""
    for subset in ["Train", "Test", "Validation"]:
        stock_path = os.path.join(BASE_PATH, subset, stock_name)
        if not os.path.exists(stock_path):
            continue
        
        backup_path = os.path.join(BASE_PATH, subset, f"{stock_name}_backup")
        if os.path.exists(backup_path):
            logging.info(f"Backup already exists: {backup_path}")
            continue
            
        try:
            shutil.copytree(stock_path, backup_path)
            logging.info(f"Backup created: {backup_path}")
        except Exception as e:
            logging.error(f"Failed to create backup: {stock_path} -> {backup_path}, Error: {str(e)}")

def mask_text(text, stock_code):
    """Apply appropriate masking rules to text based on stock code"""

    if stock_code == "CMSK":
        companies = CMSK_COMPANIES
        products = CMSK_PRODUCTS
        ceos = CMSK_CEOS
        execs = CMSK_EXECS
        peers = CMSK_PEERS
    elif stock_code == "WLY":
        companies = WLY_COMPANIES
        products = WLY_PRODUCTS
        ceos = WLY_CEOS
        execs = WLY_EXECS
        peers = WLY_PEERS
    elif stock_code == "YH":
        companies = YH_COMPANIES
        products = YH_PRODUCTS
        ceos = YH_CEOS
        execs = YH_EXECS
        peers = YH_PEERS
    elif stock_code == "PAIC":
        companies = PAIC_COMPANIES
        products = PAIC_PRODUCTS
        ceos = PAIC_CEOS
        execs = PAIC_EXECS
        peers = PAIC_PEERS
    elif stock_code == "GREE":
        companies = GREE_COMPANIES
        products = GREE_PRODUCTS
        ceos = GREE_CEOS
        execs = GREE_EXECS
        peers = GREE_PEERS
    else:
        logging.warning(f"Unknown stock code: {stock_code}, not applying masking rules")
        return text
    
    masked_text = text
    
    for company in sorted(companies, key=len, reverse=True):
        pattern = r'' + re.escape(company) + r''
        masked_text = re.sub(pattern, "COMPANY_X", masked_text)
    
    for product_group in products:
        category = product_group["category"]
        for product_name in product_group["names"]:
            if any(c in product_name for c in r'[]()*.+?^$\\|'):
                pattern = r'' + product_name + r''
            else:
                pattern = r'' + re.escape(product_name) + r''
            masked_text = re.sub(pattern, category, masked_text)
    
    for ceo in sorted(ceos, key=len, reverse=True):
        pattern = r'' + re.escape(ceo) + r''
        masked_text = re.sub(pattern, "PERSON_CEO", masked_text)
    
    for exec_name in sorted(execs, key=len, reverse=True):
        pattern = r'' + re.escape(exec_name) + r''
        masked_text = re.sub(pattern, "PERSON_EXEC", masked_text)
    
    for peer_company in sorted(peers["companies"], key=len, reverse=True):
        pattern = r'' + re.escape(peer_company) + r''
        masked_text = re.sub(pattern, "PEER_COMPANY", masked_text)
    
    for peer_product in sorted(peers["products"], key=len, reverse=True):
        if any(c in peer_product for c in r'[]()*.+?^$\\|'):
            pattern = r'' + peer_product + r''
        else:
            pattern = r'' + re.escape(peer_product) + r''
        masked_text = re.sub(pattern, "PEER_PRODUCT", masked_text)
    
    for date_pattern in DATE_PATTERNS:
        masked_text = re.sub(date_pattern, "DATE_REF", masked_text, flags=re.IGNORECASE)
    
    masked_text = re.sub(YEAR_PATTERN, "YEAR_X", masked_text)
    
    return masked_text

def find_text_files(stock_name):
    """Find all text_data.json files for the specified stock"""
    text_files = []
    
    for subset in ["Train", "Test", "Validation"]:
        stock_path = os.path.join(BASE_PATH, subset, stock_name)
        
        if not os.path.exists(stock_path):
            logging.warning(f"Path does not exist: {stock_path}")
            continue
        
        for root, dirs, files in os.walk(stock_path):
            for file in files:
                if file == "text_data.json":
                    text_files.append(os.path.join(root, file))
    
    return text_files

def process_stock_files(stock_name, stock_code):
    """Process all text_data.json files for the specified stock"""
    files_processed = 0
    texts_masked = 0
    
    backup_data(stock_name)
    
    text_files = find_text_files(stock_name)
    
    if not text_files:
        logging.warning(f"No text_data.json files found for {stock_name}")
        return files_processed, texts_masked
    
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logging.error(f"JSON parsing error: {file_path}")
                    continue
            
            texts_modified = False
            
            if isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
                for day, tweets in data.items():
                    if isinstance(tweets, dict):
                        for tweet_id, tweet_data in tweets.items():
                            if isinstance(tweet_data, dict) and "content" in tweet_data:
                                original_text = tweet_data["content"]
                                masked_text = mask_text(original_text, stock_code)
                                
                                if masked_text != original_text:
                                    tweet_data["content"] = masked_text
                                    texts_modified = True
                                    texts_masked += 1
            
            elif isinstance(data, dict) and any(isinstance(v, str) for v in data.values()):
                for key, text in data.items():
                    if isinstance(text, str):
                        masked_text = mask_text(text, stock_code)
                        
                        if masked_text != text:
                            data[key] = masked_text
                            texts_modified = True
                            texts_masked += 1
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        for field in ["text", "content", "message", "body", "文本", "内容"]:
                            if field in item and isinstance(item[field], str):
                                original_text = item[field]
                                masked_text = mask_text(original_text, stock_code)
                                
                                if masked_text != original_text:
                                    item[field] = masked_text
                                    texts_modified = True
                                    texts_masked += 1
                                break
            
            if texts_modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                files_processed += 1
                logging.info(f"Processed {file_path}, masked {texts_masked} texts")
        
        except Exception as e:
            logging.error(f"Error processing file: {file_path}, Error: {str(e)}")
    
    logging.info(f"{stock_name}({stock_code}) processing complete: processed {files_processed} files, masked {texts_masked} texts")
    return files_processed, texts_masked

def main():
    logging.info("Starting masking process")
    
    total_files = 0
    total_texts = 0
    
    global BASE_PATH
    if not os.path.exists(BASE_PATH):
        alt_path = "../" + BASE_PATH
        if os.path.exists(alt_path):
            BASE_PATH = alt_path
            logging.info(f"Using alternative path: {BASE_PATH}")
        else:
            logging.error(f"Data path does not exist: {BASE_PATH} or {alt_path}")
            return
    
    for stock_name, stock_code in STOCKS_MAP.items():
        logging.info(f"Processing {stock_name}({stock_code}) stock data")
        files, texts = process_stock_files(stock_name, stock_code)
        total_files += files
        total_texts += texts
    
    logging.info(f"All processing complete: processed {total_files} files, masked {total_texts} texts")

if __name__ == "__main__":
    main() 