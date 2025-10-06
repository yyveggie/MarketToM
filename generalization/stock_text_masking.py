#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockNet Dataset Text Masking Tool
======================
This script processes all text_data.json files for five stocks in the StockNet dataset:
AAPL, FB, T, GOOG, and AMZN. It replaces sensitive information such as company names,
product names, people's names, and dates according to defined rules.
"""

import os
import json
import re
import shutil
import logging

from masks.AAPL_masks import (
    AAPL_COMPANIES, AAPL_PRODUCTS, AAPL_CEOS, AAPL_EXECS, AAPL_PEERS
)
from masks.FB_masks import (
    FB_COMPANIES, FB_PRODUCTS, FB_CEOS, FB_EXECS, FB_PEERS
)
from masks.T_masks import (
    T_COMPANIES, T_PRODUCTS, T_CEOS, T_EXECS, T_PEERS
)
from masks.GOOG_masks import (
    GOOG_COMPANIES, GOOG_PRODUCTS, GOOG_CEOS, GOOG_EXECS, GOOG_PEERS
)
from masks.AMZN_masks import (
    AMZN_COMPANIES, AMZN_PRODUCTS, AMZN_CEOS, AMZN_EXECS, AMZN_PEERS
)
from masks.common_masks import DATE_PATTERNS, YEAR_PATTERN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_masking.log"),
        logging.StreamHandler()
    ]
)

BASE_PATH = "data/StockNet"

STOCKS = ["AAPL", "FB", "T", "GOOG", "AMZN"]

def backup_data(stock_code):
    """Create backup for specified stock data"""
    for subset in ["Train", "Test", "Validation"]:
        stock_path = os.path.join(BASE_PATH, subset, stock_code)
        if not os.path.exists(stock_path):
            continue
        
        backup_path = os.path.join(BASE_PATH, subset, f"{stock_code}_backup")
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
    
    if stock_code == "AAPL":
        companies = AAPL_COMPANIES
        products = AAPL_PRODUCTS
        ceos = AAPL_CEOS
        execs = AAPL_EXECS
        peers = AAPL_PEERS
    elif stock_code == "FB":
        companies = FB_COMPANIES
        products = FB_PRODUCTS
        ceos = FB_CEOS
        execs = FB_EXECS
        peers = FB_PEERS
    elif stock_code == "T":
        companies = T_COMPANIES
        products = T_PRODUCTS
        ceos = T_CEOS
        execs = T_EXECS
        peers = T_PEERS
    elif stock_code == "GOOG":
        companies = GOOG_COMPANIES
        products = GOOG_PRODUCTS
        ceos = GOOG_CEOS
        execs = GOOG_EXECS
        peers = GOOG_PEERS
    elif stock_code == "AMZN":
        companies = AMZN_COMPANIES
        products = AMZN_PRODUCTS
        ceos = AMZN_CEOS
        execs = AMZN_EXECS
        peers = AMZN_PEERS
    else:
        logging.warning(f"Unknown stock code: {stock_code}, not applying masking rules")
        return text
    
    masked_text = text
    
    for company in sorted(companies, key=len, reverse=True):
        pattern = r'\b' + re.escape(company) + r'\b'
        masked_text = re.sub(pattern, "COMPANY_X", masked_text, flags=re.IGNORECASE)
    
    for product_group in products:
        category = product_group["category"]
        for product_name in product_group["names"]:
            pattern = r'\b' + product_name + r'\b'
            masked_text = re.sub(pattern, category, masked_text, flags=re.IGNORECASE)
    
    for ceo in sorted(ceos, key=len, reverse=True):
        pattern = r'\b' + re.escape(ceo) + r'\b'
        masked_text = re.sub(pattern, "PERSON_CEO", masked_text)
    
    for exec_name in sorted(execs, key=len, reverse=True):
        pattern = r'\b' + re.escape(exec_name) + r'\b'
        masked_text = re.sub(pattern, "PERSON_EXEC", masked_text)
    
    for peer_company in sorted(peers["companies"], key=len, reverse=True):
        pattern = r'\b' + re.escape(peer_company) + r'\b'
        masked_text = re.sub(pattern, "PEER_COMPANY", masked_text, flags=re.IGNORECASE)
    
    for peer_product in sorted(peers["products"], key=len, reverse=True):
        pattern = r'\b' + peer_product + r'\b'
        masked_text = re.sub(pattern, "PEER_PRODUCT", masked_text, flags=re.IGNORECASE)
    
    for date_pattern in DATE_PATTERNS:
        masked_text = re.sub(date_pattern, "DATE_REF", masked_text, flags=re.IGNORECASE)
    
    masked_text = re.sub(YEAR_PATTERN, "YEAR_X", masked_text)
    
    for company in sorted(companies, key=len, reverse=True):
        store_pattern = re.escape(company) + r'\s+Store\s+'
        masked_text = re.sub(store_pattern, "COMPANY_X Store ", masked_text, flags=re.IGNORECASE)
    
    return masked_text

def process_stock_files(stock_code):
    """Process all text_data.json files for the specified stock"""
    files_processed = 0
    texts_masked = 0
    
    backup_data(stock_code)
    
    for subset in ["Train", "Test", "Validation"]:
        stock_path = os.path.join(BASE_PATH, subset, stock_code)
        
        if not os.path.exists(stock_path):
            logging.warning(f"Path does not exist: {stock_path}")
            continue
        
        file_path = os.path.join(stock_path, "text_data.json")
        if not os.path.exists(file_path):
            logging.warning(f"File does not exist: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts_modified = False
            if isinstance(data, dict):
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
            
            if texts_modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                files_processed += 1
                logging.info(f"Processed {file_path}, masked {texts_masked} texts")
        
        except Exception as e:
            logging.error(f"Error processing file: {file_path}, Error: {str(e)}")
    
    logging.info(f"{stock_code} processing complete: processed {files_processed} files, masked {texts_masked} texts")
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
    
    for stock in STOCKS:
        logging.info(f"Processing {stock} stock data")
        files, texts = process_stock_files(stock)
        total_files += files
        total_texts += texts
    
    logging.info(f"All processing complete: processed {total_files} files, masked {total_texts} texts")

if __name__ == "__main__":
    main() 