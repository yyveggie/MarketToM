#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIN_US Dataset Text Masking Tool
=====================
This script processes all text_data.json files for five stocks in the CMIN_US dataset:
MA, CMCSA, NKE, JNJ, and SBUX. It replaces sensitive information such as company names,
product names, people's names, and dates according to defined rules.
"""

import os
import json
import re
import shutil
import logging

from masks.MA_masks import (
    MA_COMPANIES, MA_PRODUCTS, MA_CEOS, MA_EXECS, MA_PEERS
)
from masks.CMCSA_masks import (
    CMCSA_COMPANIES, CMCSA_PRODUCTS, CMCSA_CEOS, CMCSA_EXECS, CMCSA_PEERS
)
from masks.NKE_masks import (
    NKE_COMPANIES, NKE_PRODUCTS, NKE_CEOS, NKE_EXECS, NKE_PEERS
)
from masks.JNJ_masks import (
    JNJ_COMPANIES, JNJ_PRODUCTS, JNJ_CEOS, JNJ_EXECS, JNJ_PEERS
)
from masks.SBUX_masks import (
    SBUX_COMPANIES, SBUX_PRODUCTS, SBUX_CEOS, SBUX_EXECS, SBUX_PEERS
)
from masks.common_masks import DATE_PATTERNS, YEAR_PATTERN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cmin_masking.log"),
        logging.StreamHandler()
    ]
)

BASE_PATH = "data/CMIN_US"

STOCKS = ["MA", "CMCSA", "NKE", "JNJ", "SBUX"]

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
    
    if stock_code == "MA":
        companies = MA_COMPANIES
        products = MA_PRODUCTS
        ceos = MA_CEOS
        execs = MA_EXECS
        peers = MA_PEERS
    elif stock_code == "CMCSA":
        companies = CMCSA_COMPANIES
        products = CMCSA_PRODUCTS
        ceos = CMCSA_CEOS
        execs = CMCSA_EXECS
        peers = CMCSA_PEERS
    elif stock_code == "NKE":
        companies = NKE_COMPANIES
        products = NKE_PRODUCTS
        ceos = NKE_CEOS
        execs = NKE_EXECS
        peers = NKE_PEERS
    elif stock_code == "JNJ":
        companies = JNJ_COMPANIES
        products = JNJ_PRODUCTS
        ceos = JNJ_CEOS
        execs = JNJ_EXECS
        peers = JNJ_PEERS
    elif stock_code == "SBUX":
        companies = SBUX_COMPANIES
        products = SBUX_PRODUCTS
        ceos = SBUX_CEOS
        execs = SBUX_EXECS
        peers = SBUX_PEERS
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

def find_text_files(stock_code):
    """Find all text_data.json files for the specified stock"""
    text_files = []
    
    for subset in ["Train", "Test", "Validation"]:
        stock_path = os.path.join(BASE_PATH, subset, stock_code)
        
        if not os.path.exists(stock_path):
            logging.warning(f"Path does not exist: {stock_path}")
            continue
        
        for root, dirs, files in os.walk(stock_path):
            for file in files:
                if file == "text_data.json":
                    text_files.append(os.path.join(root, file))
    
    return text_files

def process_stock_files(stock_code):
    """Process all text_data.json files for the specified stock"""
    files_processed = 0
    texts_masked = 0
    
    backup_data(stock_code)
    
    text_files = find_text_files(stock_code)
    
    if not text_files:
        logging.warning(f"No text_data.json files found for {stock_code}")
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
            
            elif isinstance(data, dict):
                for key, text in data.items():
                    if isinstance(text, str):
                        masked_text = mask_text(text, stock_code)
                        
                        if masked_text != text:
                            data[key] = masked_text
                            texts_modified = True
                            texts_masked += 1
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and "text" in item:
                        original_text = item["text"]
                        masked_text = mask_text(original_text, stock_code)
                        
                        if masked_text != original_text:
                            item["text"] = masked_text
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