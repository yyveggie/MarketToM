#!/usr/bin/env python3
import os
import json
import re

DATA_DIR = "./data"
DATASETS = ["StockNet", "CMIN_US", "CMIN_CN"]
SUBSETS = ["Train", "Test", "Validation"]

def replace_company_name(text, company_name):
    """
    Replace company name in text (case insensitive)
    """
    pattern = re.compile(r'\b' + re.escape(company_name) + r'\b', re.IGNORECASE)
    return pattern.sub("COMPANY_X", text)

def process_file(file_path, company_name):
    """
    Process a single text_data.json file, replacing company names
    """
    print(f"Processing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        modified = False
        
        def process_json(obj, company_name):
            nonlocal modified
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        process_json(value, company_name)
                    elif isinstance(value, str):
                        new_value = replace_company_name(value, company_name)
                        if new_value != value:
                            obj[key] = new_value
                            modified = True
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        process_json(item, company_name)
                    elif isinstance(item, str):
                        new_item = replace_company_name(item, company_name)
                        if new_item != item:
                            obj[i] = new_item
                            modified = True
        
        process_json(data, company_name)
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  Company name replaced: {company_name} -> COMPANY_X")
        else:
            print(f"  No matching company name found: {company_name}")
            
        return modified
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return False

def main():
    """
    Main function: Iterate through all datasets and process all text_data.json files
    """
    modified_count = 0
    total_count = 0
    
    for dataset in DATASETS:
        dataset_path = os.path.join(DATA_DIR, dataset)
        if not os.path.exists(dataset_path):
            print(f"Dataset path does not exist: {dataset_path}")
            continue
            
        for subset in SUBSETS:
            subset_path = os.path.join(dataset_path, subset)
            if not os.path.exists(subset_path):
                print(f"Subset path does not exist: {subset_path}")
                continue
                
            for company_dir in os.listdir(subset_path):
                company_path = os.path.join(subset_path, company_dir)
                if os.path.isdir(company_path):
                    company_name = company_dir
                    
                    text_data_path = os.path.join(company_path, "text_data.json")
                    if os.path.exists(text_data_path):
                        total_count += 1
                        if process_file(text_data_path, company_name):
                            modified_count += 1
    
    print(f"\nProcessing complete! Processed {total_count} files, {modified_count} files were modified.")

if __name__ == "__main__":
    main() 