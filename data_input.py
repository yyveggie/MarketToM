
import os
import json
import numpy as np
from typing import Dict, Tuple, Any
import logging
import re

def normalize_text(text: str) -> str:
    """Normalize text by replacing newlines and multiple spaces with single space"""
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    return text.strip()

def process_tweets(tweets: Dict) -> Dict:
    """Process tweets while maintaining the original structure"""
    processed_tweets = {}
    for tweet_id, tweet_data in tweets.items():
        content = tweet_data.get('content', '')
        if isinstance(content, list):
            content = ' '.join(content)
        processed_tweets[tweet_id] = {
            'content': normalize_text(content)
        }
    return processed_tweets

def read_json_file(file_path: str, logger) -> Dict:
    """Read and parse a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

def format_day_number(day_str: str) -> str:
    """Format day number with leading zeros"""
    number = re.search(r'\d+', day_str).group()
    formatted_number = number.zfill(3)
    return f"day{formatted_number}"

def validate_days_match(text_days: set, price_days: set, label_days: set, stock: str, logger):
    """Validate that all three files have matching days"""
    if not (text_days == price_days == label_days):
        logger.error(f"Days don't match for stock {stock}")
        logger.error(f"Text days: {text_days}")
        logger.error(f"Price days: {price_days}")
        logger.error(f"Label days: {label_days}")
        raise ValueError(f"Days don't match for stock {stock}")

def process_stock_data(stock_dir: str, logger) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Process data for a single stock directory"""
    text_path = os.path.join(stock_dir, 'text_data.json')
    price_path = os.path.join(stock_dir, 'price_data.json')
    labels_path = os.path.join(stock_dir, 'labels.json')
    
    text_data = read_json_file(text_path, logger)
    price_data = read_json_file(price_path, logger)
    labels_data = read_json_file(labels_path, logger)
    
    formatted_text_data = {
        format_day_number(day): process_tweets(tweets)
        for day, tweets in text_data.items()
    }
    
    formatted_price_data = {
        'price_data': [
            {**d, 'day': format_day_number(d['day'])}
            for d in price_data['price_data']
        ]
    }
    
    formatted_labels_data = {
        'labels': [
            {**d, 'day': format_day_number(d['day'])}
            for d in labels_data['labels']
        ]
    }
    
    text_days = set(formatted_text_data.keys())
    price_days = set(d['day'] for d in formatted_price_data['price_data'])
    label_days = set(d['day'] for d in formatted_labels_data['labels'])
    
    validate_days_match(text_days, price_days, label_days, os.path.basename(stock_dir), logger)
    
    sorted_days = sorted(text_days)
    
    price_array = np.array([
        [d['open'], d['high'], d['low'], d['close'], d['volume']]
        for d in sorted(formatted_price_data['price_data'], key=lambda x: x['day'])
    ])
    
    labels_array = np.array([
        d['label'] for d in sorted(formatted_labels_data['labels'], key=lambda x: x['day'])
    ])
    
    return formatted_text_data, price_array, labels_array

def load_stock_data(base_dir: str, stock_folders: list) -> Tuple[Dict[str, Dict], np.ndarray, np.ndarray]:
    """
    Load and process stock data from specified directory and stock folders.
    
    Args:
        base_dir (str): Base directory containing the stock data
        stock_folders (list): List of stock folder names to process
    
    Returns:
        Tuple containing:
        - Dict: Text data with tweets
        - np.ndarray: Price data array
        - np.ndarray: Labels array
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    all_text_data = {}
    all_price_data = []
    all_labels = []
    
    for stock in stock_folders:
        stock_dir = os.path.join(base_dir, stock)
        if not os.path.isdir(stock_dir):
            logger.warning(f"Skipping {stock} - not a directory")
            continue
            
        try:
            text_data, price_data, labels = process_stock_data(stock_dir, logger)
            
            stock_text_data = {
                f"{stock}{day}": tweet_data
                for day, tweet_data in text_data.items()
            }
            
            all_text_data.update(stock_text_data)
            all_price_data.append(price_data)
            all_labels.append(labels)
            
            logger.info(f"Successfully processed {stock}")
            
        except Exception as e:
            logger.error(f"Error processing {stock}: {str(e)}")
            continue
    
    if not all_price_data:
        raise ValueError("No valid data was processed")
    
    combined_price_data = np.vstack(all_price_data)
    combined_labels = np.concatenate(all_labels)
    
    # Set numpy print options to avoid scientific notation
    np.set_printoptions(suppress=True, precision=2)
    
    return all_text_data, combined_price_data, combined_labels

def debug_days_match(base_dir: str, stock: str):
    """Debug function to check day matching across text, price, and label data."""
    import os
    import json

    base_dir = os.path.abspath(base_dir)
    stock_dir = os.path.join(base_dir, stock)
    text_path = os.path.join(stock_dir, 'text_data.json')
    price_path = os.path.join(stock_dir, 'price_data.json')
    labels_path = os.path.join(stock_dir, 'labels.json')

    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)
        with open(price_path, 'r', encoding='utf-8') as f:
            price_data = json.load(f)['price_data']
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)['labels']
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    text_days = set(text_data.keys())
    price_days = set(d['day'] for d in price_data)
    label_days = set(d['day'] for d in labels_data)

    missing_in_text = price_days - text_days
    missing_in_price = text_days - price_days
    missing_in_labels = text_days - label_days

    print("Missing in text data:", missing_in_text)
    print("Missing in price data:", missing_in_price)
    print("Missing in labels data:", missing_in_labels)

# Example usage
# debug_days_match('./Data/Structured_Data/StockNet/Train', 'AAPL')