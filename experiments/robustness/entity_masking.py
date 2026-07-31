import argparse
import json
import re
import shutil
import sys
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robustness.masks.common_masks import DATE_PATTERNS, YEAR_PATTERN

MASK_PACKAGE = "experiments.robustness.masks"
REGEX_META = set("[]()*.+?^$\\|")
DATA_FILES = ("price_data.json", "labels.json")
SPLITS = ("Train", "Validation", "Test")
MASKED_SUFFIX = "_EntityMasked"

DATASET_STOCKS = {
    "StockNet": {"AAPL": "AAPL", "FB": "FB", "T": "T", "GOOG": "GOOG", "AMZN": "AMZN"},
    "CMIN_US": {"MA": "MA", "CMCSA": "CMCSA", "NKE": "NKE", "JNJ": "JNJ", "SBUX": "SBUX"},
    "CMIN_CN": {
        "招商蛇口": "CMSK",
        "五粮液": "WLY",
        "永辉超市": "YH",
        "中国平安": "PAIC",
        "格力电器": "GREE",
    },
}
CHINESE_DATASETS = {"CMIN_CN"}


def load_mask_rules(code):
    module = import_module(f"{MASK_PACKAGE}.{code}_masks")
    return {
        "companies": getattr(module, f"{code}_COMPANIES"),
        "products": getattr(module, f"{code}_PRODUCTS"),
        "ceos": getattr(module, f"{code}_CEOS"),
        "execs": getattr(module, f"{code}_EXECS"),
        "peers": getattr(module, f"{code}_PEERS"),
    }


def build_token_pattern(token, boundary):
    body = token if any(char in REGEX_META for char in token) else re.escape(token)
    return f"{boundary}{body}{boundary}"


def apply_entity_rules(text, rules, chinese):
    boundary = "" if chinese else r"\b"
    flags = 0 if chinese else re.IGNORECASE
    result = text
    for company in sorted(rules["companies"], key=len, reverse=True):
        result = re.sub(f"{boundary}{re.escape(company)}{boundary}", "COMPANY_X", result, flags=flags)
    for group in rules["products"]:
        category = group["category"]
        for name in group["names"]:
            result = re.sub(build_token_pattern(name, boundary), category, result, flags=flags)
    for ceo in sorted(rules["ceos"], key=len, reverse=True):
        result = re.sub(f"{boundary}{re.escape(ceo)}{boundary}", "PERSON_CEO", result, flags=flags)
    for name in sorted(rules["execs"], key=len, reverse=True):
        result = re.sub(f"{boundary}{re.escape(name)}{boundary}", "PERSON_EXEC", result, flags=flags)
    for peer in sorted(rules["peers"]["companies"], key=len, reverse=True):
        result = re.sub(f"{boundary}{re.escape(peer)}{boundary}", "PEER_COMPANY", result, flags=flags)
    for peer in sorted(rules["peers"]["products"], key=len, reverse=True):
        result = re.sub(build_token_pattern(peer, boundary), "PEER_PRODUCT", result, flags=flags)
    for pattern in DATE_PATTERNS:
        result = re.sub(pattern, "DATE_REF", result, flags=re.IGNORECASE)
    result = re.sub(YEAR_PATTERN, "YEAR_X", result)
    if not chinese:
        for company in sorted(rules["companies"], key=len, reverse=True):
            result = re.sub(rf"{re.escape(company)}\s+Store\s+", "COMPANY_X Store ", result, flags=flags)
    return result


def mask_text_payload(payload, rules, chinese):
    if not isinstance(payload, dict):
        return payload
    for tweets in payload.values():
        if not isinstance(tweets, dict):
            continue
        for tweet in tweets.values():
            if isinstance(tweet, dict) and isinstance(tweet.get("content"), str):
                tweet["content"] = apply_entity_rules(tweet["content"], rules, chinese)
    return payload


def derive_masked_dataset(dataset, overwrite=True):
    if dataset not in DATASET_STOCKS:
        raise ValueError(f"Unsupported dataset for entity masking: {dataset}")
    chinese = dataset in CHINESE_DATASETS
    source_root = PROJECT_ROOT / "data" / dataset
    target_root = PROJECT_ROOT / "data" / f"{dataset}{MASKED_SUFFIX}"
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")
    if target_root.exists() and overwrite:
        shutil.rmtree(target_root)
    rules_cache = {code: load_mask_rules(code) for code in set(DATASET_STOCKS[dataset].values())}
    written = []
    for split in SPLITS:
        split_dir = source_root / split
        if not split_dir.exists():
            continue
        for stock_dir, code in DATASET_STOCKS[dataset].items():
            source_stock = split_dir / stock_dir
            if not source_stock.is_dir():
                continue
            target_stock = target_root / split / stock_dir
            target_stock.mkdir(parents=True, exist_ok=True)
            for name in DATA_FILES:
                source_file = source_stock / name
                if source_file.exists():
                    shutil.copyfile(source_file, target_stock / name)
            text_file = source_stock / "text_data.json"
            if text_file.exists():
                payload = json.loads(text_file.read_text(encoding="utf-8"))
                mask_text_payload(payload, rules_cache[code], chinese)
                (target_stock / "text_data.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            written.append(str(target_stock.relative_to(PROJECT_ROOT)))
    return written


def derive_all(datasets=None, overwrite=True):
    names = datasets or list(DATASET_STOCKS.keys())
    return {name: derive_masked_dataset(name, overwrite=overwrite) for name in names}


def masked_stock_pools():
    pools = {}
    for dataset, mapping in DATASET_STOCKS.items():
        stocks = sorted(mapping.keys())
        pools[dataset] = stocks
        pools[f"{dataset}{MASKED_SUFFIX}"] = stocks
    return pools


def main(argv=None):
    parser = argparse.ArgumentParser(description="Derive entity-masked datasets")
    parser.add_argument("--datasets", nargs="+", default=None, choices=list(DATASET_STOCKS.keys()))
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args(argv)
    results = derive_all(args.datasets, overwrite=not args.keep_existing)
    for name, paths in results.items():
        print(f"{name}{MASKED_SUFFIX}: {len(paths)} stock folders written")


if __name__ == "__main__":
    main()
