#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

os.makedirs("figures", exist_ok=True)

BASE_PATH = "./data"

def get_dataset_paths():
    datasets = {
        "CMIN_US": {
            "Train": f"{BASE_PATH}/CMIN_US/Train/analysis_results.json",
            "Test": f"{BASE_PATH}/CMIN_US/Test/analysis_results.json",
            "Validation": f"{BASE_PATH}/CMIN_US/Validation/analysis_results.json"
        },
        "CMIN_CN": {
            "Train": f"{BASE_PATH}/CMIN_CN/Train/analysis_results.json",
            "Test": f"{BASE_PATH}/CMIN_CN/Test/analysis_results.json", 
            "Validation": f"{BASE_PATH}/CMIN_CN/Validation/analysis_results.json"
        },
        "StockNet": {
            "Train": f"{BASE_PATH}/StockNet/Train/analysis_results.json",
            "Test": f"{BASE_PATH}/StockNet/Test/analysis_results.json",
            "Validation": f"{BASE_PATH}/StockNet/Validation/analysis_results.json"
        }
    }
    
    visualization_datasets = {}
    for dataset, subsets in datasets.items():
        visualization_datasets[dataset] = {}
        for subset, path in subsets.items():
            visualization_datasets[dataset][subset] = f"../{path}"
    
    return datasets, visualization_datasets

def load_dataset_files():
    datasets, visualization_datasets = get_dataset_paths()
    
    use_visualization_path = not os.path.exists(BASE_PATH)
    
    data = {}
    for dataset_name, subsets in (visualization_datasets if use_visualization_path else datasets).items():
        data[dataset_name] = {}
        for subset_name, path in subsets.items():
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data[dataset_name][subset_name] = json.load(f)
                        print(f"Loaded {dataset_name}/{subset_name}")
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")
            else:
                print(f"Warning: File not found - {path}")
    
    return data

def merge_cmin_us_data(data):
    if "CMIN_US" not in data or not data["CMIN_US"]:
        print("Error: No CMIN_US data available")
        return None
    
    cmin_us_data = data["CMIN_US"]
    available_subsets = list(cmin_us_data.keys())
    
    if not available_subsets:
        print("Error: No CMIN_US subsets found")
        return None
    
    base_subset = "Train" if "Train" in available_subsets else available_subsets[0]
    merged_data = {}
    
    merged_data = cmin_us_data[base_subset].copy()
    
    all_stocks = set(merged_data["stock_statistics"].get("stocks_with_complete_text", []))
    missing_stocks = {}
    
    for stock, info in merged_data["stock_statistics"].get("stocks_with_missing_text", {}).items():
        missing_stocks[stock] = info.copy()
    
    for subset_name, subset_data in cmin_us_data.items():
        if subset_name == base_subset:
            continue
            
        all_stocks.update(subset_data["stock_statistics"].get("stocks_with_complete_text", []))
        
        for stock, info in subset_data["stock_statistics"].get("stocks_with_missing_text", {}).items():
            if stock not in missing_stocks:
                missing_stocks[stock] = info.copy()
            else:
                current_missing = missing_stocks[stock]["missing_days"]
                new_missing = info["missing_days"]
                if new_missing > current_missing:
                    missing_stocks[stock] = info.copy()
    
    merged_data["stock_statistics"]["stocks_with_complete_text"] = list(all_stocks)
    merged_data["stock_statistics"]["stocks_with_missing_text"] = missing_stocks
    merged_data["stock_statistics"]["total_stocks"] = len(all_stocks) + len(missing_stocks)
    
    print(f"Merged CMIN_US data: {len(all_stocks)} complete stocks, {len(missing_stocks)} stocks with missing data")
    return merged_data

def extract_dataset_stats(data):
    stats = {}
    
    for dataset_name, subsets in data.items():
        if not subsets:
            continue
            
        if dataset_name == "CMIN_US":
            merged_data = merge_cmin_us_data(data)
            if merged_data:
                dataset_data = merged_data
            else:
                subset_name = list(subsets.keys())[0]
                dataset_data = subsets[subset_name]
        else:
            subset_name = "Train" if "Train" in subsets else list(subsets.keys())[0]
            dataset_data = subsets[subset_name]
        
        if "stock_statistics" not in dataset_data:
            print(f"Warning: No stock statistics found in {dataset_name}")
            continue
            
        stock_stats = dataset_data["stock_statistics"]
        
        total_stocks = stock_stats.get("total_stocks", 0)
        complete_stocks = len(stock_stats.get("stocks_with_complete_text", []))
        incomplete_stocks = {}
        
        for stock, info in stock_stats.get("stocks_with_missing_text", {}).items():
            incomplete_stocks[stock] = {
                "missing_days": info["missing_days"],
                "total_days": info["total_days"],
                "missing_ratio": float(info["missing_percentage"].strip("%")) / 100
            }
        
        bins = [0, 10, 25, 50, 100, 200, 300, 400]
        labels = ['1-10', '11-25', '26-50', '51-100', '101-200', '201-300', '301-399']
        
        missing_days_counts = {}
        for stock, info in incomplete_stocks.items():
            days = info["missing_days"]
            if days in missing_days_counts:
                missing_days_counts[days] += 1
            else:
                missing_days_counts[days] = 1
        
        if missing_days_counts:
            df = pd.DataFrame(list(missing_days_counts.items()), columns=['missing_days', 'stock_count'])
            df = df.sort_values(by='missing_days')
            df['missing_interval'] = pd.cut(df['missing_days'], bins=bins, labels=labels)
            interval_counts = df.groupby('missing_interval')['stock_count'].sum()
        else:
            interval_counts = pd.Series(0, index=labels)
        
        # Store statistics
        stats[dataset_name] = {
            "total_stocks": total_stocks,
            "complete_stocks": complete_stocks,
            "incomplete_stocks": len(incomplete_stocks),
            "missing_days_data": missing_days_counts,
            "interval_counts": interval_counts,
            "incomplete_stocks_data": incomplete_stocks
        }
    
    return stats

def create_comparison_visualization(stats):
    fig = plt.figure(figsize=(16, 18))
    gs = GridSpec(3, 2, figure=fig)
    
    dataset_colors = {
        "CMIN_US": "#4287f5",  # Blue
        "CMIN_CN": "#f54242",  # Red
        "StockNet": "#42f54b"  # Green
    }
    
    ax1 = fig.add_subplot(gs[0, :])
    
    datasets = list(stats.keys())
    complete_counts = [stats[d]["complete_stocks"] for d in datasets]
    incomplete_counts = [stats[d]["incomplete_stocks"] for d in datasets]
    
    ax1.bar(datasets, complete_counts, label='Complete Data', color='#4CAF50')
    ax1.bar(datasets, incomplete_counts, bottom=complete_counts, label='Incomplete Data', color='#F44336')
    
    for i, dataset in enumerate(datasets):
        total = stats[dataset]["total_stocks"]
        ax1.text(i, total + 5, f"Total: {total}", ha='center', fontweight='bold', fontsize=12)
        
        complete_pct = stats[dataset]["complete_stocks"] / total * 100 if total > 0 else 0
        incomplete_pct = stats[dataset]["incomplete_stocks"] / total * 100 if total > 0 else 0
        
        complete_y = complete_counts[i] / 2
        ax1.text(i, complete_y, f"{complete_pct:.1f}%", ha='center', va='center', 
                 fontweight='bold', fontsize=10, color='white')
        
        if incomplete_counts[i] > 0:
            incomplete_y = complete_counts[i] + incomplete_counts[i]/2
            ax1.text(i, incomplete_y, f"{incomplete_pct:.1f}%", ha='center', va='center', 
                     fontweight='bold', fontsize=10, color='white')
    
    ax1.set_title('Complete vs Incomplete Stocks Across Datasets', fontsize=20, fontweight='bold', pad=20)
    ax1.set_ylabel('Number of Stocks', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, :])
    
    intervals = stats[datasets[0]]["interval_counts"].index
    x = np.arange(len(intervals))
    width = 0.8 / len(datasets)
    
    for i, dataset in enumerate(datasets):
        interval_data = stats[dataset]["interval_counts"]
        offset = (i - len(datasets)/2 + 0.5) * width
        bars = ax2.bar(x + offset, interval_data, width, label=dataset, color=dataset_colors[dataset], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax2.set_title('Missing Days Distribution Across Datasets', fontsize=20, fontweight='bold', pad=20)
    ax2.set_xlabel('Missing Days Range', fontsize=14)
    ax2.set_ylabel('Number of Stocks', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(intervals, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])
    
    color_maps = {
        "CMIN_US": LinearSegmentedColormap.from_list('cmin_us_cmap', [(0, '#d4f1f9'), (1, '#0077be')], N=256),
        "CMIN_CN": LinearSegmentedColormap.from_list('cmin_cn_cmap', [(0, '#fcd9d9'), (1, '#a20000')], N=256),
        "StockNet": LinearSegmentedColormap.from_list('stocknet_cmap', [(0, '#d4f9d5'), (1, '#007500')], N=256)
    }
    
    plot_top_stocks(stats["CMIN_US"], "CMIN_US", ax3, color_maps["CMIN_US"], top_n=10, show_most=False)
    
    datasets_to_show = [d for d in datasets if d != "CMIN_US"]
    if datasets_to_show:
        plot_combined_top_stocks(stats, datasets_to_show, ax4, color_maps)
    
    fig.suptitle('Dataset Comparison: Missing Text Data Analysis', fontsize=24, fontweight='bold', y=0.98)
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    
    if os.path.exists("Visualization"):
        save_path_prefix = "Visualization/figures"
    else:
        save_path_prefix = "figures"
        
    plt.savefig(f"{save_path_prefix}/dataset_comparison.png")
    plt.savefig(f"{save_path_prefix}/dataset_comparison.pdf")
    
    print(f"Visualization complete. Figure saved to {save_path_prefix}/ directory.")

def plot_top_stocks(dataset_stats, dataset_name, ax, cmap, top_n=10, show_most=False):
    if not dataset_stats["incomplete_stocks_data"]:
        ax.text(0.5, 0.5, f"No data available for {dataset_name}", 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(f"{dataset_name}: No Data Available", fontsize=16, fontweight='bold')
        return
        
    df = pd.DataFrame.from_dict(dataset_stats["incomplete_stocks_data"], orient='index')
    
    if show_most:
        top_stocks = df.nlargest(top_n, 'missing_days')
        title_prefix = "Most"
    else:
        top_stocks = df.nsmallest(top_n, 'missing_days')
        title_prefix = "Least"
    
    top_n = min(top_n, len(top_stocks))
    
    if top_n == 0:
        ax.text(0.5, 0.5, f"No stocks with missing data in {dataset_name}", 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        return
    
    idx = np.arange(top_n)
    stocks = top_stocks.index.tolist()
    missing_days = top_stocks['missing_days'].values
    
    colors = cmap(missing_days / missing_days.max() if missing_days.max() > 0 else np.zeros_like(missing_days))
    
    bars = ax.barh(
        idx,
        missing_days,
        color=colors,
        height=0.7,
        edgecolor='white',
        linewidth=1
    )
    
    for i, (stock, days) in enumerate(zip(stocks, missing_days)):
        ax.text(
            days + 0.5, 
            i,
            f"{stock} ({int(days)} days)",
            va='center',
            fontsize=10,
            fontweight='bold',
            color='#333333'
        )
    
    ax.set_title(f"{dataset_name}: Top {top_n} Stocks with {title_prefix} Missing Days", 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Missing Days', fontsize=12)
    ax.set_yticks(idx)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

def plot_combined_top_stocks(stats, datasets, ax, color_maps):
    all_top_stocks = []
    
    for dataset in datasets:
        if not stats[dataset]["incomplete_stocks_data"]:
            continue
            
        df = pd.DataFrame.from_dict(stats[dataset]["incomplete_stocks_data"], orient='index')
        df['dataset'] = dataset
        top_stocks = df.nsmallest(5, 'missing_days')
        all_top_stocks.append(top_stocks)
    
    if not all_top_stocks:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14, transform=ax.transAxes)
        return
        
    combined_df = pd.concat(all_top_stocks)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'stock'}, inplace=True)
    
    combined_df.sort_values('missing_days', inplace=True)
    
    idx = np.arange(len(combined_df))
    
    for i, (_, row) in enumerate(combined_df.iterrows()):
        dataset = row['dataset']
        color = color_maps[dataset](0.7)
        
        ax.barh(
            i,
            row['missing_days'],
            color=color,
            height=0.7,
            edgecolor='white',
            linewidth=1,
            label=dataset if dataset not in ax.get_legend_handles_labels()[1] else ""
        )
        
        ax.text(
            row['missing_days'] + 0.5, 
            i,
            f"{row['stock']} ({int(row['missing_days'])} days, {dataset})",
            va='center',
            fontsize=10,
            fontweight='bold',
            color='#333333'
        )
    
    ax.set_title(f"Top Stocks with Least Missing Days (Other Datasets)", 
                fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Missing Days', fontsize=12)
    ax.set_yticks(idx)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')

def print_top_stocks_by_missing_ratio(stats):
    print("\n===== Top 5 Stocks with Lowest Missing Text Ratio by Dataset =====")
    
    for dataset_name, dataset_stats in stats.items():
        print(f"\n{dataset_name} Dataset:")
        
        if not dataset_stats["incomplete_stocks_data"]:
            print("  No stocks with missing text data")
            continue
            
        df = pd.DataFrame.from_dict(dataset_stats["incomplete_stocks_data"], orient='index')
        
        top_stocks = df.nsmallest(5, 'missing_ratio')
        
        print("  Stock    |  Missing Days  |  Total Days  |  Missing Ratio")
        print("  " + "-" * 45)
        
        for stock, row in top_stocks.iterrows():
            print(f"  {stock:<8} |  {int(row['missing_days']):>6}  |  {int(row['total_days']):>6}  |  {row['missing_ratio']*100:>6.2f}%")

if __name__ == "__main__":
    data = load_dataset_files()
    
    if not data:
        print("Error: No dataset files found")
        exit(1)
    
    stats = extract_dataset_stats(data)
    
    if not stats:
        print("Error: Failed to extract statistics from datasets")
        exit(1)
    
    print_top_stocks_by_missing_ratio(stats)
    
    create_comparison_visualization(stats) 