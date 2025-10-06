#!/bin/bash
# MarketToM Web Server Starter
# Automatically kills old process and starts new one

echo "🔍 Checking for existing processes..."
pkill -9 -f "python.*app.py" 2>/dev/null && echo "✅ Old process killed" || echo "ℹ️  No existing process"

sleep 1

echo "🚀 Starting MarketToM Web Server..."
cd "$(dirname "$0")"
python app.py

