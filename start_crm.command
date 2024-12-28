#!/bin/bash

# Change to the script's directory
cd "$(dirname "$0")"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed"
    echo "📥 Please install Ollama from https://ollama.ai"
    echo "Press any key to exit"
    read -n 1
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "🚀 Starting Ollama..."
    open -a Ollama
    sleep 5  # Give it time to start
fi

# Check if the gemma2:9b model is installed
if ! ollama list | grep -q "gemma2:9b"; then
    echo "📥 Downloading model gemma2:9b..."
    ollama pull mistral
fi

# Launch the application
echo "🚀 Starting CRM..."
streamlit run crm.py
