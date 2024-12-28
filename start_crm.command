#!/bin/bash

# Переходим в директорию скрипта
cd "$(dirname "$0")"

# Проверяем установлен ли Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama не установлен"
    echo "📥 Установите Ollama с https://ollama.ai"
    echo "Нажмите любую клавишу для выхода"
    read -n 1
    exit 1
fi

# Проверяем запущен ли Ollama
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "🚀 Запускаем Ollama..."
    open -a Ollama
    sleep 5  # Даем время на запуск
fi

# Проверяем установлена ли модель mistral
if ! ollama list | grep -q "mistral"; then
    echo "📥 Загружаем модель mistral..."
    ollama pull mistral
fi

# Запускаем приложение
echo "🚀 Запускаем CRM..."
streamlit run crm.py
