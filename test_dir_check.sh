#!/bin/bash

# Простой скрипт для проверки доступности директории
TARGET_DIR="/Users/i.grebenkin/pythonProjects/учеба/wire_drawing/analytics/statistical_analysis"

# Проверка существования директории
echo "Проверяем существование директории $TARGET_DIR"
if [ ! -d "$TARGET_DIR" ]; then
    echo "Ошибка: директория $TARGET_DIR не существует"
    exit 1
fi

# Пробуем перейти в целевую директорию
echo "Пробуем перейти в директорию $TARGET_DIR"
cd "$TARGET_DIR" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Ошибка: не удалось перейти в директорию $TARGET_DIR"
    exit 1
else
    echo "Успешно перешли в директорию $(pwd)"
    ls -la | head -5
fi
