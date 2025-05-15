#!/bin/bash

# Тестовый скрипт для проверки функциональности split_commits.sh
# Сначала копируем оригинальный скрипт во временный файл

TEMP_SCRIPT=$(mktemp)
cp /Users/i.grebenkin/pythonProjects/учеба/wire_drawing/split_commits.sh $TEMP_SCRIPT

# Модифицируем временный скрипт, чтобы git команды не выполнялись реально
sed -i '' 's/git add/echo "ТЕСТ: git add"/g' $TEMP_SCRIPT
sed -i '' 's/git commit/echo "ТЕСТ: git commit"/g' $TEMP_SCRIPT

# Запускаем модифицированный скрипт
echo "===== Запускаем тестирование скрипта ======"
bash -x $TEMP_SCRIPT

# Удаляем временный файл
rm $TEMP_SCRIPT
