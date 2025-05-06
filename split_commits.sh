#!/bin/bash

# Скрипт для разбиения большого количества измененных файлов на отдельные коммиты по 100 штук

# Сообщение коммита (можно изменить)
COMMIT_MESSAGE="Batch commit of filtered files"

# Получаем список всех измененных файлов
CHANGED_FILES=$(git status --porcelain | grep -v "??" | awk '{print $2}')
TOTAL_FILES=$(echo "$CHANGED_FILES" | wc -l)

echo "Найдено $TOTAL_FILES измененных файлов"

# Создаем временный файл для хранения списка файлов
TMP_FILE=$(mktemp)
echo "$CHANGED_FILES" > "$TMP_FILE"

# Определяем количество полных коммитов и оставшихся файлов
FULL_COMMITS=$((TOTAL_FILES / 100))
REMAINDER=$((TOTAL_FILES % 100))

echo "Будет создано $FULL_COMMITS полных коммитов и 1 коммит с оставшимися $REMAINDER файлами"
echo "Начинаем процесс..."

# Функция для создания коммита из указанных файлов
commit_files() {
    local start_line=$1
    local end_line=$2
    local batch_num=$3
    local files_to_commit=$(sed -n "${start_line},${end_line}p" "$TMP_FILE")
    
    echo "Коммит $batch_num: добавляем файлы строки с $start_line по $end_line..."
    
    # Добавляем файлы в индекс
    echo "$files_to_commit" | xargs git add
    
    # Создаем коммит
    git commit -m "$COMMIT_MESSAGE (batch $batch_num/$((FULL_COMMITS + (REMAINDER > 0 ? 1 : 0))))"
    
    echo "Коммит $batch_num завершен"
}

# Создаем полные коммиты по 100 файлов
for ((i=0; i<FULL_COMMITS; i++)); do
    START_LINE=$((i * 100 + 1))
    END_LINE=$(((i + 1) * 100))
    commit_files $START_LINE $END_LINE $((i + 1))
done

# Создаем последний коммит с оставшимися файлами, если они есть
if [ $REMAINDER -gt 0 ]; then
    START_LINE=$((FULL_COMMITS * 100 + 1))
    END_LINE=$((FULL_COMMITS * 100 + REMAINDER))
    commit_files $START_LINE $END_LINE $((FULL_COMMITS + 1))
fi

# Удаляем временный файл
rm "$TMP_FILE"

echo "Все коммиты успешно созданы!"
