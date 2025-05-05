import os

folder = '/путь/к/папке'
filename = 'имя_файла.txt'
full_path = os.path.join(folder, filename)
print(full_path)