import os
from correct_odb_listing import get_odb_files_not_in_exclude
from report_maker import for_report, path_import

# Укажите путь к exclude.txt и папке с ODB
exclude_txt_path = os.path.join(path_import, 'exclude.txt')
odb_folder_path = path_import

if not os.path.exists(exclude_txt_path):
    raise FileNotFoundError('exclude.txt не найден по пути: {}'.format(exclude_txt_path))

odb_files = get_odb_files_not_in_exclude(exclude_txt_path, odb_folder_path)
print('ODB files for report:', odb_files)
for_report(odb_files)
