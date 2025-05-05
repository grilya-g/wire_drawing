import os

def get_odb_files_not_in_exclude(exclude_txt_path, odb_folder_path):
    """
    Returns a list of ODB filenames in the folder such that the filename (without extension, minus last 4 chars)
    is not present in the exclude list (also minus last 4 chars for each entry).
    """
    with open(exclude_txt_path, 'r', encoding='utf-8') as f:
        exclude_names = set(line.strip()[:-4] for line in f if line.strip())

    odb_files = [f for f in os.listdir(odb_folder_path) if f.lower().endswith('.odb')]

    result = []
    for fname in odb_files:
        name_wo_ext = os.path.splitext(fname)[0]
        name_cut = name_wo_ext[:-4]
        if name_cut not in exclude_names:
            result.append(fname)
    return result

# Example usage:
# files = get_odb_files_not_in_exclude('path/to/exclude_list.txt', 'path/to/odb_folder')
# for f in files:
#     print(f)