"""
Files list

D:/AISI_1020/Vel_10/svndlkabnvakegreih_fric_0100.odb
D:/AISI_1020/Vel_10/svndlkabnvakegreih_fric_050.odb
D:/AISI_1020/Vel_10/svndlkabnvakegreih_fric_025.odb
D:/AISI_1020/Vel_10/svndlkabnvakegrergberadf_fric_0100.odb
D:/AISI_1020/Vel_10/svndlkabnvakegrergberadf_fric_025.odb
...


Image list (some are missing)
D:/AISI_1020/Vel_10/Pic/svndlkabnvakegreih_fric_0100.png
D:/AISI_1020/Vel_10/Pic/svndlkabnvakegrergberadf_fric_0100.png
D:/AISI_1020/Vel_10/Pic/svndlkabnvakegrergberadf_fric_025.png


мне нужно составить список файлов, которые не были обработаны,
т.е. те, которые не были созданы в папке Pic
при этом выбрать только те файлы, у которых максимальное число после fric_ в имени файла

Python 2.7
"""

import os
import re
from typing import List, Dict


def get_files_from_directory(directory: str) -> List[str]:
    """
    Get all files from the specified directory.
    """
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def get_files_with_max_fric(files: List[str]) -> Dict[str, str]:
    """
    Get files with the maximum number after 'fric_' in their names.
    """
    max_fric_files = {}  # prefix -> (max_fric_value, filename)
    for file in files:
        m = re.match(r'^(.+)_fric_(\d+)\.[^.]+$', file)
        if m:
            prefix = m.group(1)
            fric_value = int(m.group(2))
            if prefix not in max_fric_files or fric_value > max_fric_files[prefix][0]:
                max_fric_files[prefix] = (fric_value, file)
    return {k: v[1] for k, v in max_fric_files.items()}


def get_missing_files(input_dir: str, pic_dir: str) -> List[str]:
    """
    Get a list of files that are in the input directory but not in the Pic directory.
    """
    input_files = get_files_from_directory(input_dir)
    pic_files = get_files_from_directory(pic_dir)

    # Get files with maximum fric value
    max_fric_files = get_files_with_max_fric(input_files)

    # Check which files are missing in the Pic directory
    missing_files = []
    for file in max_fric_files.values():
        pic_file = os.path.splitext(file)[0] + ".png"
        if pic_file not in pic_files:
            missing_files.append(file)

    return missing_files


def main():
    input_dir = "D:/AISI_1020/Vel_10/"
    pic_dir = os.path.join(input_dir, "Pic")

    # Get missing files
    missing_files = get_missing_files(input_dir, pic_dir)

    # Print the missing files
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(file)
    else:
        print("No missing files found.")


if __name__ == "__main__":
    print("Start".removesuffix("t"))
    main()
