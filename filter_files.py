#!/usr/bin/env python3
import os
import shutil

def read_exclude_list(file_path):
    """Read a list of files to be excluded from the recalc list file."""
    if not os.path.exists(file_path):
        print(f"Warning: Recalc list {file_path} not found")
        return []
    
    exclude_files = []
    exclude_patterns = []  # Store base patterns without friction coefficients
    
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines, comments, headings or headers
            line = line.strip()
            if not line or line.startswith('//') or line == "На пересчет" or line.endswith('.txt'):
                continue
            
            # Extract just the base filename without comment
            if ' - ' in line:
                basename = line.split(' - ')[0].strip()
            else:
                basename = line.strip()
            
            # Skip if this is not a proper filename
            if not basename or len(basename) < 5:  # Arbitrary minimum length check
                continue
            
            exclude_files.append(basename)
            
            # Extract the base pattern without friction coefficient
            base_pattern = None
            if '_f_' in basename:
                base_pattern = basename.rsplit('_f_', 1)[0]
            elif '_fric_' in basename:
                base_pattern = basename.rsplit('_fric_', 1)[0]
            
            if base_pattern and base_pattern not in exclude_patterns:
                exclude_patterns.append(base_pattern)
    
    return exclude_files, exclude_patterns

def process_directory(vel_dir, exclude_data, dry_run=True):
    """Process a single velocity directory, removing files in the exclude list."""
    exclude_list, exclude_patterns = exclude_data
    
    if not os.path.exists(vel_dir):
        print(f"Directory {vel_dir} not found")
        return
    
    # First count files to be removed
    files_to_remove = []
    for filename in os.listdir(vel_dir):
        file_path = os.path.join(vel_dir, filename)
        
        # Skip directories and the recalc list itself
        if os.path.isdir(file_path) or filename.startswith('recalc_list'):
            continue
        
        # Get the base name without extension for comparison
        base_name = os.path.splitext(filename)[0]
        
        # Check for exact match first
        if base_name in exclude_list:
            files_to_remove.append(file_path)
            continue
            
        # Then check if the base name matches any of the patterns
        # (ignoring friction coefficient)
        for pattern in exclude_patterns:
            if base_name.startswith(pattern) and (
                '_f_' in base_name or '_fric_' in base_name):
                files_to_remove.append(file_path)
                break
    
    # Report and remove files
    if files_to_remove:
        print(f"Found {len(files_to_remove)} files to remove in {vel_dir}")
        
        # Show files to be removed
        for file_path in files_to_remove:
            filename = os.path.basename(file_path)
            print(f"{'Would move' if dry_run else 'Moving'} {filename}")
        
        # If not dry run, actually move the files
        if not dry_run:
            # Create backup directory
            backup_dir = os.path.join(vel_dir, 'removed_files_backup')
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            # Move files to backup instead of deleting
            for file_path in files_to_remove:
                filename = os.path.basename(file_path)
                backup_path = os.path.join(backup_dir, filename)
                shutil.move(file_path, backup_path)
                print(f"Moved {filename} to backup")
    else:
        print(f"No files to remove in {vel_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter files based on recalc lists')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually doing it')
    args = parser.parse_args()
    
    base_dir = "/Users/i.grebenkin/pythonProjects/учеба/wire_drawing/new_components_reports"
    velocities = ["5", "10", "20", "40"]
    
    print(f"Running in {'DRY RUN' if args.dry_run else 'EXECUTE'} mode")
    
    for vel in velocities:
        vel_dir = os.path.join(base_dir, f"Vel_{vel}")
        recalc_list_path = os.path.join(vel_dir, f"recalc_list_Vel_{vel}.txt")
        
        print(f"\nProcessing {vel_dir}...")
        exclude_data = read_exclude_list(recalc_list_path)
        exclude_list, exclude_patterns = exclude_data
        
        print(f"Found {len(exclude_list)} files to exclude in recalc list:")
        # Print the exclude list entries for verification
        for entry in exclude_list:
            print(f"  - {entry}")
            
        print(f"Found {len(exclude_patterns)} base patterns for filtering:")
        for pattern in exclude_patterns:
            print(f"  - {pattern}")
        
        process_directory(vel_dir, exclude_data, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
