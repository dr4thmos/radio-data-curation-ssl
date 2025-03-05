import os

def check_folders_without_info_json(base_directory):
    missing_info_json = []
    
    for root, dirs, files in os.walk(base_directory):
        for directory in dirs:
            folder_path = os.path.join(root, directory)
            if 'info.json' not in os.listdir(folder_path):
                missing_info_json.append(folder_path)
    
    return missing_info_json

# Specifica la directory di base
base_directory = "/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS/cutouts/sw_cutouts_256"

missing_folders = check_folders_without_info_json(base_directory)
if missing_folders:
    print("Cartelle senza info.json:")
    for folder in missing_folders:
        print(folder)
else:
    print("Tutte le cartelle contengono un file info.json.")
