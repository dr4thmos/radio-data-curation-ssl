import os
import json

# Directory principale
dataset_root = "/home/tcecconello/radioimgs/radio-data-curation-ssl/LoTSS"
cutout_root = os.path.join(dataset_root, "cutouts")

# Funzione per unire i file json di un metodo specifico
def merge_json_files(cutouts_method_path, output_path):
    combined_data = {}
    counter = 0

    for mosaic_name in os.listdir(cutouts_method_path):
        mosaic_path = os.path.join(cutouts_method_path, mosaic_name)
        info_path = os.path.join(mosaic_path, "info.json")

        if os.path.isfile(info_path):
            with open(info_path, "r") as f:
                json_data = json.load(f)

                for item in json_data:
                    combined_data[str(counter)] = {
                        "file_path": os.path.join(mosaic_path, "npy", item["filename"]),
                        "source_type": "UNKNOWN",
                        "survey": item["survey"],
                        "mosaic_name": item["mosaic_name"],
                        "position": item["position"],
                        "size": item["size"]
                    }
                    counter += 1

    # Scrive il file unificato
    with open(output_path, "w") as f:
        json.dump(combined_data, f, indent=4)

# Funzione principale per iterare sui metodi di cutout e creare i file di output
def main():
    global_output = {}
    global_counter = 0

    for cutouts_method in os.listdir(cutout_root):
        cutouts_method_path = os.path.join(cutout_root, cutouts_method)

        if os.path.isdir(cutouts_method_path):
            # Percorso del file unificato per il metodo
            output_path = os.path.join(cutouts_method_path, "info.json")
            
            print(cutouts_method_path)
            print(output_path)
            # Unisce i file json per il metodo specifico
            merge_json_files(cutouts_method_path, output_path)

            # Aggiunge al file globale
            with open(output_path, "r") as f:
                method_data = json.load(f)

                for key, value in method_data.items():
                    global_output[str(global_counter)] = value
                    global_counter += 1

    # Scrive il file globale unificato
    global_output_path = os.path.join(cutout_root, "info.json")
    with open(global_output_path, "w") as f:
        json.dump(global_output, f, indent=4)

if __name__ == "__main__":
    main()
