import os
import yaml
from data_loader import LungDataset
from utils import get_data_splits

def generate_and_save_split_manifests(config):
    """
    Creates a permanent, human-readable record of which files belong to
    the training, validation, and test sets for all defined datasets.
    """
    output_dir = config['output_base_dir']
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in config['datasets']:
        print(f"--- Generating data split manifest for '{dataset_name}' dataset ---")
        full_dataset = LungDataset(dataset_name=dataset_name, config=config)
        train_set, val_set, test_set = get_data_splits(full_dataset)

        def write_manifest(file_path, subset, split_name):
            with open(file_path, 'w') as f:
                f.write(f"# {split_name} set for '{dataset_name}'\n# Total files: {len(subset)}\n# ---\n")
                for i in subset.indices:
                    f.write(os.path.basename(full_dataset.image_files[i]) + "\n")
            print(f"Saved {split_name} set manifest to: {file_path}")

        write_manifest(os.path.join(output_dir, f"{dataset_name}_training_files.txt"), train_set, "Training")
        write_manifest(os.path.join(output_dir, f"{dataset_name}_validation_files.txt"), val_set, "Validation")
        write_manifest(os.path.join(output_dir, f"{dataset_name}_test_files.txt"), test_set, "Test")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    generate_and_save_split_manifests(config)
