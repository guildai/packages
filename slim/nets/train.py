import sys

sys.path.insert(0, "slim")

import train_image_classifier

def main():
    sys.argv.extend([
        "--train_dir", ".",
        "--dataset_split_name", "train",
        "--dataset_dir", "data"
    ])
    train_image_classifier.main([])

if __name__ == "__main__":
    main()
