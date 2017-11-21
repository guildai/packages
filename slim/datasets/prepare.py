import sys

sys.path.insert(0, "slim")

import download_and_convert_data
from datasets import dataset_utils
from tensorflow import gfile

def main():
    # Patch download function as it doesn't check for existing files
    # and we download ourselves (this lets us cache the downloads).
    dataset_utils.download_and_uncompress_tarball = lambda *_args: None

    # Patch gfile.Remove as the slim download wants to delete the
    # archive as cleanup, but it's not there.
    gfile.Remove = lambda *_args: None
    gfile.DeleteRecursively = lambda *_args: None

    # Pass the command along (args to main aren't used)
    download_and_convert_data.main([])

if __name__ == "__main__":
    main()
