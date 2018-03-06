import os
import sys

from commonsdownloader import commonsdownloader

if __name__ == "__main__":
    try:
        commonsdownloader.main()
    except KeyboardInterrupt:
        sys.stderr.write("Terminating run\n")
    sys.stderr.write("Images are available in %s\n" % os.path.abspath("."))
