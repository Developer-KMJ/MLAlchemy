import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import config
from data.data_module import MriDataModule


def main():
    dataModule = MriDataModule(data_dir = config.DATA_DIR, zip_file = config.ZIP_FILE)
    dataModule.prepare_data()

if __name__ == '__main__':
    main()