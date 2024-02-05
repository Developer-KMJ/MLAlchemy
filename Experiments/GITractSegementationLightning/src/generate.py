# Local application/library specific imports
from common import config, MriPlane
from common.consts import MriPlane
from data.data_module import MriDataModule

def main():
    model = MriDataModule(data_dir = config.DATA_DIR, zip_file = config.ZIP_FILE, mri_plane=MriPlane.AXIAL)
    model.prepare_data()

if __name__ == '__main__':
    main()