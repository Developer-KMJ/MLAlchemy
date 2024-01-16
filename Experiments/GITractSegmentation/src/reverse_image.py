
from generate_working_mri_images import generate_axial_from_coronal, generate_sagittal_mri


if __name__ == "__main__":
    case_axial_image_path = '/home/kevin/Documents/gitract/data/Generated_3dView/training/Axial/images/case2_day2/'
    outpath = '/home/kevin/Documents/gitract/data/out/'
    generate_axial_from_coronal(case_axial_image_path, outpath, 2, 2)