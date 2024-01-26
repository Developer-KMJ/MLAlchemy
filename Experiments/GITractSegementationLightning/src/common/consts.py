#CONSTS

class LossMetric():
    TRAIN_LOSS = 'train_loss'
    VAL_LOSS = 'val_loss'
    MSE_LOSS = 'mse_loss'
    DICE_LOSS = 'dice_loss'
    HAUSDORFF_LOSS = 'hausdorff_loss'
    MIXED_LOSS = 'mixed_loss'

class MriPlane():
    AXIAL='axial'
    CORONAL='coronal'
    SAGITTAL='sagittal'

class DataType():
    IMAGES='images'
    MASKS='masks'

class TrainingStage():
    TRAIN='train'
    VALIDATION='validation'
    TEST='test'
    MINITEST='minitest'
    INTERMEDIATE='intermediate'