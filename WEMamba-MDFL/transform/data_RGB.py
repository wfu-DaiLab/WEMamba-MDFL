import os
from .dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest

def get_training_data(rgb_dir, mask_dir, img_options):
    assert os.path.exists(rgb_dir)
    assert os.path.exists(mask_dir)
    return DataLoaderTrain(rgb_dir, mask_dir, img_options)

def get_validation_data(rgb_dir, mask_dir, img_options):
    assert os.path.exists(rgb_dir)
    assert os.path.exists(mask_dir)
    return DataLoaderVal(rgb_dir, mask_dir, img_options)

def get_test_data(rgb_dir, mask_dir):
    assert os.path.exists(rgb_dir)
    assert os.path.exists(mask_dir)
    return DataLoaderTest(rgb_dir, mask_dir)
