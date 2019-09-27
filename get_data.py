import shutil
import os
from reorg_data import reorg_data

if __name__ == '__main__':
    if not os.path.exists('./dataset'):
        os.mkdir('./dataset')
    shutil.copytree('./dataset/train', './dataset/train_valid')
    os.mkdir('./dataset/valid')
    reorg_data(
        train_dir='./dataset/train',
        valid_dir='./dataset/valid',
        valid_ratio=0.3)
    print('All preprocess finished!')