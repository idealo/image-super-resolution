import os
import sys
import zipfile
import urllib.request
from multiprocessing.pool import ThreadPool

FILES_INFO = [
    {'archive': 'DIV2K_valid_HR.zip', 'folder': 'DIV2K_valid_HR', 'download': True, 'extract': True, 'size': 100},
    {
        'archive': 'DIV2K_valid_LR_bicubic_X2.zip',
        'folder': 'DIV2K_valid_LR_bicubic/X2',
        'download': True,
        'extract': True,
        'size': 100,
    },
    {'archive': 'DIV2K_train_HR.zip', 'folder': 'DIV2K_train_HR', 'download': True, 'extract': True, 'size': 800},
    {
        'archive': 'DIV2K_train_LR_bicubic_X2.zip',
        'folder': 'DIV2K_train_LR_bicubic/X2',
        'download': True,
        'extract': True,
        'size': 800,
    },
]

URL = 'http://data.vision.ee.ethz.ch/cvl/DIV2K/{}'

SCRIPT_DIR = sys.path[0]
ISR_DIR = os.path.dirname(SCRIPT_DIR)
WORKDIR = os.path.join(ISR_DIR, 'data', 'DIV2K')

if not os.path.exists(WORKDIR):
    os.makedirs(WORKDIR)


def download_and_extract(file_info):
    archive = file_info['archive']
    file_dest = os.path.join(WORKDIR, archive)

    if file_info['download']:
        print('Downloading', archive)
        try:
            urllib.request.urlretrieve(URL.format(archive), file_dest)
            print('Downloaded', archive)
        except Exception as e:
            print(e)
            print('Failed to download', archive)
            return 1
        extract = True
    if file_info['extract']:
        print('Extracting', archive)
        try:
            zip = zipfile.ZipFile(file_dest)
            zip.extractall(WORKDIR)
        except Exception as e:
            print(e)
            print('Failed to extract', archive)
            return 2
    return 0


def images_are_present(file_info):
    """Checks if folder exists, if so, checks number of images."""
    currentdir = os.path.join(WORKDIR, file_info['folder'])
    if not os.path.exists(currentdir):
        return False
    count = len([x for x in os.listdir(currentdir) if x.endswith('.png')])
    if count != file_info['size']:
        print([x for x in os.listdir(currentdir) if x.endswith('.png')])
        print('Count does not match')
        print(count)
        print(file_info['size'])
        return False
    return True


def archive_exists(file_info):
    """Checks if zip archive is present."""
    if not os.path.exists(os.path.join(WORKDIR, file_info['archive'])):
        return False
    return True


def get_DIV2K():
    for file_info in FILES_INFO:
        if images_are_present(file_info):
            file_info['extract'] = False
        if archive_exists(file_info):
            file_info['download'] = False

    # download and extract in parallel
    pool = ThreadPool(processes=4)
    pool.map(download_and_extract, FILES_INFO)


if __name__ == '__main__':
    print('Data will be under', WORKDIR)
    get_DIV2K()
