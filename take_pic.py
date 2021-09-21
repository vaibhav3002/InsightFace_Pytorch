import cv2
import argparse
from pathlib import Path
from PIL import Image
from mtcnn import MTCNN
from datetime import datetime

from PIL import Image
import numpy as np
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name','-n', default='unknown', type=str,help='input the name of the recording person')
parser.add_argument('--image','-i', default=0, type=str,help='image path. Defaults to camera index 0')
args = parser.parse_args()
from pathlib import Path
data_path = Path('data')
save_path = data_path/'facebank'/args.name
if not save_path.exists():
    save_path.mkdir()

mtcnn = MTCNN()

try:
    frame = cv2.imread(args.image)
    p =  Image.fromarray(frame[...,::-1])
    warped_face = np.array(mtcnn.align(p))[...,::-1]
    cv2.imwrite(str(save_path/'{}.jpg'.format(str(datetime.now())[:-7].replace(":","-").replace(" ","-"))), warped_face)
    print('image saved')
except:
    print('no face captured')
