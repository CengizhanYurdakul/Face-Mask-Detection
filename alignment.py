import cv2
from mtcnn import MTCNN
import os
import numpy as np
from skimage import transform as trans
from tqdm import tqdm

def getAffine_points(landmark):
    leftEyeX, leftEyeY = landmark["left_eye"][0], landmark["left_eye"][1]
    rightEyeX, rightEyeY = landmark["right_eye"][0], landmark["right_eye"][1]
    noseX, noseY = landmark["nose"][0], landmark["nose"][1]
    rightMouthX, rightMouthY = landmark["mouth_right"][0], landmark["mouth_right"][1]
    leftMouthX, leftMouthY = landmark["mouth_left"][0], landmark["mouth_left"][1]

    affine_points = [[leftEyeX, leftEyeY], [rightEyeX, rightEyeY], [noseX, noseY], [leftMouthX, leftMouthY],
                    [rightMouthX, rightMouthY]]
    return np.array(affine_points).astype(np.float32)

def get_biggest_face(detections):
    _area=0
    dst = None
    for item in detections:
            box, _, landmark = item.values()
            area = box[2]*box[3]
            if area > _area:
                _area = area
                dst = getAffine_points(landmark)
    return dst

path = "./Dataset"
dirs = os.listdir(path)

alignment_template = np.array([
                                [128*(30.2946+8)/112, 128*51.6963/112],
                                [128*(65.5318+8)/112, 128*51.5014/112],
                                [128*(48.0252+8)/112, 128*71.7366/112],
                                [128*(33.5493+8)/112, 128*92.3655/112],
                                [128*(62.7299+8)/112, 128*92.2041/112]], dtype=np.float32)

alignment_template[:,1]=(alignment_template[:,1]-136)*(128/98)+128
alignment_template[:,0]=(alignment_template[:,0]-15)*(128/98) 

detector = MTCNN()
similarity_transform = trans.SimilarityTransform()

for i in dirs:
    n_path = os.path.join(path, i)
    n_dirs = os.listdir(n_path)

    for j in tqdm(n_dirs):
        img = cv2.imread(n_path + "/" + j)
        if img is not None:
            results = detector.detect_faces(img)
            
            if len(results) != 0:    
                dst = get_biggest_face(results)
                similarity_transform.estimate(dst, alignment_template)
                M = similarity_transform.params[0:2, :]
                aligned = cv2.warpAffine(img, M, (128, 128), borderMode=cv2.BORDER_REPLICATE).astype("uint8")
                cv2.imwrite(n_path + "/" + j, aligned)
            
            else:
                os.remove(n_path + "/" + j)
        else:
            os.remove(n_path + "/" + j)