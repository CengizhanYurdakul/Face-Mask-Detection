import cv2
from mtcnn import MTCNN
from torchvision import models
import torch
import torchvision.transforms as transforms
from skimage import transform as trans
import numpy as np
import torch.nn as nn
from tqdm import tqdm

transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

alignment_template = np.array([
                                [128*(30.2946+8)/112, 128*51.6963/112],
                                [128*(65.5318+8)/112, 128*51.5014/112],
                                [128*(48.0252+8)/112, 128*71.7366/112],
                                [128*(33.5493+8)/112, 128*92.3655/112],
                                [128*(62.7299+8)/112, 128*92.2041/112]], dtype=np.float32)

alignment_template[:,1]=(alignment_template[:,1]-136)*(128/98)+128
alignment_template[:,0]=(alignment_template[:,0]-15)*(128/98) 

## MOBILENETV2 -------------------------------------
# model = models.mobilenet_v2(pretrained=False)
# model.classifier = nn.Sequential(
#                                 nn.Dropout(0.2),
#                                 nn.Linear(1280, 2)
#                                 )
# model.load_state_dict(torch.load("./Models/MobilenetV2.pth"))
# model.eval()
# model.cuda()
## ------------------------------------------------

## RESNET18 ---------------------------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load("./Models/Resnet18.pth"))
model.eval()
model.cuda()
## ------------------------------------------------

detector = MTCNN()
similarity_transform = trans.SimilarityTransform()

cap = cv2.VideoCapture("./Inputs/video.MOV")
ret, frame = cap.read()
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter("./Outputs/output_video.mp4", cv2.VideoWriter_fourcc(*"XVID"), 30, (frame_width, frame_height))

for i in tqdm(range(total_frame - 15)):
    ret, frame = cap.read()

    if ret is not None:
        frame = cv2.flip(frame, 0)
        results = detector.detect_faces(frame)
        for j in range(len(results)):
            box = results[j]["box"]
            keypoints = results[j]["keypoints"]
            confidence = results[j]["confidence"]

            leftEyeX = keypoints["left_eye"][0]
            leftEyeY = keypoints["left_eye"][1]
            rightEyeX = keypoints["right_eye"][0]
            rightEyeY = keypoints["right_eye"][1]
            noseX = keypoints["nose"][0]
            noseY = keypoints["nose"][1]
            rightMouthX = keypoints["mouth_right"][0]
            rightMouthY = keypoints["mouth_right"][1]
            leftMouthX = keypoints["mouth_left"][0]
            leftMouthY = keypoints["mouth_left"][1]

            affine_points = [[leftEyeX, leftEyeY],
                            [rightEyeX, rightEyeY],
                            [noseX, noseY],
                            [leftMouthX, leftMouthY],
                            [rightMouthX, rightMouthY]]

            x1 = box[0]
            y1 = box[1]
            x2 = box[0] + box[2]
            y2 = box[1] + box[3]

            
            dst = np.array(affine_points).astype(np.float32)
            similarity_transform.estimate(dst, alignment_template)
            M = similarity_transform.params[0:2, :]
            aligned_img = cv2.warpAffine(frame, M, (128, 128), borderMode=cv2.BORDER_REPLICATE).astype("uint8")
            
            transform_img = transform_test(aligned_img)
            
            output = model(transform_img.unsqueeze(0).cuda())
            
            result = torch.argmax(output)

            if int(result) == 1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Face without Mask!", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Face with Mask!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    out.write(frame)
cap.release()