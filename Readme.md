# Expression Classification via PyTorch!

Hi! In this project, I will guide you to train face mask classifier and use it on image, video or webcam in PyTorch.

![Test](Outputs/output-video.gif)

### Steps
I will follow 3 steps;
#### 1. Prepate Data
#### 2. Train Model
#### 3. Process Video or Image with Trained Model

## Prepare Data

Firstly, we need to download dataset that used can be downloaded [here](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG?usp=sharing). Then move dataset folder to main root. `train_test_split.py` will split dataset into 2 classes. Here, the test size was selected as 0.15. You can modify with this value according to the database that you will use.

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
```

## Train Model

In the training part, MobilenetV2, Resnet18 and Resnet50 are available to use. You can choose one of them to train.

### Tensorboard

The tensorboard section is available in the code to examine the results during training. You can change the file extension according to which the results will be examined.

```
## TENSORBOARD
logdir = "./Tensorboard/Experiment3/"
writer = SummaryWriter(logdir)
```

### Transform and Dataloader

Requirements for converting the images in dataset to tensor and getting them ready for the model are available in the code. In transform, data augmentation methods were used. You can remove this part or change the probabilities in accordance with your own project.

```
## TRANSFORM
transform_ori = transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.2, hue=0.05),
                                    transforms.RandomPerspective(distortion_scale=0.04, p=0.4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
```

Since the image sizes are small, I set the batch size to 64. You can change this value according to the computer you use.

```
## DATASET
batch_size = 64
train_load = torch.utils.data.DataLoader(dataset = train_dataset,
                                         batch_size = batch_size,
                                         shuffle = True)
```
### Model Selection

Since we have 2 classes, we are reducing the last layer of the models(MobilenetV2, Resnet18, Resnet50)  to 2. If this number is different in your dataset, you can change it. This dataset is relatively small.

```
## MOBILENETV2 --------------------------------------
# model = models.mobilenet_v2(pretrained=False)
# model.classifier = nn.Sequential(
#                                 nn.Dropout(0.2),
#                                 nn.Linear(1280, 2)
#                                 )
# ---------------------------------------------------

## RESNET18 -----------------------------------------
# model = models.resnet18(pretrained=False)
# model.fc = nn.Linear(512, 2)
# ---------------------------------------------------

## RESNET50 -----------------------------------------
# model = models.resnet50(pretrained=False)
# model.fc = nn.Linear(2048, 2)
# ---------------------------------------------------
```

### Other Parameters

Loss function and optimizer are as shown.

```
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```
### Training

Once all the preparations have been made we can run the code. After printing the layers of the model, we can see the results for each epoch. Then we can examine the results we got from the tensorboard.

```
Epoch 1/100, Training Loss: 0.576, Training Accuracy: 84.000, Testing Loss: 0.545, Testing Acc: 83.000, Time: 10.3512s
...
Epoch 7/100, Training Loss: 0.066, Training Accuracy: 97.000, Testing Loss: 0.061, Testing Acc: 98.000, Time: 10.3451s
```

## Process Video or Image with Trained Model

### Test Image
In line 47, you can change image path that you want to be tested. Output image will be saved to Outputs folder.

```
img = cv2.imread("./Inputs/with_mask.JPG")
```

### Test Video
In line 48, you can change video path that you want to be tested. Output video will be saved to Outputs folder

```
cap = cv2.VideoCapture("./Inputs/video.MOV")
```

## Requirements
- Torch
- Torchvision
- Matplotlib
- Numpy
- MTCNN
- Opencv-Python
- Pillow
- Pandas

```
pip install -r requirements.txt
```

## Outputs
Pretrained MobilenetV2 will be in Models folder.
