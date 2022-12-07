# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from  sklearn.model_selection import KFold


class CraterDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        self.classes = ['Background','Crater']

    def __getitem__(self, idx):
        # load images and boxes
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        annot_path = os.path.join(self.root, "Annotations", self.annots[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img = cv2.resize(img, (640, 640), cv2.INTER_AREA)
        img= img/255.0

        # retrieve bbox list and format to required type,
        # if annotation file is empty, fill dummy box with label 0
        if os.path.getsize(annot_path) != 0:
            bboxs = np.loadtxt(annot_path, ndmin=2)
            bboxs = self.convert_box_cord(bboxs, 'normxywh', 'xyminmax', img.shape)
            num_objs = len(bboxs)
            bboxs = torch.as_tensor(bboxs, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        else:
            bboxs = torch.as_tensor([[0, 1, 2, 3]], dtype=torch.float32)
            labels = torch.zeros((1,), dtype=torch.int64)
            iscrowd = torch.zeros((1,), dtype=torch.int64)

        area = (bboxs[:, 3] - bboxs[:, 1]) * (bboxs[:, 2] - bboxs[:, 0])
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = bboxs
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            transform_sample = self.transforms(image=img,
                                     bboxes=target['boxes'],
                                     labels=labels)


        img = transform_sample['image']
        target['boxes'] = torch.tensor(transform_sample['bboxes'])
        target['labels'] = torch.tensor(transform_sample['labels'])
        # plot_img_bbox(img,target)
        if target['boxes'].shape == 0:
            target['boxes'] = torch.as_tensor([[0, 1, 2, 3]], dtype=torch.float32)

        return img, target

    def __len__(self):
        return len(self.imgs)


    # Converts boundry box formats, this version assumes single class only!
    def convert_box_cord(self,bboxs, format_from, format_to, img_shape):
        if format_from == 'normxywh':
            if format_to == 'xyminmax':
                xw = bboxs[:, (1, 3)] * img_shape[1]
                yh = bboxs[:, (2, 4)] * img_shape[0]
                xmin = xw[:, 0] - xw[:, 1] / 2
                xmax = xw[:, 0] + xw[:, 1] / 2
                ymin = yh[:, 0] - yh[:, 1] / 2
                ymax = yh[:, 0] + yh[:, 1] / 2
                coords_converted = np.column_stack((xmin, ymin, xmax, ymax))

        return coords_converted


def get_model_bbox(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    if train:
        return A.Compose([  # Check if beneficial, need to edit box attributes
            A.Flip(p=0.5),
            A.RandomResizedCrop(height=640,width=640,p=0.2),
            A.Perspective(p=0.2),
            A.Rotate(p=0.5),
            A.Transpose(p=0.3),
            ToTensorV2(p=1.0)],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([ToTensorV2(p=1.0)],
                         bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img.permute((1,2,0)))
    for box in (target['boxes']):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 edgecolor='r',
                                 facecolor='none',
                                 clip_on=False)
        a.annotate('Crater', (x,y-20), color='red', weight='bold',
                   fontsize=10, ha='left', va='top')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()



def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    k_folds = 5
    num_epochs = 1
    # For fold results
    results = []

    # our dataset has two classes only - background and crater
    num_classes = 2
    # use our dataset and defined transformations
    dataset = CraterDataset('Craters', get_transform(train=True))
    dataset_val = CraterDataset('Craters', get_transform(train=False))

    # Prints an example of image with annotations
    # img, target = dataset[40]
    # plot_img_bbox(img, target)

    # split the dataset in train and test set
    # indices = torch.randperm(len(dataset)).tolist()


    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        dataset_subset = torch.utils.data.Subset(dataset, list(train_ids))
        dataset_val_subset = torch.utils.data.Subset(dataset_val, list(val_ids))


        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
                dataset_subset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val_subset, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        # get the model using our helper function
        model = get_model_bbox(num_classes)
        # model.apply(reset_weights) # Check if beneficial
        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,  # Check if beneficial
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)



        # let's train!

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            test1 = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            results.append = evaluate(model, data_loader_val, device=device)

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(model.state_dict(), save_path)
        print("That's it!")


if __name__ == "__main__":
    main()
