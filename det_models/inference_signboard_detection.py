import pytorch_lightning as pl
import numpy as np
import os
from PIL import Image,ImageDraw
from datasets_signboard_detection.dataset import Labelizer

class Color_convert():
    def __init__(self):
        super().__init__()
        self.labels = {'bien': "red"}
        
    def transform(self, label):
        return self.labels[label]
    
    def num_classes(self):
        return len(self.labels)

def compose(output, mask):
    w,h = mask.shape

    for i in range(0, w):
        for j in range(0,h):
            if (mask[i,j] > 0.5):
                output[i,j] = 1
    return output

class POIDetectionTask(pl.LightningModule):
    def __init__(self,
                 model,
                 data_path):
        super().__init__()
        
        self.model = model
        self.data_path = data_path
        self.output = []
        self.pred_targets = []
        self.pred_masks = []
        self.pred_labels = []
        self.color_convert = Color_convert()

    def forward(self, x):
        output = self.model(x)
        return output

    def predict_step(self, test_batch, batch_idx):
        images, targets = test_batch
        outputs = self(images)
        for target in outputs:
            shape = target['boxes']
            masks = target['masks']
            score = target['scores']
            labels = target['labels']
            shape = shape.numpy()
            masks = masks.numpy()
            score = score.numpy()
            labels = labels.numpy()
            select_shape = []
            select_masks = []
            select_labels = []
            for i in range(0,len(score)):
                if (score[i]>0.7):
                    select_shape.append(shape[i])
                    select_masks.append(masks[i])
                    select_labels.append(labels[i])
            select_shape = [select_shape]
            select_masks = [select_masks]
            select_labels = [select_labels]
            self.pred_targets.append(select_shape)
            self.pred_masks.append(select_masks)
            self.pred_labels.append(select_labels)
    
    def on_predict_end(self):
        list_img = os.listdir(self.data_path)
        for i in range(0,len(list_img)):
            image = Image.open(os.path.join(self.data_path,list_img[i])).convert('RGBA')

            pred_boxes = self.pred_targets[i]
            pred_masks = self.pred_masks[i]
            pred_labels = self.pred_labels[i]

            pred_boxes = pred_boxes[0]
            pred_masks = pred_masks[0]
            pred_labels = pred_labels[0]

            width, height = image.size
            outputs = np.zeros((height,width), dtype="uint8")
            for j in range(0,len(pred_masks)):
                mask = pred_masks[j]
                outputs = compose(outputs, mask[0])
            self.output.append(outputs)
        