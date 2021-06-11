import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader, dataloader
from model import YOLOv1
from dataset import YOLODataset
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)

from loss import YOLOLoss
from datetime import datetime

current_month = datetime.today().month
current_day = datetime.today().day
current_year = datetime.today().year
save_date = str(current_month)+"_"+str(current_day)+"_"+str(current_year)

seed = 123
torch.manual_seed(seed)



##########################################
########### HYPERPARAMETERS ##############
#########################################
learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
weight_decay = 0
epochs = 100
num_workers = 2
pin_memory = True
load_model = False
load_model_file = "data/saved/yolov1"+save_date+".pth"
img_dir = "data/BBlox/BBloxImages"
label_dir = "data/BBlox/labels"
img_dim = (448, 448)

class Compose(object):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms
    
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        
        return img, bboxes


transform = Compose([transforms.Resize(img_dim), transforms.ToTensor()])


def train_yolo(train_loader, model, optimizer, loss_function):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_function(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())
    print(f"mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = YOLOLoss()

    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optimizer)

    train_dataset = YOLODataset("data/BBlox/labels/labels.csv", img_dir=img_dir, label_dir=label_dir)
    # add test dataset here aswell

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4)
        train_yolo(train_loader, model, iou_threshold=0.5, threshold=0.4)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint")

        print(f"train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.9:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, file_name=load_model_file)
            import time 
            time.sleep(10)

if __name__ == "__main__":
    main()

