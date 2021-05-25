import torch
import torch.nn as nn
from utils import intersection_over_union




##############################
########## OBJECT DETECTION ##########
########################################

# Y = [class1, class2, ... , class 20, Pc1, Bx1, By1, Bw1, Bh1, Pc2, Bx2, By2, Bw2, Bh2] 

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 26:30])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        ##### for box coordinates
        box_predictions = exists_box*(bestbox*predictions[..., 26:30] + (1-bestbox) *predictions[..., 21:25])
        box_targets = exists_box*target[..., 21:25]
        box_predictions[..., 2:4] =  torch.sign(box_predictions[..., 2:4]) * torch.sqrt( torch.abs( box_predictions[..., 2:4]) + 1e-6 )

        # (N, S, S, 4) => (N*S*S, 4)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_targets, end_dim=-2))

        ##### for object loss
        pred_box = (bestbox * predictions[..., 25:26] + (1-bestbox) * predictions[..., 20:21])
        # (N*S*S)
        object_loss = self.mse(torch.flatten(exists_box*pred_box), torch.flatten(exists_box*target[..., 20:21]) )

        ##### for no object loss
        no_object_loss = self.mse(torch.flatten((1-exists_box) *predictions[..., 20:21], start_dim=1), torch.flatten((1-exists_box)*target[..., 20:21], start_dim=1))
        # NOTE: is this an error??
        no_object_loss += self.mse(torch.flatten((1-exists_box) *predictions[..., 25:26], start_dim=1), torch.flatten((1-exists_box)*target[..., 20:21], start_dim=1))

        ##### for class loss

        # (N, S, S, 20) => (N*S*S, 20)
        class_loss = self.mse(  torch.flatten(exists_box*predictions[..., :20], end_dim=-2), torch.flatten(exists_box*target[..., :20], end_dim=-2))
        loss = (self.lambda_coord*box_loss+object_loss+self.lambda_noobj*no_object_loss+class_loss)

        return loss

##############################
########## OBJECT DETECTION (end) ##########
########################################