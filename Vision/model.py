import torch
import torch.nn as nn
import torch.nn.functional as F 


##############################
########## OBJECT DETECTION ##########
########################################

class CNNBlock(nn.Module):
    """
    KEY:

    architecture_config:
        - tuples => kernel_size, num_filters, stride, padding
        - "M" => max pooling
        - lists => (tuples) and then last integer is the number of repeats


    Y = [class1, class2, ... , class 20, Pc1, Bx1, By1, Bw1, Bh1, Pc2, Bx2, By2, Bw2, Bh2] ( length = 30 )

    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        self.architecture = [
                                (7, 64, 2, 3), 
                                "M",
                                (3, 192, 1, 1),
                                "M", 
                                (1, 128, 1, 0),
                                (3, 256, 1, 1),
                                (1, 256, 1, 0),
                                (3, 512, 1, 1),
                                "M",
                                [(1, 256, 1, 0), (3, 512, 1, 1), 4],
                                (1, 512, 1, 0),
                                (3, 1024, 1, 1),
                                "M",
                                [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
                                (3, 1024, 1, 1),
                                (3, 1024, 2, 1),
                                (3, 1024, 1, 1),
                                (3, 1024, 1, 1),
                            ]

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def _create_conv_layers(self):
        layers = []
        in_channels = self.in_channels

        for x in self.architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers+= [CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers+= [CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]

                    in_channels = conv2[1]
        return nn.Sequential(*layers)


    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C+B*5))

        )


def test_YOLOv1(S=7, B=2, C=20):
    model = YOLOv1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)            


#test()

##############################
########## OBJECT DETECTION (end) ##########
########################################



##############################
########## CLASSIFICATION ##########
########################################
class LeNet5(nn.Module):
    """
    Use this model to classify numbers, TODO try training it on mnist
    """

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.architecture = [
            # (W, kernel_size, P, S)
            # TODO
            

        ]
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs






##############################
########## CLASSIFICATION (end) ##########
########################################