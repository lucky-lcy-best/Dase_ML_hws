import os
import numpy as np
from sklearn.cluster import KMeans
import cv2
from imutils import build_montages
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(resnet50.conv1,
                                    resnet50.bn1,
                                    resnet50.relu,
                                    resnet50.maxpool,
                                    resnet50.layer1,
                                    resnet50.layer2,
                                    resnet50.layer3,
                                    resnet50.layer4)

    def forward(self, x):
        x = self.resnet(x)
        return x

net = Net().eval()

image_path = []
all_images = []
file_path='./faces_4' #设置文件路径（这里为当前目录下的ORL文件夹）
targets = os.listdir(file_path)
for each in targets:
    cur_path = file_path + '/' + each + '/'
    img_names = os.listdir(cur_path)
    for img_name in img_names:
        image_path.append(cur_path + img_name)

for path in image_path:
    image = Image.open(path).convert('RGB')
    image = transforms.Resize([224,224])(image)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    image = net(image)
    image = image.reshape(-1, )
    all_images.append(image.detach().numpy())
clt = KMeans(n_clusters=4)
clt.fit(all_images)
labelIDs = np.unique(clt.labels_)

for labelID in labelIDs:
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size=min(50, len(idxs)),
        replace=False)
    show_box = []
    for i in idxs:
        image = cv2.imread(image_path[i])
        image = cv2.resize(image, (96, 96))
        show_box.append(image)
    montage = build_montages(show_box, (96, 96), (5, 5))[0]

    title = "Type {}".format(labelID)
    cv2.imshow(title, montage)
    cv2.waitKey(0)
