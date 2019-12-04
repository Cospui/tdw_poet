import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # TODO: this part might need adjustment.
        # what NN shall I use??
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
    def forward(self, x):
        x = F.relu( self.bn1( self.conv1(x)) )
        x = F.relu( self.bn2( self.conv2(x)) )
        x = F.relu( self.bn3( self.conv3(x)) )
        x = F.relu( self.bn4( self.conv4(x)) )
        return x 

resiz = T.Compose([T.ToPILImage(),
                    T.Resize((80,80)),
                    T.ToTensor()])

def main():
    x_np = np.ndarray(shape=(3, 84, 84), dtype=int)
    x_np = np.uint8( x_np.transpose(2,0,1) )
    with torch.no_grad():
        x_np = resiz(x_np).unsqueeze(0).float()
    
    m = Model()
    res = m(x_np)
    print( res.size() ) # (1, 32, 5, 5)
    x = res.view( res.size(0), -1 )
    print( x.size() ) # (1, 800)

if __name__ == "__main__":
    main()