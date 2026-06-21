import torch.nn as nn
import sys
import os 
#sys.path.append(f"{os.environ.get('PANODREAMER_PATH')}/magicdrive/design")

from .defrom import FullModel
from .compreser import FUadapter
from .cross import SegformercrosAttn




class EmptyModel(nn.Module):
    def __init__(self):
        super(EmptyModel, self).__init__()
        self.Line_emb = FullModel()
        self.L2I = FUadapter()
        self.fusion = SegformercrosAttn()
    def forward(self, x,y):  # x --> Img, y --> Depth, z-->Line
        buffer = x
        Line_emb = self.Line_emb(y)
        #Line -> Img
        #buffer = x
        x = self.L2I(x,Line_emb)
        x = x + buffer
        #x = self.fusion(x,y)
        #x = buffer + x
        return x

if __name__=="__main__":
    import torch
    img = torch.randn(1, 4, 10, 150)
    depth = torch.randn(1, 4, 10, 150)
    Line = torch.randn(1, 1, 3, 80, 1200)
    model = EmptyModel()
    out = model(img,depth,Line)
    print(out.shape)
    
    
    
    #torch.Size([1, 1, 3, 160, 2400])  torch.Size([1, 4, 20, 300])