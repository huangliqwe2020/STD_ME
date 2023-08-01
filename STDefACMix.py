import cv2
import torch
import numpy as np
import torch.nn as nn
from torchsummary  import summary
import torch.nn.functional as F
from thop import profile
import cv2 as cv
from ltr.models.STDeformable_ACMix.deformable_attn import DeformableHeadAttention
from ltr.models.STDeformable_ACMix.defAttn3D import DeformableAttention3D
from ltr.models.STDeformable_ACMix.dcn.modules.modulated_deform_conv import ModulatedDeformConvPack as dcn
from ltr.models.STDeformable_ACMix.dcn1.modules.deform_conv import DeformConvPack as dcn1
def generate_ref_points(width: int,
                        height: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid_y = grid_y / (height - 1)
    grid_x = grid_x / (width - 1)

    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    return grid


def restore_scale(width: int,
                  height: int,
                  ref_point: torch.Tensor):
    new_point = ref_point.clone().detach()
    c = new_point[..., 0]
    d = [[1,2,3],[1,2,3]]
    new_point[..., 0] = new_point[..., 0] * (width - 1)
    new_point[..., 1] = new_point[..., 1] * (height - 1)

    return new_point

def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)

class STDefACMixAttn(nn.Module):
    r"""
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, H, W, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        # self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim//4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.DefAttn = DeformableHeadAttention(self.num_heads,
                                          self.dim//8,
                                          4,
                                          H,
                                          W,
                                          scales=1,
                                          dropout=0.1,
                                          need_attn=False).cuda()
        self.DefAttn3D = DeformableAttention3D(dim=self.dim//8,
                                     dim_head=32,
                                     heads=1,
                                     dropout=0,
                                     downsample_factor=(2, 8, 8),
                                     offset_scale=(2, 8, 8),
                                     offset_kernel_size=(4, 10, 10), )
        self.H = H
        self.W = W

        # fully connected layer in Fig.2
        self.fc = nn.Conv2d(self.num_heads*2, 9, kernel_size=1, bias=True).cuda()
        self.large = nn.Conv2d(64, 512, kernel_size=1, bias=True).cuda()

        self.offset = nn.Conv2d(1536, 18,1)
        self.offset1 = nn.Linear(1536, 64)
        self.dep_conv = dcn(9 * dim // (8*self.num_heads), 27*dim // (8*self.num_heads), kernel_size=(3,3), stride=1,
                            groups=dim // (8*self.num_heads), padding=1)
        self.STConv = dcn1(27*dim // (8*self.num_heads), dim, groups=dim // (8*self.num_heads), kernel_size=1, stride=1,
                             padding=0)
        # rates for both paths
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        ones(self.rate1)
        ones(self.rate2)


    def forward(self, x, nimg, mode="encoder", mask=None):
        sumflops = 0
        H = self.H
        W = self.W

        x = x.permute(1,0,2)
        B, nhw, C = x.shape
        C1 = C//8
        qkv = self.qkv(x)

        # Flops, params = profile(self.qkv, inputs=(x,))
        # sumflops += Flops

        # fully connected layer
        f_all = qkv.reshape(B, nimg*H*W, self.num_heads*2, -1).permute(0, 2, 1, 3) #B,N*H*W,2*nhead,c//nhead

        # Flops, params = profile(self.fc, inputs=(f_all,))
        # sumflops += Flops
        f_conv = self.fc(f_all).permute(0, 3, 1, 2).reshape(B, 9 * C1 // self.num_heads, nimg, H,
                                                              W)  # B, 9*C//nhead, H, W
        # group conovlution
        out_conv = []
        for i in range(nimg):
            # Flops, params = profile(self.dep_conv, inputs=(f_conv[:,:,i,:,:].contiguous(),))
            # sumflops += Flops

            out_conv.append(self.dep_conv(f_conv[:,:,i,:,:].contiguous()))

        if nimg == 1:
            out_conv = out_conv[0].view(B, 27*C1//self.num_heads, nimg, H, W)
        else:
            temp = out_conv[0]
            for i in range(nimg-1):
                temp = torch.cat((temp,out_conv[i+1]),dim=2)
            # out_conv = temp.permute(2,3,0,1).contiguous().view(nimg*H*W,B,C)
            out_conv = temp.view(B, 27*C1//self.num_heads, nimg, H, W)
        # Flops, params = profile(self.STConv, inputs=(out_conv,))
        # sumflops += Flops

        out_conv = self.STConv(out_conv).permute(2,3,4,0,1).contiguous().view(nimg*H*W,B,C)

        xlist = []
        qkv = qkv.view(B,nimg,H*W,C1*2) # nW*B, window_size*window_size, C  768,3,512  3*hw,3,2,256
        for i in range(nimg):
            xlist.append(qkv[:,i,:,:])

        qkv = xlist
        # C = C // 2

        for i in range(nimg):
            # qkv = qkv.reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            temp = qkv[i].reshape(B, H, W, 2, C1).permute(3, 0, 1, 2, 4)
            q, k = temp[0], temp[1]  # 3,8,256,32
            k = [k]
            ref_point = generate_ref_points(width=H,
                                            height=W)
            ref_point = ref_point.type_as(q)
            # H, W, 2 -> B, H, W, 2
            ref_point = ref_point.unsqueeze(0).repeat(B, 1, 1, 1)
            qkv[i] = self.DefAttn(q, k, ref_point)[0]

        if nimg == 1:
            x = qkv[0].permute(0,3,1,2).contiguous().reshape(B,C//8,H,W)
            x = self.large(x)
            x = x.permute(2, 3, 0, 1).contiguous().view(nimg * H * W, B, C)
        else:
            x = qkv[0].permute(0,3,1,2)
            for i in range(nimg-1):
                x = torch.cat((x,qkv[i].permute(0,3,1,2)),dim=2)
            x = x.contiguous().view(B,C//8,nimg,H,W)

            # Flops, params = profile(self.DefAttn3D, inputs=(x,))
            # sumflops += Flops

            x = self.DefAttn3D(x)
            x = x.permute(2,3,4,0,1).contiguous().view(nimg*H*W,B,C)
            # x = qkv[0].permute(1,2,0,3).contiguous().view(H*W,B,C)
            # for i in range(nimg-1):
            #     x = torch.cat((x,qkv[i].permute(1,2,0,3).contiguous().view(H*W,B,C)),dim=0)


        x = self.rate1 * x  + self.rate2 * out_conv
        x = self.proj_drop(x)
        # x = x.view(B, H * W, C)
        return x


def getModelParam(model):
    # 

    print(model)
    print('*********************************************************************')

    for param_tensor in model.state_dict(): 
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    print('*********************************************************************')

    from prettytable import PrettyTable

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    count_parameters(model)
    print('*********************************************************************')

    for para in model.named_parameters():  
       
        print(para[0], '\t', para[1].size())

    print('*********************************************************************')

    
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print('*********************************************************************')
    params = list(model.parameters())
    
    print(params.__len__())


if __name__ == "__main__":
   

    model = STDefACMixAttn(512,8,22,22).cuda()

    
