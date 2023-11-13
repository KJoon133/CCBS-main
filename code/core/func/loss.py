
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.insert(0, '/home/railab/Workspace/CCBS/code/')

from core.utils.visualize_heatmap import visualize_heatmap

torch.set_printoptions(profile='default')
torch.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(linewidth=sys.maxsize)
torch.set_printoptions(precision=2)

np.set_printoptions(precision=2)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

def compute_gaussian(alpha, beta, x, y, sigma):
    # print(alpha, beta)
    return np.round(np.exp(-1 * (((x-alpha)**2 + (y-beta)**2) / (2*(sigma**2)))), 3)

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        target_weight = target_weight.reshape((batch_size, num_joints, 1))
        # print(target_weight)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                # mse_loss = self.criterion(
                #     heatmap_pred.mul(target_weight[:, idx]),
                #     heatmap_gt.mul(target_weight[:, idx])
                # )
                # print(mse_loss)
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(WeightedBinaryCrossEntropy, self).__init__()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta
    
    def forward(self, y_pred, y_target):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1 / (1 + (self.theta / self.epsilon)**(self.alpha - y_target)))*(self.alpha - y_target)*((self.theta/self.epsilon)**(self.alpha-y_target-1))*(1/self.epsilon)
        C = ((self.theta * A) - (self.omega*torch.log(1+(self.theta/self.epsilon)**(self.alpha-y_target))))

        case1_ind = torch.abs(y_target-y_pred) < self.theta
        case2_ind = torch.abs(y_target-y_pred) >= self.theta

        lossMat[case1_ind] = self.omega*torch.log(1+torch.abs((y_target[case1_ind]-y_pred[case1_ind])/self.epsilon)**(self.alpha-y_target[case1_ind]))
        lossMat[case2_ind] = A[case2_ind]*torch.abs(y_target[case2_ind]-y_pred[case2_ind]) - C[case2_ind]

        return lossMat

def main():
    loss = AdaptiveWingLoss()

    pred_0 = torch.zeros((3, 32, 32))
    pred = torch.rand((3, 32, 32))
    target = torch.zeros((3, 32, 32))

    # print(target[0])

    for x in range(target[0].shape[0]):
        for y in range(target[0].shape[0]):
            target[0][x][y] = compute_gaussian(alpha=15, beta=15, x=x, y=y, sigma=2.0)
            target[2][x][y] = compute_gaussian(alpha=15, beta=15, x=x, y=y, sigma=2.0)

    print("TARGET")
    # print(target)
    visualize_heatmap(target, "Target")

    pred_0 = target
    print("PRED")
    # print(pred_0)
    visualize_heatmap(pred, "Prediction")

    mask_map = torch.where(target > 0.2, 1, target)
    mask_map = torch.where(mask_map <= 0.2, 0, mask_map)

    print("MASK_MAP")
    # print(mask_map)
    visualize_heatmap(mask_map, "Mask map")

    loss_0 = loss(pred, target)
    print("LOSS MAT")
    # print(loss_0)
    visualize_heatmap(loss_0, "Loss mat")

    mask_loss = torch.mul(loss_0, 10*mask_map+1)
    print("MASKED LOSS")
    # print(mask_loss)
    visualize_heatmap(mask_loss, "Mask * Loss")

    # print(loss_)


if __name__ == "__main__":
    main()