import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss

class clDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.):
        super(clDiceLoss, self).__init__()
        self.smooth = smooth

    def soft_cldice_loss(self, pred, target, target_skeleton=None):
        '''
        inputs shape  (batch, channel, height, width).
        calculate clDice loss
        Because pred and target at moment of loss calculation will be a torch tensors
        it is preferable to calculate target_skeleton on the step of batch forming,
        when it will be in numpy array format by means of opencv
        '''
        cl_pred = self.soft_skeletonize(pred)
        if target_skeleton is None:
            target_skeleton = self.soft_skeletonize(target)
        iflat = self.norm_intersection(cl_pred, target)
        tflat = self.norm_intersection(target_skeleton, pred)
        intersection = (iflat * tflat).sum()
        return 1. - (2. * intersection) / (iflat + tflat).sum()

    def dice_loss(self, pred, target):
        '''
        inputs shape  (batch, channel, height, width).
        calculate dice loss per batch and channel of sample.
        E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
        '''
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss

    def soft_skeletonize(self, x, thresh_width=10):
        '''
        Differenciable aproximation of morphological skelitonization operaton
        thresh_width - maximal expected width of vessel
        '''
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
        return x

    def norm_intersection(self, center_line, vessel):
        '''
        inputs shape  (batch, channel, height, width)
        intersection formalized by first ares
        x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
        '''
        smooth = 1.
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)
        intersection = (clf * vf).sum(-1)
        return (intersection + smooth) / (clf.sum(-1) + smooth)

    def forward(self, pred, target):
        return 0.8 * self.dice_loss(pred, target) + 0.2 * self.soft_cldice_loss(pred, target)