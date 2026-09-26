"""Local maximum mean discrepancy used by the original DSAN baseline."""

import torch
import torch.nn.functional as F

from .mmd import guassian_kernel


class LMMD_loss:
    def __init__(self, class_num=31, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        self.class_num = class_num
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def get_loss(self, source, target, s_label, t_logits):
        if source.size(0) != target.size(0):
            n = min(source.size(0), target.size(0))
            source, target = source[:n], target[:n]
            s_label = s_label[:n]
            t_logits = t_logits[:n]
        s_weight = self.cal_weight(s_label, self.class_num).to(source.device)
        t_weight = self.cal_weight(t_logits, self.class_num).to(source.device)
        kernels = guassian_kernel(source, target, self.kernel_mul,
                                   self.kernel_num, self.fix_sigma)
        n = source.size(0)
        loss = source.new_tensor(0.)
        for i in range(self.class_num):
            for j in range(self.class_num):
                weight = s_weight[:, i:i + 1].mm(s_weight[:, j:j + 1].t())
                weight += t_weight[:, i:i + 1].mm(t_weight[:, j:j + 1].t())
                weight -= s_weight[:, i:i + 1].mm(t_weight[:, j:j + 1].t())
                weight -= t_weight[:, i:i + 1].mm(s_weight[:, j:j + 1].t())
                if i == j:
                    loss = loss + (weight * kernels[:n, :n]).sum()
                    loss = loss + (weight * kernels[n:, n:]).sum()
                    loss = loss - (weight * kernels[:n, n:]).sum()
                    loss = loss - (weight * kernels[n:, :n]).sum()
        return loss / max(n, 1)

    @staticmethod
    def cal_weight(labels, class_num):
        if labels.ndim == 1:
            one_hot = F.one_hot(labels.long(), class_num).float()
        else:
            one_hot = labels.float()
        counts = one_hot.sum(0, keepdim=True).clamp_min(1.)
        return one_hot / counts
