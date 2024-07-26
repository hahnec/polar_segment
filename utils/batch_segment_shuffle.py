import random
import torch
import warnings

class BatchSegmentShuffler(object):

    def __init__(self, method: str = 'mask', *args, **kwargs):
        self.method = method

    @staticmethod
    def batch_shuffle_crop(img, label=None, batch_dim=0):

        w, h = img.shape[-2:][::-1]
        i = random.randint(h//4, h//4*3)
        j = random.randint(w//4, w//4*3)
        
        b = img.shape[batch_dim]
        if b == 1: warnings.warn("Batch size is 1 such that batch shuffling has no effect")
        idcs_a = torch.randperm(b, device=img.device)
        idcs_b = torch.randperm(b, device=img.device)
        idcs_c = torch.randperm(b, device=img.device)
        idcs_d = torch.randperm(b, device=img.device)
        img[..., i:, j:] = img[..., i:, j:].index_select(batch_dim, idcs_a)
        img[..., i:, :j] = img[..., i:, :j].index_select(batch_dim, idcs_b)
        img[..., :i, j:] = img[..., :i, j:].index_select(batch_dim, idcs_c)
        img[..., :i, :j] = img[..., :i, :j].index_select(batch_dim, idcs_d)
        if label is not None: 
            label[..., i:, j:] = label[..., i:, j:].index_select(batch_dim, idcs_a)
            label[..., i:, :j] = label[..., i:, :j].index_select(batch_dim, idcs_b)
            label[..., :i, j:] = label[..., :i, j:].index_select(batch_dim, idcs_c)
            label[..., :i, :j] = label[..., :i, :j].index_select(batch_dim, idcs_d)
            return img, label
        return img
    
    @staticmethod
    def batch_shuffle_mask(img, label, batch_dim=0, class_dim=1, class_ch=0, shuffle_mask_invert=False):

        d = img.device
        b = img.shape[batch_dim]
        if b == 1: warnings.warn("Batch size is 1 such that batch shuffling has no effect")

        # random mask selection from batch
        i = random.randint(0, b-1)
        mask = label.index_select(batch_dim, torch.tensor((i), device=d)).index_select(class_dim, torch.tensor((class_ch), device=d)) > 0
        ishape = [img.shape[i] if img.shape[i] != mask.shape[i] else 1 for i in range(len(img.shape))]
        lshape = [label.shape[i] if label.shape[i] != mask.shape[i] else 1 for i in range(len(label.shape))]
        imask = mask.repeat(ishape)
        lmask = mask.repeat(lshape)
        #imask = mask.expand_as(torch.tensor(img.shape))
        #lmask = mask.expand_as(torch.tensor(label.shape))

        # batch-wise shuffling based on mask
        idcs_a = torch.randperm(b, device=d)
        img[imask] = img.index_select(batch_dim, idcs_a)[imask]
        label[lmask] = label.index_select(batch_dim, idcs_a)[lmask]
        if shuffle_mask_invert:
            idcs_b = torch.randperm(b, device=d)
            img[~imask] = img.index_select(batch_dim, idcs_b)[~imask]
            label[~lmask] = label.index_select(batch_dim, idcs_b)[~lmask]

        return img, label

    def __call__(self, img, label=None, *args, **kwargs):
        if self.method == 'mask':
            return self.batch_shuffle_mask(img, label)
        elif self.method == 'crop':
            return self.batch_shuffle_crop(img, label)
        else:
            raise NotImplementedError('unrecognized batch-shuffling method')
