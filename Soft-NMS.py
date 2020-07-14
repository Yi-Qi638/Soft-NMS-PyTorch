import torch


def Soft_NMS(dets, box_scores, iou_thresh=0, sigma=0.5, thresh=0.001):

    """
    reference https://github.com/DocF/Soft-NMS.git
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[x1, y1, x2, y2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        iou_thresh   iou_thresh if use method2 else 0
        thresh:      score thresh       
    # Return
        the sorted index of the selected boxes
    """
    N = dets.shape[0]  # the number of boxes

    # Indexes concatenate boxes with the last column
    indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1) 
    dets = torch.cat((dets, indexes), dim=1)

    # Sort the scores of the boxes from largest to smallest
    box_scores, conf_sort_index = torch.sort(box_scores, descending=True)
    dets = dets[conf_sort_index]

    for i in range(N):

        pos=i+1

        #iou calculate
        ious = box_iou(dets[i][0:4].view(-1,4), dets[pos:,:4])
        

        # Gaussian decay 
        # method1
        box_scores[pos:] = torch.exp(-(ious * ious) / sigma) * box_scores[pos:]

        # method2
        # zero = torch.zeros_like(ious)
        # ious  = torch.where(ious < iou_thresh, zero , ious)
        #box_scores[pos:] = torch.exp(-(ious * ious) / sigma) * box_scores[pos:]

        box_scores[pos:] = box_scores[pos:]
        box_scores[pos:], arg_sort = torch.sort(box_scores[pos:], descending=True)

        a=dets[pos:]
        
        dets[pos:] = a[arg_sort]

     # select the boxes and keep the corresponding indexes
    keep = dets[:,4][box_scores>thresh].long()

    return keep





def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)