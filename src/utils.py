import torch
def point_form(boxes):
    """
    Convert prior_boxes to (xmin, ymin, xmax, ymax)
        args: 
            boxes: (tensor of shape (num_boxes, 4))
            image_size: Size of square image
        return:
        boxes (tensor of shape (num_boxes, 4)) 
    """
    center  = boxes[:, :2]
    align = boxes[:, 2:] / 2
    top_left = center - align
    bottom_right = center + align
    
    return torch.cat([top_left, bottom_right], dim=-1)

def jaccard(priors, truths):
    """
    calculate IoU of every default box with every gt box
        args:
            truths: tensor of shape (num_objects, 4)
            priors: tensor of shape (8732,        4)
    """
    # unsqueeze dim 0 of truths and dim 1 of priors to compare every box of priors with every box of truth
    # both are broadcasted (num_priors, num_objects, 4)
    device = priors.device
    truths = truths.to(device)
    num_priors = priors.size(0)
    num_objects = truths.size(0)
    truths = truths.unsqueeze(0).expand(num_priors, -1, -1)
    priors = priors.unsqueeze(1).expand(-1, num_objects, -1)
    x1y1 = torch.max(truths[..., :2], priors[..., :2])
    x2y2 = torch.min(truths[..., 2:], priors[..., 2:])
    inter_area = (x2y2 - x1y1).clamp(min=0).prod(dim=-1)
    truth_area = (truths[..., 2] - truths[..., 0]) * (truths[..., 3] - truths[..., 1])
    prior_area = (priors[..., 2] - priors[..., 0]) * (priors[..., 3] - priors[..., 1])
    eps = 1e-7
    return inter_area / (prior_area + truth_area - inter_area + eps)

def matches(threshold, truths, priors):
    """
        return a 1d Tensor, ith position is the index of gt boxes
        match with i-th prior box 
        Args:
            truths: (tensor) shape [num_objects, 5] (xmin, ymin, xmax, ymax, labels)
            priors: (tensor) shape [num_priors, 4] (cx, cy, h, w)
    """
    truths = truths[..., :-1]
    
    overlaps = jaccard(point_form(priors), truths)
    
    best_gt_scores, best_gt_indexes = overlaps.max(dim=1)
    
    best_prior_scores, best_prior_indexes = overlaps.max(dim=0)
    
    #this guarantees each gt box has atleast matches 1 prior box 
    for k in range(best_prior_indexes.size(0)):
        best_gt_scores[best_prior_indexes[k]] = 2.0
        best_gt_indexes[best_prior_indexes[k]] = k
    #take only those with scores above threshold value
    best_gt_indexes[best_gt_scores < threshold] = -1
    
    return best_gt_indexes
    
    