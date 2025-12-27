import torch
import torch.nn as nn
import torch.nn.functional as F
def hard_negative_cross_entropy(pred_conf, target_conf, neg_pos_ratio=3):
    """
    Args:
        pred_conf: (tensor) shape [batch_size, num_priors, num_classes]
        target_conf: (tensor) shape [batch_size, num_priors]
        neg_pos_ratio: (float)
    Returns:
        loss: (tensor) shape [1]
    """
    # make sure target_conf is long
    target_conf = target_conf.long()
    pos_mask = (target_conf>0)
    pos_num = pos_mask.sum(dim=1, keepdim=True)
    batched_pred_conf = pred_conf.view(-1, pred_conf.size(-1))
    batched_target_conf = target_conf.view(-1)
    
    conf_loss = F.cross_entropy(batched_pred_conf, batched_target_conf, reduction='none')
    conf_loss = conf_loss.view(pred_conf.size(0), -1)
    
    #ignore positive losses for doing hard negative mining
    conf_loss[pos_mask] = 0
    
    #sort loss
    _, loss_idx = conf_loss.sort(1, descending=True)
    _, rank_idx = loss_idx.sort(1)
    #neg num
    neg_num = torch.clamp(neg_pos_ratio * pos_num, max=pos_mask.size(1)-1)
    
    neg_mask = rank_idx < neg_num
    prior_mask = pos_mask | neg_mask
    hnm_conf_pred = pred_conf[prior_mask]
    hnm_conf_target = target_conf[prior_mask]
    return F.cross_entropy(hnm_conf_pred, hnm_conf_target)


from .utils import matches
class CustomMultiBoxLoss(nn.Module):
    def __init__(self, threshold, priors, conf_weight = 1):
        super().__init__()
        self.threshold = threshold
        self.priors = priors
        self.conf_weight = conf_weight

    def encode(self, gt_list, matched_priors_boxes, variances=[0.1, 0.1, 0.2, 0.2]):
        """
        Args:
            gt_list: List of 1D tensors (absolute GT coords)
            matched_priors_boxes: 2D tensor (corresponding anchor priors)
        Returns:
            List of 1D tensors (encoded offsets)
        """
        device = matched_priors_boxes.device
        gt_list = [t.to(device) for t in gt_list]
        if len(gt_list) == 0:
            return []

        matched_gt = torch.stack(gt_list)

        g_cx = (matched_gt[:, 0] + matched_gt[:, 2]) / 2
        g_cy = (matched_gt[:, 1] + matched_gt[:, 3]) / 2
        g_w  = matched_gt[:, 2] - matched_gt[:, 0]
        g_h  = matched_gt[:, 3] - matched_gt[:, 1]

        p_cx = matched_priors_boxes[:, 0]
        p_cy = matched_priors_boxes[:, 1]
        p_w  = matched_priors_boxes[:, 2]
        p_h  = matched_priors_boxes[:, 3]

        # SSD Encoding Math
        enc_cx = (g_cx - p_cx) / (p_w * variances[0])
        enc_cy = (g_cy - p_cy) / (p_h * variances[1])
        enc_w  = torch.log(g_w / p_w + 1e-5) / variances[2]
        enc_h  = torch.log(g_h / p_h + 1e-5) / variances[3]

        return torch.stack([enc_cx, enc_cy, enc_w, enc_h], dim=1)

    def forward(self, preds, targets):
        """
            Args:
                preds: (tuple) includes loc, conf
                    loc: (tensor) shape [batch_size, num_priors, 4]
                    conf: (tensor) shape [batch_size, num_priors, num_classes] (num_classes include background as 0)

                targets: (list) shape [batch_size, num_object_i, 5]
        """
        device = preds[0].device
        pred_loc, pred_conf = preds

        batch_size = pred_loc.size(0)
        num_priors = pred_loc.size(1)

        # loss matching
        matched_priors, matched_gt_boxes = [], []
        # create a tensor that has shape [batch_size, num_priors]
        target_conf = torch.zeros((batch_size, num_priors))
        for i in range(batch_size):
            matched_gt_boxes_one = []
            # calculate jaccard scores
            truth_indexes = matches(self.threshold, torch.as_tensor(targets[i], device=device), self.priors)
            truth_indexes = truth_indexes.to(device)
            truth_mask = truth_indexes != -1
            match_priors_one = pred_loc[i][truth_mask] # match those doesn't predict bg
            # for truth_id in truth_indexes:
            #     if truth_id != -1:
            #         matched_gt_boxes_one.append(torch.as_tensor(targets[i][truth_id], device=device)[:-1])
            # filter out -1
            filtered_indexes = truth_indexes[truth_indexes!=-1]
            # make gt tensor
            one_image_target = torch.as_tensor(targets[i], device=device)[:, :-1]
            
            matched_gt_boxes_one = one_image_target[filtered_indexes]
            matched_priors.append(match_priors_one)
            matched_gt_boxes.append(self.encode(matched_gt_boxes_one, self.priors[truth_indexes != -1]))

            # this could be vectorized for parallelizing
            # for j in range(num_priors):
            #     target_conf[i, j] = targets[i][truth_indexes[j]][-1]
            safe_mask = truth_indexes.clone()
            safe_mask[safe_mask == -1] = 0
            target_batch_i = torch.where(
                truth_indexes==-1,
                torch.tensor(0, device=device),
                torch.as_tensor(targets[i], device=device)[:, -1][safe_mask]
            )
            target_conf[i, :] = target_batch_i
        matched_priors = torch.cat(matched_priors, dim=0).to(device)
        matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0).to(device)

        loc_loss = F.smooth_l1_loss(matched_priors, matched_gt_boxes, reduction='sum') / batch_size

        # conf loss
        target_conf = target_conf.to(device)
        conf_loss = hard_negative_cross_entropy(pred_conf, target_conf) / batch_size * self.conf_weight


        return loc_loss, conf_loss

