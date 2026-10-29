import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert

class HungarianMatcher(nn.Module):
    """
    Solves the bipartite matching problem between predicted and ground truth boxes.
    """
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["logits"].shape[:2]

        # Flatten predictions for matching
        out_prob = outputs["logits"].flatten(0, 1).sigmoid() 
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Flatten ground truth
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if len(tgt_ids) == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) for _ in range(batch_size)]

        # 1. Compute Class Cost (Focal loss approximation)
        alpha, gamma = 0.25, 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # 2. Compute L1 Bounding Box Cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 3. Compute GIoU Cost (Convert from cx,cy,w,h to x1,y1,x2,y2 for torchvision)
        out_bbox_xyxy = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        tgt_bbox_xyxy = box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)

        # Final Cost Matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()

        # Split sizes to handle varying number of targets per image
        sizes = [len(v["boxes"]) for v in targets]
        
        # Perform the Hungarian matching using SciPy
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class RTDetrLoss(nn.Module):
    def __init__(self, num_classes=68):
        super().__init__()
        self.matcher = HungarianMatcher()
        self.num_classes = num_classes

    def _compute_single_layer_loss(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        device = outputs["logits"].device

        src_logits = outputs["logits"]
        src_boxes = outputs["pred_boxes"]

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        
        # 1. Normalized Classification Loss
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=device)
        
        if len(batch_idx) > 0:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes[batch_idx, src_idx] = target_classes_o
            
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        
        prob = src_logits.sigmoid()
        alpha, gamma = 0.25, 2.0
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        
        alpha_t = alpha * target_classes_onehot + (1 - alpha) * (1 - target_classes_onehot)
        
        # Correctly normalized WITH the alpha_t multiplier preserved
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device).clamp_(min=1).item()
        loss_ce = (alpha_t * ce_loss * ((1 - p_t) ** gamma)).sum() / num_boxes

        if len(batch_idx) == 0:
            return {"loss_ce": loss_ce * 2.0, "loss_bbox": src_logits.sum() * 0, "loss_giou": src_logits.sum() * 0}

        # 2. Bounding Box L1 Loss
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        src_boxes_matched = src_boxes[batch_idx, src_idx]
        loss_bbox = F.l1_loss(src_boxes_matched, target_boxes, reduction="mean")

        # 3. GIoU Loss
        src_boxes_xyxy = box_convert(src_boxes_matched, in_fmt="cxcywh", out_fmt="xyxy")
        target_boxes_xyxy = box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes_xyxy, target_boxes_xyxy)).mean()

        return {
            "loss_ce": loss_ce * 2.0,
            "loss_bbox": loss_bbox * 5.0,
            "loss_giou": loss_giou * 2.0,
        }

    def forward(self, outputs, targets):
        # 1. Compute loss for the final decoder layer
        losses = self._compute_single_layer_loss(outputs, targets)
        
        # 2. Compute and add auxiliary losses for intermediate decoder layers
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_losses = self._compute_single_layer_loss(aux_outputs, targets)
                for k, v in aux_losses.items():
                    # Appending layer index to key ensures they are summed properly in the training loop
                    losses[f"{k}_aux_{i}"] = v
                    
        # 3. Compute and add loss for the encoder's two-stage proposals
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            enc_losses = self._compute_single_layer_loss(enc_outputs, targets)
            for k, v in enc_losses.items():
                losses[f"{k}_enc"] = v
                    
        return losses