import torch


class YoloCriterion(torch.nn.Module):
    def __init__(self, args, model):
        super().__init__()

        device = next(model.parameters()).device  # get model device

        m = model.head  # Head() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device

        # task aligned assigner
        self.top_k = 10
        self.alpha = 0.5
        self.beta = 6.0
        self.eps = 1e-9

        self.bs = 1
        self.num_max_boxes = 0

        # print(args)
        #Loss weights
        self.classCost = args.get('classCost')
        self.bboxCost = args.get('bboxCost')
        self.dlfCost = args.get('dlfCost')

        # DFL Loss params
        self.dfl_ch = m.dfl.ch
        self.project = torch.arange(self.dfl_ch, dtype=torch.float, device=device)

    def forward(self, x: dict[str, torch.Tensor], y: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return self.compute_loss(x,y)
    
    def compute_loss(self, x: dict[str, torch.Tensor], y: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        
        # MARK: - compute statistics
        # with torch.no_grad():

        #     print(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        #     ious = boxIoU(pred_bboxes[fg_mask], target_bboxes[fg_mask])[0]
        #     iou = torch.diag(ious)
        #     print(len(iou), len(fg_mask))
        #     iou_th = [50, 75, 95]
        #     map_th = []
        #     ap = []
        #     for threshold in range(50, 100, 5):
        #         ap_th = ((iou >= threshold / 100) * fg_mask).sum() / (len(ious) + 1e-6)
        #         ap.append(ap_th)
        #         if threshold in iou_th:
        #             map_th.append(ap_th)

        #     ap = torch.mean(torch.stack(ap))
        return {'classification loss': None, #loss_cls,
                'bbox loss': None, #loss_box,
                'DFL loss': None, #loss_dfl,
                'mAP': None, #ap,
                'mAP_50': None, #map_th[0],
                'mAP_75': None, #map_th[1],
                'mAP_95': None, #map_th[2]
                }