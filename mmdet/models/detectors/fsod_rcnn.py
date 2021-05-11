import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import pandas as pd
import pdb
@DETECTORS.register_module()
class FsodRCNN(BaseDetector):
    """Base class for two-stage detectors of FSOD

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(FsodRCNN, self).__init__(init_cfg)
        backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      support_imgs = None,
                      support_bboxes = None,
                      support_labels = None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

            support_imgs (None | List): support img

            support_bboxes (None | List): support bboxes

            support_labels (None | List): supprt labels

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        
        # squeeze unused dimensions in bboxs and labels
        # originally these dimensions are for bypass some configs in mmdet
        support_bboxes = torch.squeeze(support_bboxes)
        support_labels = torch.squeeze(support_labels)

        # extract support features
        B, N, C, H, W  = support_imgs.shape
        support_imgs = support_imgs.reshape(B*N, C, H, W)
        support_features = self.extract_feat(support_imgs)


        assert self.with_rpn # otherwise it's not fsod :)
        losses_rpn_cls, losses_rpn_bbox, losses_cls, losses_bbox, acces = [], [], [], [], []
         
        # RPN forward and loss
        for i in range(B):

            losses_perbatch = dict()
            x_i = tuple(x_[i].unsqueeze(0) for x_ in x)

            assert self.with_rpn, 'we need rpn with feature aggregation'
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
            # cls_dim = torch.zeros_like(support_bboxes[i])
            # _support_bboxes = torch.cat([cls_dim, support_bboxes[i]], axis = -1)[:,3:].float().contiguous()
            # self.roi_head.bbox_roi_extractor(support_features,_support_bboxes)
            
            # # extract roi features
            batch_size = support_bboxes.shape[1]
            support_bbox_features = []
            for support_features_ in support_features:
                for support_feature, support_bbox in zip(support_features_[i * batch_size: (i + 1) * batch_size],support_bboxes[i]):
                # extract roi features in res5
                    support_bbox = torch.cat([torch.zeros_like(support_bbox[:1]), support_bbox]).float().contiguous()
                    support_bbox_features.append(self.roi_head.bbox_roi_extractor([support_feature.unsqueeze(0)],support_bbox.unsqueeze(0)))

            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x_i,
                [img_metas[i]], 
                [gt_bboxes[i]],
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            #proposal_list[0].shape = torch.Size([2000, 5])
            losses_perbatch.update(rpn_losses)

            roi_losses = self.roi_head.forward_train(x_i, [img_metas[i]], proposal_list,
                                                    [gt_bboxes[i]], [gt_labels[i]],
                                                    support_bbox_features, #support_features,
                                                    gt_bboxes_ignore, gt_masks, 
                                                    **kwargs)
            losses_perbatch.update(roi_losses)
            # losses_perbatch.keys() = dict_keys(['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'acc', 'loss_bbox'])

            # update losses
            losses_rpn_cls.append(torch.stack(losses_perbatch['loss_rpn_cls']).sum())
            losses_rpn_bbox.append(torch.stack(losses_perbatch['loss_rpn_bbox']).sum())
            losses_cls.append(losses_perbatch['loss_cls'])
            acces.append(losses_perbatch['acc'])
            losses_bbox.append(losses_perbatch['loss_bbox'])

        # sum up losses
        losses = dict()
        losses['loss_rpn_cls'] = torch.stack(losses_rpn_cls).mean()
        losses['loss_rpn_bbox'] = torch.stack(losses_rpn_bbox).mean()
        losses['loss_cls'] = torch.stack(losses_cls).mean()
        losses['acc'] = torch.stack(acces).mean()
        losses['loss_bbox'] = torch.stack(losses_bbox).mean()

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # print('start simple testing')
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        x = self.extract_feat(img)
        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, support_bbox_features = [], rescale=rescale)

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        # raise NotImplementedError
        # print('start async simple testing')
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # raise NotImplementedError
        # print('start aug testing')
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
