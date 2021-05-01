from ..builder import DETECTORS
from .two_stage import TwoStageDetector

@DETECTORS.register_module()
class FsodRCNN(TwoStageDetector):
    """Implementation of `FSOD <https://arxiv.org/pdf/1908.01998v1.pdf>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FsodRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
