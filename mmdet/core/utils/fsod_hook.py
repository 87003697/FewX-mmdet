from mmcv.runner import HOOKS, Hook
import pdb

@HOOKS.register_module()
class Hook_fsod(Hook):

    def __init__(self, a, b):
        pass

    def before_val_epoch(self, runner):
        pdb.set_trace()
        pass
