from torch.optim.lr_scheduler import (
    SequentialLR,
    ConstantLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)


class Scheduler:
    def __init__(self, cfg, opt):
        self.cfg = cfg
        self.opt = opt

        if not cfg.training.lr_scheduling:
            self._scheduler = ConstantLR(
                opt, factor=1.0, total_iters=cfg.training.epochs
            )
            self._mode = "flat"

        else:
            self._scheduler = ReduceLROnPlateau(
                opt,
                mode="min",
                factor=0.5,
                patience=cfg.training.lr_patience,
                threshold=cfg.training.min_delta,
                threshold_mode="abs",
            )
            self._mode = "plateau"

    def step(self, metric=None):
        if self._mode == "plateau":
            if metric is None:
                raise ValueError("Plateau scheduler requires a metric")
            self._scheduler.step(metric)
        else:
            self._scheduler.step()

    def get_lr(self):
        return self.opt.param_groups[0]["lr"]
