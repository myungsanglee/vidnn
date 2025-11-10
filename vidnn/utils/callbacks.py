import torch
from pytorch_lightning.callbacks import LearningRateMonitor

class Float32LrMonitor(LearningRateMonitor):
    def _extract_stats(self, trainer, interval):
        # Re-implementation of _extract_stats to avoid the float64 bug on MPS.
        # We don't call super()._extract_stats() as it contains the faulty code.
        
        stats = {}
        for opt in trainer.optimizers:
            for i, param_group in enumerate(opt.param_groups):
                # Simplified naming.
                name = f"lr-pg{i+1}"
                stats[name] = param_group['lr']

        if stats:
            metrics = {
                k: torch.tensor(v, dtype=torch.float32, device=trainer.strategy.root_device)
                for k, v in stats.items()
            }
            trainer.callback_metrics.update(metrics)
        
        return stats