import os

import numpy as np
import torch


class WandbLogger:
    def __init__(self, args):
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "wandb is not installed. Please install it with `pip install wandb`."
            ) from error

        self.wandb = wandb
        output_dir = args.output_dir if args.output_dir else None
        run_name = args.wandb_run_name
        if not run_name and args.output_dir:
            run_name = os.path.basename(os.path.abspath(args.output_dir))

        self.run = self.wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name if run_name else None,
            config=vars(args),
            mode=args.wandb_mode,
            dir=output_dir,
        )
        self.log_dir = self.run.dir if self.run is not None else output_dir
        self._defined_metrics = set()
        if self.run is not None:
            self.run.define_metric("global_step")

    def add_scalar(self, key, value, step):
        if key not in self._defined_metrics:
            self.run.define_metric(key, step_metric="global_step")
            self._defined_metrics.add(key)
        self.wandb.log({"global_step": int(step), key: float(value)})

    def add_images(self, key, images, step, captions=None):
        if isinstance(images, torch.Tensor):
            image_batch = images.detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
        else:
            image_batch = np.asarray(images)

        wb_images = []
        for idx, image in enumerate(image_batch):
            image_uint8 = np.round(np.clip(image * 255.0, 0, 255)).astype(np.uint8)
            caption = captions[idx] if captions is not None and idx < len(captions) else None
            wb_images.append(self.wandb.Image(image_uint8, caption=caption))
        if key not in self._defined_metrics:
            self.run.define_metric(key, step_metric="global_step")
            self._defined_metrics.add(key)
        self.wandb.log({"global_step": int(step), key: wb_images})

    def flush(self):
        # W&B handles flushing internally.
        return

    def finish(self):
        self.wandb.finish()
