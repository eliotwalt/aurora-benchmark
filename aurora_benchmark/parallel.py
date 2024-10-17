from typing import Generator
import dataclasses
from functools import partial
from datetime import timedelta
import torch

from aurora import Aurora, Batch

from aurora_benchmark.data import aurora_batch_scatter, aurora_batch_gather

class AuroraBatchDataParallel(torch.nn.DataParallel):    
    def scatter(self, inputs: tuple, kwargs: dict|None, device_ids: list[int]):
        return aurora_batch_scatter(inputs[0], kwargs, device_ids)
    
    def gather(self, outputs, output_device):
        return aurora_batch_gather(outputs, output_device)
    
def rollout(model: Aurora, batch: Batch, steps: int) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.

    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    
    # get the patch size
    if isinstance(model, torch.nn.DataParallel):
        patch_size = model.module.patch_size
    else:
        patch_size = model.patch_size
        
    # Ensure consistency of the device and the data type throughout the batch.
    device = next(iter(batch.surf_vars.values())).device
    batch = batch.type(torch.float32)
    batch = batch.crop(patch_size=patch_size)
    batch = batch.to(device)

    for _ in range(steps):        
        pred = model.forward(batch)        
        yield pred

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )

class ParallelAurora(Aurora):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`Batch`): Batch to run the model on.

        Raises:
            ValueError: If no metric is provided.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        
        # Ensure everything is on the same device and in the right format.
        device = next(iter(batch.surf_vars.values())).device
        batch = batch.type(torch.float32)
        batch = batch.normalise()
        batch = batch.crop(patch_size=self.encoder.patch_size)
        batch = batch.to(device)
        
        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        x = self.encoder(
            batch,
            lead_time=timedelta(hours=6),
        )
        x = self.backbone(
            x,
            lead_time=timedelta(hours=6),
            patch_res=patch_res,
            rollout_step=batch.metadata.rollout_step,
        )
        pred = self.decoder(
            x,
            batch,
            lead_time=timedelta(hours=6),
            patch_res=patch_res,
        )

        # Remove batch and history dimension from static variables.
        B, T = next(iter(batch.surf_vars.values()))[0]
        pred = dataclasses.replace(
            pred,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )

        # Insert history dimension in prediction. The time should already be right.
        pred = dataclasses.replace(
            pred,
            surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
            atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
        )

        pred = pred.unnormalise()

        return pred
    
ParallelAuroraSmall = partial(
    ParallelAurora,
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
    use_lora=False,
)