import torch
from torch.utils.data import DataLoader, random_split, default_collate
import lightning.pytorch as pl
from torchinfo import summary
from models.timae.pl_model import LitTiMAE
from data.dataset import ShallowWaterDataset
from data.utils import generate_random_mask


def main():
    dataset = ShallowWaterDataset(dilation=3, masking_steps=[1, 2, 3, 4, 5, 6])
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

    def collate_fn(batch):
        steps = int(batch[0][2].sum())
        new_batch = []
        # ensure same masking steps in a batch
        for x, y, mask, config in batch:
            mask = generate_random_mask(x.size(0), steps)
            mask = torch.from_numpy(mask).float()
            new_batch.append((x, y, mask, config))
        return default_collate(new_batch)

    train_loader = DataLoader(train_dataset, 32, num_workers=4, persistent_workers=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, 32, num_workers=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, 32, num_workers=2, collate_fn=collate_fn)

    model = LitTiMAE(dataset)
    summary(model.model)

    trainer = pl.Trainer(
        max_epochs=50,
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir='logs/timae',
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    trainer.predict(model, test_loader)


if __name__ == "__main__":
    main()
