from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from torchinfo import summary
from models.tikan.pl_model import LitTiKAN
from data.dataset import ShallowWaterDataset


def main():
    dataset = ShallowWaterDataset(dilation=3)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.9, 0.05, 0.05])

    train_loader = DataLoader(train_dataset, 32, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, 32, num_workers=2)
    test_loader = DataLoader(test_dataset, 32, num_workers=2)

    model = LitTiKAN(dataset)
    summary(model.model)

    trainer = pl.Trainer(
        max_epochs=25,
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir='logs/tikan',
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    trainer.predict(model, test_loader)


if __name__ == "__main__":
    main()
