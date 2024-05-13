from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchinfo import summary
from timae.pl_model import LitTiMAE
from data.dataset import ShallowWaterDataset


def main():
    model = LitTiMAE()
    summary(model.model)

    train_dataset = ShallowWaterDataset(split='train', flatten=True)
    val_dataset = ShallowWaterDataset(split='val', flatten=True)
    test_dataset = ShallowWaterDataset(split='test', flatten=True)

    train_loader = DataLoader(train_dataset, 32, shuffle=True)
    val_loader = DataLoader(val_dataset, 32)
    test_loader = DataLoader(test_dataset, 32)

    trainer = pl.Trainer(
        max_epochs=20,
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=False,
        default_root_dir='logs/timae',
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    trainer.predict(model, test_loader)


if __name__ == "__main__":
    main()
