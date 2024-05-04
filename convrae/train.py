from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchinfo import summary
from convrae.pl_model import LitConvRAE
from data.dataset import ShallowWaterDataset


def main():
    model = LitConvRAE()
    summary(model.model)

    train_dataset = ShallowWaterDataset(path='shallow_water/train', flatten=False)
    val_dataset = ShallowWaterDataset(path='shallow_water/val', flatten=False)

    train_loader = DataLoader(train_dataset, 8, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, 8, num_workers=0)

    trainer = pl.Trainer(
        max_epochs=10,
        logger=False
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
