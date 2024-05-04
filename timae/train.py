from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchinfo import summary
from timae.pl_model import LitTiMAE
from data.dataset import ShallowWaterDataset


def main():
    model = LitTiMAE(input_dim=3*64*64)
    summary(model.model)

    train_dataset = ShallowWaterDataset(path='shallow_water/train', flatten=True)
    val_dataset = ShallowWaterDataset(path='shallow_water/val', flatten=True)

    train_loader = DataLoader(train_dataset, 32, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, 32, num_workers=0)

    trainer = pl.Trainer(
        max_epochs=10,
        logger=False
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
