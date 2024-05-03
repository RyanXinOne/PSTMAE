from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchinfo import summary
from timae.nn.pl_model import LitTiMAE
from data.dataset import ShallowWaterDataset


def main():
    model = LitTiMAE(12288)
    summary(model.model)

    train_dataset = ShallowWaterDataset(path='shallow_water/train', flatten=True)
    eval_dataset = ShallowWaterDataset(path='shallow_water/eval', flatten=True)

    train_loader = DataLoader(train_dataset, 32, num_workers=0, shuffle=True)
    eval_loader = DataLoader(eval_dataset, 32, num_workers=0)

    trainer = pl.Trainer(
        max_epochs=10,
        logger=False
    )
    trainer.fit(model, train_loader, eval_loader)


if __name__ == "__main__":
    main()
