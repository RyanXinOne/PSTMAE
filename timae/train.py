from torch import utils
import lightning.pytorch as pl
from torchinfo import summary
from nn.pl_model import LitTiMAE
from data.dataset import ShallowWaterDataset


def main():
    autoencoder = LitTiMAE(12288)
    summary(autoencoder.model)

    train_dataset = ShallowWaterDataset(path='shallow_water/train')
    eval_dataset = ShallowWaterDataset(path='shallow_water/eval')

    train_loader = utils.data.DataLoader(train_dataset, 32, num_workers=0, shuffle=True)
    eval_loader = utils.data.DataLoader(eval_dataset, 32, num_workers=0)

    trainer = pl.Trainer(
        max_epochs=10,
        logger=False
    )
    trainer.fit(autoencoder, train_loader, eval_loader)


if __name__ == "__main__":
    main()
