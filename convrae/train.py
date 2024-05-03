import lightning.pytorch as pl
from torchinfo import summary
from convrae.pl_model import LitConvRAE
from data.dataset import ShallowWaterDataset


def main():
    model = LitConvRAE()
    summary(model.model)

    train_dataset = ShallowWaterDataset(path='shallow_water/train', flatten=False)
    eval_dataset = ShallowWaterDataset(path='shallow_water/eval', flatten=False)

    trainer = pl.Trainer(
        max_epochs=10,
        logger=False
    )
    trainer.fit(model, train_dataset, eval_dataset)


if __name__ == "__main__":
    main()
