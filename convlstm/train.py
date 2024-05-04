from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchinfo import summary
from convlstm.pl_model import LitConvLSTM
from data.dataset import ShallowWaterDataset


def main():
    model = LitConvLSTM()
    summary(model.model)

    train_dataset = ShallowWaterDataset(path='shallow_water/train', flatten=False)
    eval_dataset = ShallowWaterDataset(path='shallow_water/eval', flatten=False)

    train_loader = DataLoader(train_dataset, 8, num_workers=0, shuffle=True)
    eval_loader = DataLoader(eval_dataset, 8, num_workers=0)

    trainer = pl.Trainer(
        max_epochs=10,
        logger=False
    )
    trainer.fit(model, train_loader, eval_loader)


if __name__ == "__main__":
    main()
