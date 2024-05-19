from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchinfo import summary
from models.convlstm.pl_model import LitConvLSTM
from data.dataset import ShallowWaterDataset


def main():
    model = LitConvLSTM()
    summary(model.model)

    train_dataset = ShallowWaterDataset(split='train', flatten=False)
    val_dataset = ShallowWaterDataset(split='val', flatten=False)
    test_dataset = ShallowWaterDataset(split='test', flatten=False)

    train_loader = DataLoader(train_dataset, 32, shuffle=True)
    val_loader = DataLoader(val_dataset, 32)
    test_loader = DataLoader(test_dataset, 32, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=20,
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=False,
        default_root_dir='logs/convlstm',
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    trainer.predict(model, test_loader)


if __name__ == "__main__":
    main()
