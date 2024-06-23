from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchinfo import summary
from models.convlstm.pl_model import LitConvLSTM
from data.dataset import ShallowWaterDataset


def main():
    model = LitConvLSTM()
    summary(model.model)

    train_dataset = ShallowWaterDataset(split='train')
    val_dataset = ShallowWaterDataset(split='val')
    test_dataset = ShallowWaterDataset(split='test')

    train_loader = DataLoader(train_dataset, 32, shuffle=True, num_workers=5)
    val_loader = DataLoader(val_dataset, 32, num_workers=5)
    test_loader = DataLoader(test_dataset, 32, shuffle=True, num_workers=5)

    trainer = pl.Trainer(
        max_epochs=20,
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir='logs/convlstm',
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    trainer.predict(model, test_loader)


if __name__ == "__main__":
    main()
