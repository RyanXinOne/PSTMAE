from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchinfo import summary
from models.autoencoder.pl_model import LitAutoEncoder
from data.dataset import ShallowWaterDataset


def main():
    model = LitAutoEncoder()
    summary(model.model)

    train_dataset = ShallowWaterDataset(split='train', forecast_steps=0, flatten=False)
    val_dataset = ShallowWaterDataset(split='val', forecast_steps=0, flatten=False)
    test_dataset = ShallowWaterDataset(split='test', forecast_steps=0, flatten=False)

    train_loader = DataLoader(train_dataset, 32, shuffle=True)
    val_loader = DataLoader(val_dataset, 32)
    test_loader = DataLoader(test_dataset, 32, shuffle=True)

    trainer = pl.Trainer(
        max_epochs=50,
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir='logs/autoencoder',
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    trainer.predict(model, test_loader)


if __name__ == "__main__":
    pl.seed_everything(42)
    main()
