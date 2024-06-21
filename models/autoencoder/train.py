import lightning.pytorch as pl
from torchinfo import summary
from models.autoencoder.pl_model import LitAutoEncoder
from data.dataset import CompressibleNavierStokesDataset
from torch.utils.data import DataLoader, random_split


def main():
    model = LitAutoEncoder()
    summary(model.model)

    dataset = CompressibleNavierStokesDataset(sequence_steps=1, forecast_steps=0, masking_steps=0)
    train_size = int(0.9 * len(dataset))
    val_size = int(0.05 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, 64, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, 64, num_workers=2)
    test_loader = DataLoader(test_dataset, 64, num_workers=2)

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
