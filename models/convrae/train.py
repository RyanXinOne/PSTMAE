from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from torchinfo import summary
from models.convrae.pl_model import LitConvRAE
from data.dataset import DummyDataset


def main():
    dataset = DummyDataset()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_dataset, 32, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, 32, num_workers=2)
    test_loader = DataLoader(test_dataset, 32, num_workers=2)

    model = LitConvRAE(dataset)
    summary(model.model)

    trainer = pl.Trainer(
        max_epochs=20,
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir='logs/convrae',
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    trainer.predict(model, test_loader)


if __name__ == "__main__":
    main()
