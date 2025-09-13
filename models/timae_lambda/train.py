# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from torchinfo import summary
from models.timae.pl_model import LitTiMAE
# from data.dataset import DummyDataset
from data.dataset import ShallowWaterDataset

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lambda-latent", type=float, default=0.5,
        help="λ, for loss = mse + λ * latent_mse"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # dataset = DummyDataset()
    dataset = ShallowWaterDataset()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_dataset, 32, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, 32, num_workers=2)
    test_loader = DataLoader(test_dataset, 32, num_workers=2)

    # model = LitTiMAE(dataset)
    model = LitTiMAE(dataset, lambda_latent=args.lambda_latent)
    summary(model.model)

    trainer = pl.Trainer(
        max_epochs=2,
        logger=True,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir=f'logs/timae/lambda_{args.lambda_latent}',
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    trainer.predict(model, test_loader)


if __name__ == "__main__":
    main()