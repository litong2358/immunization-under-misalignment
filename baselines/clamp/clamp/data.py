import pytorch_lightning as pl
from timm.data import create_dataset, create_loader

IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]


def combined_loader(loader1, loader2):
    iter1 = iter(loader1)
    iter2 = iter(loader2)
    while True:
        try:
            batch1 = next(iter1)
        except StopIteration:
            break
        try:
            batch2 = next(iter2)
        except StopIteration:
            iter2 = iter(loader2)
            batch2 = next(iter2)
        yield batch1, batch2


class CombinedLoader:
    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2
        self.len1 = len(loader1)
        self.len2 = len(loader2)

    def __iter__(self):
        return combined_loader(self.loader1, self.loader2)

    def __len__(self):
        return self.len2


class ImmunizationData(pl.LightningDataModule):
    def __init__(self, config, debug=False):
        super().__init__()
        self.config = config
        self.batch_size = config["batch_size"]
        self.num_workers = config.get("num_workers", 4)
        self.image_size = config.get("image_size", 224)
        self.name = config["harmful_name"]
        self.debug = debug
        self.primary_train = None
        self.primary_val = None
        self.harmful_train = None
        self.harmful_val = None

    def setup(self, stage=None):
        if self.primary_train is not None:
            return
        config = self.config
        self.primary_train = create_dataset("imagenet", root=config["primary_root"], split="train")
        self.primary_val = create_dataset("imagenet", root=config["primary_root"], split="val")
        self.harmful_train = create_dataset(
            "image_folder", root=config["harmful_root"], split=config["harmful_train_split"]
        )
        self.harmful_val = create_dataset(
            "image_folder", root=config["harmful_root"], split=config["harmful_val_split"]
        )
        print(
            f"[data] primary train={len(self.primary_train)} val={len(self.primary_val)} | "
            f"harmful({self.name}) train={len(self.harmful_train)} val={len(self.harmful_val)}"
        )

    def _loader(self, dataset, training):
        return create_loader(
            dataset,
            batch_size=self.batch_size,
            input_size=(3, self.image_size, self.image_size),
            is_training=training,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            interpolation="bicubic",
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        self.setup()
        return CombinedLoader(
            self._loader(self.primary_train, True),
            self._loader(self.harmful_train, True),
        )

    def harmful_dataloaders(self, augment=True):
        self.setup()
        return self._loader(self.harmful_train, augment), self._loader(self.harmful_val, False)
