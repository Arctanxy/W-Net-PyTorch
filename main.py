# train & validation
from src.config import conf
from src.trainer import Trainer

if conf.preload:
    from src.preload_data import PreLoadData as CHNData
else:
    from src.data import CHNData

try:
    from torch.utils.data import DataLoader
    from prefetch_generator import BackgroundGenerator

    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())


except Exception as e:
    print("未找到prefetch_generator")
    from torch.utils.data import DataLoader as DataLoaderX

if __name__ == "__main__":
    train_data = CHNData()
    eval_data = CHNData(subset="val")
    train_dl = DataLoaderX(
        train_data,
        batch_size=conf.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=conf.num_workers,
    )
    eval_dl = DataLoaderX(
        eval_data,
        batch_size=conf.batch_size * 2,
        shuffle=False,
        drop_last=True,
        num_workers=conf.num_workers,
    )

    trainer = Trainer(train_dl, eval_dl)

    for epoch in range(conf.start_epoch, conf.start_epoch + conf.num_epochs):
        trainer.train_one_epoch(epoch)

