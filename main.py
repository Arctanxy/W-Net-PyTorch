# train + validation | inference
import sys
import os
from src.config import conf, infer_conf
from src.trainer import Trainer
from src.data import generate_img

if conf.preload:
    from src.preload_data import PreLoadData as CHNData
    from src.preload_data import CustomBatchSampler, CustomSampler
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
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "train"):
        train_data = CHNData()
        eval_data = CHNData(subset="val")
        if not conf.custom_batch:
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
        else:
            train_sampler = CustomSampler(train_data)
            train_batch_sampler = CustomBatchSampler(
                train_sampler, conf.batch_size, False
            )
            train_dl = DataLoaderX(
                train_data, batch_sampler=train_batch_sampler
            )
            eval_sampler = CustomSampler(eval_data)
            eval_batch_sampler = CustomBatchSampler(
                eval_sampler, conf.batch_size, False
            )
            eval_dl = DataLoaderX(
                eval_data, batch_sampler=eval_batch_sampler
            )

        trainer = Trainer(train_dl, eval_dl)

        for epoch in range(
            conf.start_epoch, conf.start_epoch + conf.num_epochs
        ):
            trainer.train_one_epoch(epoch)

    # 推理部分尚未测试
    elif len(sys.argv) > 1 and sys.argv[1] == "inference":
        ckpt = infer_conf.ckpt
        target_word = infer_conf.target_word
        num_style_words = infer_conf.num_style_words
        target_font_file = infer_conf.target_font_file
        src_img = generate_img(
            target_word,
            osp.join(conf.folder, "fonts", "MSYHBD.TTF"),
            font_size=60,
        )
        target_img = generate_img(target_word, target_font_file, font_size=60)

        totensor = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )
        topil = transforms.ToPILImage()

        src_tensor = totensor(src_img).unsqueeze(0)
        target_tensor = totensor(target_img).unsqueeze(0)

        out = wnet(src_tensor, target_tensor)
        out_tensor = out[0]

        out_img = topil(out_tensor.squeeze(0))
        try:
            plt.subplot(131)
            plt.imshow(src_img, cmap="gray")
            plt.subplot(132)
            plt.imshow(out_img, cmap="gray")
            plt.subplot(133)
            plt.imshow(target_img, cmap="gray")
            plt.show()
        except Exception as e:
            print(e)
            out_img.save("./out.jpg")
            src_img.save("./src.jpg")
            target_img.save("./target.jpg")

        # TODO：批量输入风格字体。
