import ast
import random
from pathlib import Path

import pandas as pd
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import transform_test, transform_train
from src.data.utils import FrameLoader, id2int, pre_caption
from src.tools.files import write_txt
from src.tools.utils import print_dist

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class WebVidCoVRDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        annotation: dict = {"train": "", "val": ""},
        vid_dirs: dict = {"train": "", "val": ""},
        emb_dirs: dict = {"train": "", "val": ""},
        mm_emb_dirs: dict = {"train": "", "val": ""},
        image_size: int = 384,
        emb_pool: str = "query",
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        n_embs: int = 15,
        si_tc_weight=0,
        noise_ratio=0.0,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames

        self.transform_train = transform_train(image_size)
        self.transform_test = transform_test(image_size)

        self.data_train = WebVidCoVRDataset(
            transform=self.transform_train,
            annotation=annotation["train"],
            vid_dir=vid_dirs["train"],
            emb_dir=emb_dirs["train"],
            mm_emb_dir=mm_emb_dirs["train"],
            split="train",
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
            n_embs=n_embs,
            si_tc_weight=si_tc_weight,
            noise_ratio=noise_ratio,
        )
        self.data_val = WebVidCoVRDataset(
            transform=self.transform_test,
            annotation=annotation["val"],
            vid_dir=vid_dirs["val"],
            emb_dir=emb_dirs["val"],
            mm_emb_dir=mm_emb_dirs["val"],
            split="val",
            emb_pool=self.emb_pool,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
            n_embs=n_embs,
            noise_ratio=noise_ratio,
        )

    def prepare_data(self):
        # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
        # download data, pre-process, split, save to disk, etc...
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class WebVidCoVRTestDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        annotation: str,
        vid_dirs: str,
        emb_dirs: str,
        mm_emb_dirs:str,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 384,
        emb_pool: str = "query",
        n_embs: int = 15,
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        **kwargs,  # type: ignore
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        print("Init WebVid-CoVR_CEL Test Dataset !!!")
        

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.emb_pool = emb_pool
        self.iterate = iterate
        self.vid_query_method = vid_query_method
        self.vid_frames = vid_frames

        self.transform_test = transform_test(image_size)

        self.data_test = WebVidCoVRDataset(
            transform=self.transform_test,
            annotation=annotation,
            vid_dir=vid_dirs,
            emb_dir=emb_dirs,
            mm_emb_dir=mm_emb_dirs,
            split="test",
            emb_pool=self.emb_pool,
            n_embs=n_embs,
            iterate=self.iterate,
            vid_query_method=self.vid_query_method,
            vid_frames=self.vid_frames,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class WebVidCoVRDataset(Dataset):
    def __init__(
        self,
        transform,
        annotation: str,
        vid_dir: str,
        emb_dir: str,
        mm_emb_dir:str,
        split: str,
        max_words: int = 100,
        emb_pool: str = "query",
        n_embs: int = 15,
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
        si_tc_weight=0,
        noise_ratio=0.0,
    ) -> None:
        super().__init__()

        self.transform = transform

        self.annotation_pth = Path(annotation)
        assert (
            self.annotation_pth.exists()
        ), f"Annotation file {annotation} does not exist"
        self.df = pd.read_csv(annotation)

        self.vid_dir = Path(vid_dir)
        self.emb_dir = Path(emb_dir)
        self.mm_emb_dir = Path(mm_emb_dir)
        
        assert self.vid_dir.exists(), f"Image directory {self.vid_dir} does not exist"
        assert self.emb_dir.exists(), f"Embedding directory {emb_dir} does not exist"
        assert self.mm_emb_dir.exists(), f"Multimodel Embedding directory {self.mm_emb_dir} does not exist"


        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of train, val, or test"
        self.split = split

        vid_pths = self.vid_dir.glob("*/*.mp4")
        emb_pths = self.emb_dir.glob("*/*.pth")
        mm_emb_pths = self.mm_emb_dir.glob("*/*.pth")

        id2vidpth = {
            vid_pth.parent.stem + "/" + vid_pth.stem: vid_pth for vid_pth in vid_pths
        }
        id2embpth = {
            emb_pth.parent.stem + "/" + emb_pth.stem: emb_pth for emb_pth in emb_pths
        }
        id2mmembpth = {
            mm_emb_pth.parent.stem + "/" + mm_emb_pth.stem: mm_emb_pth for mm_emb_pth in mm_emb_pths
        }        

        assert len(id2vidpth) > 0, f"No videos found in {vid_dir}"
        assert len(id2embpth) > 0, f"No embeddings found in {emb_dir}"
        assert len(id2mmembpth) > 0, f"No mm embeddings found in {id2mmembpth}"


        self.df["path1"] = self.df["pth1"].apply(lambda x: id2vidpth.get(x, None))  # type: ignore
        self.df["path2"] = self.df["pth2"].apply(lambda x: id2embpth.get(x, None))  # type: ignore
        self.df["mmembpath"] = self.df["pth2"].apply(lambda x: id2mmembpth.get(x, None))  # type: ignore

        # Count unique missing paths
        missing_pth1 = self.df[self.df["path1"].isna()]["pth1"].unique().tolist()
        missing_pth1.sort()
        total_pth1 = self.df["pth1"].nunique()

        missing_pth2 = self.df[self.df["path2"].isna()]["pth2"].unique().tolist()
        missing_pth2.sort()
        total_pth2 = self.df["pth2"].nunique()
        
        missing_pth3 = self.df[self.df["mmembpath"].isna()]["mmembpath"].unique().tolist()
        missing_pth3.sort()
        total_pth3 = self.df["mmembpath"].nunique()

        if len(missing_pth1) > 0:
            print_dist(
                f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%), saving them to missing_pth1-{split}.txt"
            )
            write_txt(missing_pth1, f"missing_pth1-{split}.txt")
        if len(missing_pth2) > 0:
            print_dist(
                f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%), saving them to missing_pth2-{split}.txt"
            )
            write_txt(missing_pth2, f"missing_pth2-{split}.txt")
        if len(missing_pth3) > 0:
            print_dist(
                f"Missing {len(missing_pth3)} mmembpath's ({len(missing_pth3)/total_pth3 * 100:.1f}%), saving them to missing_pth3-{split}.txt"
            )
            write_txt(missing_pth3, f"missing_pth3-{split}.txt")

        # Remove missing paths
        self.df = self.df[self.df["path1"].notna()]
        self.df = self.df[self.df["path2"].notna()]
        self.df = self.df[self.df["mmembpath"].notna()]
        self.df.reset_index(drop=True, inplace=True)

        self.max_words = max_words

        assert emb_pool in [
            "middle",
            "mean",
            "query",
        ], f"Invalid emb_pool: {emb_pool}, must be one of middle, mean, or query"
        self.emb_pool = emb_pool
        self.n_embs = n_embs

        if iterate in ["idx", "triplets"]:
            iterate = "idx"
            self.df["idx"] = self.df.index
        # iterate = 'pth2'
        self.iterate = iterate
        self.target_txts = self.df[iterate].unique()
        assert (
            iterate in self.df.columns
        ), f"{iterate} not in {self.annotation_pth.stem}"
        self.df.sort_values(iterate, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df["int1"] = self.df["pth1"].apply(lambda x: id2int(x, sub="0"))
        self.df["int2"] = self.df["pth2"].apply(lambda x: id2int(x, sub="0"))
        self.pairid2ref = self.df["int1"].to_dict()
        assert (
            self.df["int1"].nunique() == self.df["pth1"].nunique()
        ), "int1 is not unique"
        assert (
            self.df["int2"].nunique() == self.df["pth2"].nunique()
        ), "int2 is not unique"
        # int2id is a dict with key: int1, value: pth1
        self.int2id = self.df.groupby("int1")["pth1"].apply(set).to_dict()
        self.int2id = {k: list(v)[0] for k, v in self.int2id.items()}

        self.pairid2tar = self.df["int2"].to_dict()
        self.df.set_index(iterate, inplace=True)
        self.df[iterate] = self.df.index

        if split == "test":
            assert (
                len(self.target_txts) == self.df.shape[0]
            ), "Test split should have one caption per row"

        assert (
            vid_query_method
            in [
                "middle",
                "mean",
            ]
        ), f"Invalid vid_query_method: {vid_query_method}, must be one of middle, or mean"
        self.frame_loader = FrameLoader(
            transform=self.transform, method=vid_query_method, frames_video=vid_frames
        )

                
        self.noise_ratio = noise_ratio
        print(f"self.noise_ratio : {self.noise_ratio}")

        if self.split == 'train' and self.noise_ratio > 0:
            self.shuffle()
        else:
            print(f"all clean !!\n")

    def shuffle(self):
        print(f'shuffle data with noise ratio {self.noise_ratio}')
        unique_indices = self.df.index.unique() 
        num_noise = int(len(unique_indices) * self.noise_ratio)
        shuffle_unique_idx = random.sample(list(unique_indices), num_noise)  
        self.shuffle_unique_idx = shuffle_unique_idx
        
        def batch_shuffle_all(field, indices):
            values = self.df.loc[indices, field].tolist()  
            random.shuffle(values)
            self.df.loc[indices, field] = values
            
        # batch_shuffle_all(field="path1", indices=shuffle_unique_idx)
        batch_shuffle_all(field="edit", indices=shuffle_unique_idx)
        batch_shuffle_all(field="path2", indices=shuffle_unique_idx)
        
        print('shuffle data with noise done')


    def __len__(self) -> int:
        return len(self.target_txts)

    def __getitem__(self, index):
        target_txt = self.target_txts[index]
        ann = self.df.loc[target_txt]
        if ann.ndim > 1:
            if self.noise_ratio == 0 or target_txt in self.shuffle_unique_idx:
                ann = ann.sample()
            ann = ann.iloc[0]

        reference_pth = str(ann["path1"])
        reference_vid = self.frame_loader(reference_pth)
        
        caption = pre_caption(ann["edit"], self.max_words)
        description = str(ann["txt1"])
        
        return_dict = {
            "ref_img": reference_vid,
            "edit": caption,
            "pair_id": index,
            "description": description
        }
        
        
        target_pth_mmemb = str(ann["mmembpath"])
        target_mmemb = torch.load(target_pth_mmemb).cpu()
        if self.emb_pool == "middle":
            target_mmemb = target_mmemb[len(target_mmemb) // 2]
        else:
            target_mmemb = target_mmemb.mean(0)
        return_dict["target_mmemb"] = target_mmemb
 
        # Get target embeddings
        target_pth = str(ann["path2"])
        target_emb = torch.load(target_pth, weights_only=True).cpu().to(torch.float32)
        if self.emb_pool == "middle":
            return_dict["tar_img_feat"] = target_emb[len(target_emb) // 2]
            return return_dict

        n_target_emb = min(self.n_embs, len(target_emb))
        sampled_indices = random.sample(range(len(target_emb)), n_target_emb)
        sampled_target_emb = target_emb[sampled_indices]

        if self.emb_pool == "mean":
            return_dict["tar_img_feat"] = sampled_target_emb.mean(0)
            return return_dict

        assert self.emb_pool == "query", f"Invalid emb_pool: {self.emb_pool}"

        vid_scores = ast.literal_eval(str(ann["scores"]))
        if len(vid_scores) == 0 or len(target_emb) != len(vid_scores):
            vid_scores = [1.0] * n_target_emb
        else:
            vid_scores = [vid_scores[i] for i in sampled_indices]
        vid_scores = torch.Tensor(vid_scores)
        vid_scores = (vid_scores / 0.1).softmax(dim=0)
        if len(target_emb.shape) == 2:
            return_dict["tar_img_feat"] = torch.einsum(
                "f,fe->e", vid_scores, sampled_target_emb
            )
        elif len(target_emb.shape) == 3:
            return_dict["tar_img_feat"] = torch.einsum(
                "f,fqc->qc", vid_scores, sampled_target_emb
            )
        else:
            raise ValueError(
                f"target_emb must be 2 or 3 dimensional, got {len(target_emb.shape)}"
            )

        return return_dict
