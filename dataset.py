import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import gzip
import shutil

from gensim.models.fasttext import load_facebook_vectors

import timm

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets
import albumentations as A



# класс конфигурации модели
class MultiModelConfig:
    def __init__(self):
        
        # текст
        self.fasttext_model_path = r"D:\folders\fasttext_models\cc.ru.300.bin.gz"

        # изображения
        self.image_model_name = 'maxxvitv2_nano_rw_256'
        self.resize = 256
        self.mean = [0.485, 0.456, 0.406] # многие модели обучаются на ImageNet, средние возьмем оттуда
        self.std = [0.229, 0.224, 0.225]

        # масса
        self.mass_encoder_output_dim = 1

        # общие параметры
        self.emb_dim = 128
        self.batch_size = 4
        self.epochs = 20
        self.weight_decay = 1e-4
        self.criterion = 'MAE'

        # lr
        self.lr_fasttext_proj = 0.01
        self.lr_image_proj = 0.01
        self.lr_mass_model = 0.01
        self.lr_mass_proj = 0.01
        self.lr_head = 0.01


        # глобальные параметры
        self.model_name = ''
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_path = Path(r'C:\Users\bagen\OneDrive\Рабочий стол\Data Analyst\Deep Learning\dl_sprint_4\data')
        self.seed = 42
        self.save_path = Path(r'C:\Users\bagen\OneDrive\Рабочий стол\Data Analyst\Deep Learning\dl_sprint_4\results')

    def show_all_attrs(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")



# создание датафреймов
def create_dataframes(cfg):
    data_path = cfg.data_path

    imgs_paths = datasets.ImageFolder(root=data_path / 'images')
    df_paths = pd.DataFrame(
        [(path, idx, imgs_paths.classes[idx]) for path, idx in imgs_paths.imgs],
        columns=['image_path', 'class_index', 'dish_id']
    )[['image_path', 'dish_id']]

    df_ingredients = pd.read_csv(data_path / 'ingredients.csv')

    df_dish = pd.read_csv(data_path / 'dish.csv')
    df_dish.ingredients = df_dish.ingredients.str.split(';')

    df_dish_ingredients = df_dish[['dish_id', 'ingredients']]
    df_dish_exploded = df_dish_ingredients.explode('ingredients', ignore_index=True)
    df_dish_exploded['ingredient_code'] = df_dish_exploded.ingredients.str.extract(r'ingr_0*(\d+)').astype(int)
    df_dish_exploded = pd.merge(left=df_dish_exploded, right=df_ingredients, left_on='ingredient_code', right_on='id')
    df_dish_exploded = df_dish_exploded.groupby('dish_id')['ingr'].apply(';'.join).reset_index()
    df_dish_exploded.rename(columns={'ingr': 'ingredients_string'}, inplace=True)

    df_dish_final = pd.merge(left=df_dish, right=df_dish_exploded, on='dish_id')[['dish_id', 'total_calories', 'total_mass', 'split', 'ingredients_string']]
    df_dish_final = pd.merge(left=df_dish_final, right=df_paths, on='dish_id')

    df_train = df_dish_final[df_dish_final['split'] == 'train'].reset_index(drop=True)
    df_test = df_dish_final[df_dish_final['split'] == 'test'].reset_index(drop=True)

    return df_train, df_test

# fasttext для токенизации
def get_fasttext(cfg):
    """
        Возвращает модель fasttext из пути формата "<путь к файлу fasttext>.bin.gz"
        Либо берет уже распакованный файл "<путь к файлу fasttext>.bin"
    """
    gz_path = cfg.fasttext_model_path
    bin_path = gz_path.replace('.gz', '')
    
    # если bin-файл ещё не существует, распаковываем
    if not Path(bin_path).exists():
        with gzip.open(gz_path, 'rb') as f_in:
            with open(bin_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    ft_model = load_facebook_vectors(bin_path)
    return ft_model

# Опциональный код предобработки/аугментации данных. Зависит от конфига
def get_transforms(cfg, data_type, other_transforms=None):
    """
        Возвращает вариант обработки изображений в зависимости от типа датасета: train или test
    """
    if data_type == "train":
        transforms = A.Compose(
            [
                A.Resize(cfg.resize, cfg.resize),
                A.HorizontalFlip(p=0.3),
                A.SquareSymmetry(p=0.5),
                A.CoarseDropout(
                        num_holes_range=(1, 2),
                        hole_height_range=(int(0.02 * cfg.resize), int(0.07 * cfg.resize)),
                        hole_width_range=(int(0.02 * cfg.resize), int(0.07 * cfg.resize)),
                        fill=0,
                        p=0.3),
                A.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    p=0.7
                    ),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)  # конвертируем numpy HxWxC в torch.Tensor CxHxW
            ],
            seed=cfg.seed
        )
    else:
        transforms = A.Compose(
            [
                A.Resize(cfg.resize, cfg.resize),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)  # конвертируем numpy HxWxC в torch.Tensor CxHxW
            ],
            seed=cfg.seed
        )
    return other_transforms if other_transforms else transforms # возвращаем либо измененный пайплайн, либо стандартный


# датасет
class MultiModalDataset(Dataset):
    def __init__(self, df, fasttext_model, transforms, cfg):
        self.df = df
        self.image_cfg = timm.get_pretrained_cfg(cfg.image_model_name)
        self.fasttext_model = fasttext_model
        self.transforms = transforms

        # вычисляем текстовые векторы сразу для всех примеров (FastText не обучается)
        self.text_vectors = []
        for idx in range(len(df)):
            text = df.loc[idx, 'ingredients_string']
            vec = torch.from_numpy(self.fasttext_model.get_sentence_vector(text)).float()
            self.text_vectors.append(vec)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text_vector = self.text_vectors[idx]
        total_mass = self.df.loc[idx, 'total_mass']

        total_calories = self.df.loc[idx, 'total_calories']

        image_path = self.df.loc[idx, 'image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image=np.array(image))["image"]

        return {
            "text_vector": text_vector, 
            "image": image, 
            "total_mass": total_mass, 
            "total_calories": total_calories
            }

# для загрузки в dataloader
def collate_fn(batch):
    text_vectors = torch.stack([item["text_vector"] for item in batch])
    masses = torch.FloatTensor([item["total_mass"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    total_calories = torch.FloatTensor([item["total_calories"] for item in batch])
    
    return {
        "text_vectors": text_vectors,
        "image": images,
        "masses": masses,
        "total_calories": total_calories
    }

# подготовить loaders
def prepare_loaders(df_train, df_test, ft_model, image_model_name=None, resize=None, model_name='m1'):
    cfg = MultiModelConfig()

    if image_model_name:
        cfg.image_model_name = image_model_name
    if resize:
        cfg.resize = resize if resize else None
    if model_name:
        cfg.model_name = model_name

    transforms_train = get_transforms(cfg, data_type='train')
    transforms_test = get_transforms(cfg, data_type='test')

    ds_train = MultiModalDataset(df_train, ft_model, transforms_train, cfg)
    ds_test = MultiModalDataset(df_test, ft_model, transforms_test, cfg)

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn) 
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn) 

    return cfg, train_loader, test_loader

