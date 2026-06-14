import timm
import numpy as np
import torch
import torch.nn as nn
import time
import random
import matplotlib.pyplot as plt
from dataset import *


# преобразователь массы
class MassEncoder(nn.Module):
    """
        Преобразует массу в вектор
    """
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Linear(1, cfg.mass_encoder_output_dim)
    
    def forward(self, x):
        return self.net(x)


class BaseMultimodalModel(nn.Module):
    """
        Мультимодальная сеть
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb_dim = cfg.emb_dim

        # Текстовая ветка
        # self.text_model = AutoModel.from_pretrained(cfg.text_model_name)

        # Визуальная ветка
        self.image_model = timm.create_model(
            cfg.image_model_name,
            pretrained=True,
            num_classes=0 # Возвращаем вектор признаков, а не классы
        )
        # Заморозка всех параметров image_model, обучаться будут только последующие линейные части
        for param in self.image_model.parameters():
            param.requires_grad = False

        # Ветка для массы
        self.mass_model = MassEncoder(cfg)

        # эмбеддинги для каждого параметра, приведенные к emb_dim
        self.fasttext_proj = nn.Sequential(
            nn.Linear(300, self.emb_dim), # линейный слой для преобразования эмбеддингов fasttext к нужной размерности
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            )
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_model.num_features, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            )
        self.mass_proj = nn.Sequential(
            nn.Linear(cfg.mass_encoder_output_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.ReLU(),
            )

        # конечная голова, которая определяет калорийность
        # self.head = nn.Linear(self.emb_dim*3, 1)
        self.head = nn.Linear(self.emb_dim, 1)

    def get_cfg(self, show=False):

        if show:
            print(f'''
            текст:
                fasttext_model_path = {self.cfg.fasttext_model_path}
                
            изображения:
                image_model_name = {self.cfg.image_model_name}
                resize = {self.cfg.resize}
                mean = {self.cfg.mean}
                std = {self.cfg.std}

            масса:
                mass_encoder_output_dim = {self.cfg.mass_encoder_output_dim}

            lr:
                lr_fasttext_proj = {self.cfg.lr_fasttext_proj}
                lr_image_proj = {self.cfg.lr_image_proj}
                lr_mass_model = {self.cfg.lr_mass_model}
                lr_mass_proj = {self.cfg.lr_mass_proj}
                lr_head = {self.cfg.lr_head}

            общие параметры:
                batch_size = {self.cfg.batch_size}
                epochs = {self.cfg.epochs}
                weight_decay = {self.cfg.weight_decay}
                save_path = {self.cfg.save_path}
                criterion = {self.cfg.criterion}

            глобальные параметры:
                device = {self.cfg.device}
                data_path = {self.cfg.data_path}
                seed = {self.cfg.seed}
            ''')

        return self.cfg

    def forward(self, text_vector, image_input, mass_input):
        # эмбеддинги текста
        text_emb = self.fasttext_proj(text_vector)

        # эмбеддинги изображений
        image_features = self.image_model(image_input)
        image_emb = self.image_proj(image_features)
    
        # эмбеддинги массы
        mass_features = self.mass_model(mass_input)
        mass_emb = self.mass_proj(mass_features)

        # final_emb = torch.cat([text_emb, image_emb, mass_emb], dim=1)
        final_emb = text_emb*image_emb*mass_emb

        result_pred = self.head(final_emb)

        return result_pred

# подсчитать кол-во параметров
def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров в модели: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")

# тренировка на 1 эпохе
def train_one_epoch(cfg, model, loader, optimizer, criterion):
    device = cfg.device
    model.train()
    total_loss = 0
    for batch in loader:
        text_vec = batch["text_vectors"].to(device)
        images = batch["image"].to(device)
        masses = batch["masses"].to(device).unsqueeze(1)
        targets = batch["total_calories"].to(device)

        optimizer.zero_grad()
        masses = masses
        preds = model(text_vec, images, masses)
        loss = criterion(preds.squeeze(1), targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * text_vec.size(0)

    return total_loss / len(loader.dataset)

# валидация
def evaluate(cfg, model, loader, criterion):
    with torch.no_grad():
        device = cfg.device
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        for batch in loader:
            text_vec = batch["text_vectors"].to(device)
            images = batch["image"].to(device)
            masses = batch["masses"].to(device).unsqueeze(1)
            targets = batch["total_calories"].to(device)

            preds = model(text_vec, images, masses)
            loss = criterion(preds.squeeze(1), targets)
            total_loss += loss.item() * text_vec.size(0)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

        mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        pred_kcal = np.abs(np.array(all_preds))
        return total_loss / len(loader.dataset), mae, pred_kcal

# сохранить модель
def save_model(model, best_model_path):
    torch.save(model.state_dict(), best_model_path)

# загрузить модель
def load_model(load_path):
    torch.load(load_path)

# запуск обучения
def train(cfg, model, train_loader, test_loader, optimizer, criterion, scheduler):
    epochs = cfg.epochs
    train_losses = []
    val_losses = []
    best_val_loss = 50.0**2 if cfg.criterion == 'MSE' else 50.0 # если выбран критерий MSE, то берем порог в 2500 единиц, иначе 50 (Huber примерно равен MAE)
    best_model_path = cfg.save_path / f'best_model_{cfg.model_name}.pt'

    print(f'Обучение на основе модели {cfg.image_model_name} \n')

    for epoch in range(epochs):
        start = time.time()
        train_loss = train_one_epoch(cfg, model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate(cfg, model, test_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        end = time.time() - start
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | epoch time {end:.2f} sec")

        # сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, best_model_path)
            print(f"Лучшая модель с val_loss={val_loss:.4f} сохранена в {best_model_path}")

    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))

    # График loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# получить критерий
def get_criterion(cfg):
    if cfg.criterion == 'MAE':
        cr = nn.L1Loss()
    if cfg.criterion == 'MSE':
        cr = nn.MSELoss()
    if cfg.criterion == 'Huber Loss':
        cr = nn.SmoothL1Loss()

    return cr

# установить значение seed
def set_seed(cfg):
        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        g = torch.Generator()
        g.manual_seed(cfg.seed)

# собрать модель и начать обучение
def start_pipeline(cfg, train_loader, test_loader):

    m = BaseMultimodalModel(cfg).to(cfg.device)
    count_params(m)
    criterion = get_criterion(cfg)

    optimizer = torch.optim.AdamW([
        {'params': m.fasttext_proj.parameters(), 'lr': cfg.lr_fasttext_proj},
        {'params': m.image_proj.parameters(), 'lr': cfg.lr_image_proj},
        {'params': m.mass_model.parameters(), 'lr': cfg.lr_mass_model},
        {'params': m.mass_proj.parameters(), 'lr': cfg.lr_mass_proj},
        {'params': m.head.parameters(), 'lr': cfg.lr_head},
        ], 
        weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    train(cfg, m, train_loader, test_loader, optimizer, criterion, scheduler)

    return m

# получить точечное предсказание
def predict_kcal(model, fasttext_model, image_path, text, mass):
    model.eval()
    device = next(model.parameters()).device
    cfg = model.get_cfg()

    text_vec = torch.from_numpy(fasttext_model.get_sentence_vector(text)).float()
    text_vec = text_vec.unsqueeze(0).to(device)

    transform = get_transforms(cfg, data_type='test')
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image=np.array(image))["image"]
    image_tensor = image_tensor.unsqueeze(0).to(device)

    mass_tensor = torch.tensor([mass], dtype=torch.float32).view(1, 1).to(device)

    with torch.no_grad():
        pred = model(text_vec, image_tensor, mass_tensor)
        kcal = pred.squeeze().cpu().item()

    return kcal

# получить топ-k плохих предсказаний тестового датасета
def get_top_k_errors(model, test_loader, df_test, k=5):
    cfg = model.get_cfg()
    device = cfg.device
    model.eval()
    
    all_dish_ids = []
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            text_vec = batch["text_vectors"].to(device)
            images = batch["image"].to(device)
            masses = batch["masses"].to(device).unsqueeze(1)
            targets = batch["total_calories"].to(device)
            
            preds = model(text_vec, images, masses).squeeze(1)
            
            all_true.extend(targets.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    
    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)
    abs_error = np.abs(pred_arr - true_arr)
    
    top_k_idx = np.argsort(abs_error)[::-1][:k]
    
    result = df_test.iloc[top_k_idx][['dish_id', 'total_calories']].copy()
    result.rename(columns={'total_calories': 'true_calories'}, inplace=True)
    result['pred_calories'] = pred_arr[top_k_idx]
    result['abs_error'] = abs_error[top_k_idx]
    
    return result