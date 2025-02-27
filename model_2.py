# ----------------------------------------------------------------------------------------------------
# 라이브러리 목록

# 기본 라이브러리
import os
import logging
import numpy as np

# sklearn 라이브러리
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# tqdm 라이브러리
from tqdm import tqdm

# torch 라이브러리 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta, NAdam, RAdam
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
# ----------------------------------------------------------------------------------------------------





# ----------------------------------------------------------------------------------------------------------
# Multihead Attention 모듈
class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, embed_size, num_heads=4):
        super(MultiheadAttention, self).__init__()

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads  # 각 헤드의 차원 크기

        # Query, Key, Value 변환을 위한 선형 레이어
        self.qkv_proj = nn.Linear(in_channels, embed_size * 3)  # Query, Key, Value를 동시에 계산
        self.fc_out = nn.Linear(embed_size, embed_size)  # Multihead Attention의 최종 출력

        self.softmax = nn.Softmax(dim=-1)
        self.scale = self.head_dim ** 0.5  # Scaling factor
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_length, in_channels = x.shape  # (N, 1, in_channels)
        
        # QKV 변환 (N, seq_length, embed_size * 3) → (N, seq_length, 3, num_heads, head_dim)
        qkv = self.qkv_proj(x).reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (N, seq_length, num_heads, head_dim)
        
        # Transpose to match dot product dimensions: (N, num_heads, seq_length, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (N, num_heads, seq_length, seq_length)
        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # (N, num_heads, seq_length, head_dim)

        # Concatenate heads (N, seq_length, num_heads * head_dim) = (N, seq_length, embed_size)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.num_heads * self.head_dim)

        # 최종 출력 변환
        out = self.fc_out(attn_output)  # (N, seq_length, embed_size)
        
        return out
# ----------------------------------------------------------------------------------------------------------





# ----------------------------------------------------------------------------------------------------------
class MLPWithAttention(nn.Module):
    def __init__(self, in_channels, embed_size=128, num_heads=4):
        super(MLPWithAttention, self).__init__()

        # Multihead Attention 모듈
        self.attn = MultiheadAttention(in_channels, embed_size, num_heads=num_heads)  # 수정된 부분

        # Fully connected layers and batch normalization
        self.fc1 = nn.Linear(embed_size, 64)  # 167 -> 64로 변경
        self.bn1 = nn.BatchNorm1d(64)  # 167 -> 64로 변경
        self.relu1 = nn.LeakyReLU(0.01)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(64, 32)  # 64 -> 32로 변경
        self.bn2 = nn.BatchNorm1d(32)  # 64 -> 32로 변경
        self.relu2 = nn.LeakyReLU(0.01)
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(32, 32)  # 64 -> 32로 변경
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.LeakyReLU(0.01)
        self.drop3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.relu4 = nn.LeakyReLU(0.01)
        self.drop4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(16, 8)
        self.bn5 = nn.BatchNorm1d(8)
        self.relu5 = nn.LeakyReLU(0.01)
        self.drop5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(8, 4)
        self.bn6 = nn.BatchNorm1d(4)
        self.relu6 = nn.LeakyReLU(0.01)
        self.drop6 = nn.Dropout(0.2)

        self.fc7 = nn.Linear(4, 2)
        self.bn7 = nn.BatchNorm1d(2)
        self.relu7 = nn.LeakyReLU(0.01)
        self.drop7 = nn.Dropout(0.2)

        self.fc8 = nn.Linear(2, 1)

    def forward(self, x):
        # Attention 적용 (입력 데이터에 적용)
        x = x.unsqueeze(1)  # (N, 1, in_channels)
        x = self.attn(x)  # (N, 1, embed_size)
        x = x.squeeze(1)  # (N, embed_size)

        # Fully connected layers
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.drop1(x1)

        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.drop2(x2)

        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        x3 = self.drop3(x3)

        x4 = self.fc4(x3)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)
        x4 = self.drop4(x4)

        x5 = self.fc5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)
        x5 = self.drop5(x5)

        x6 = self.fc6(x5)
        x6 = self.bn6(x6)
        x6 = self.relu6(x6)
        x6 = self.drop6(x6)

        x7 = self.fc7(x6)
        x7 = self.bn7(x7)
        x7 = self.relu7(x7)
        x7 = self.drop7(x7)

        # 최종 레이어
        output = self.fc8(x7)

        return output 
# ----------------------------------------------------------------------------------------------------------





# ----------------------------------------------------------------------------------------------------------
class MultiAttentionMLP:
    def __init__(self, 
                 continue_logging=False, initalize_weights=True,
                 use_early_stopping=False, early_stopping_patience=10, use_model_checkpoint=False,
                 use_mixed_precision=False, gradient_accumulation_steps=1):
        
        # gpu 설정 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 정의 
        self.model = MLPWithAttention(in_channels=52, embed_size=4, num_heads=2).to(self.device)

         # 가중치 초기화
        if initalize_weights:
            self._initialize_weights("xavier_uniform")

        # 손실 함수 설정 
        self.criterion = self._get_loss_function("huber")

        # 옵티마이저 설정 
        self.optimizer = self._get_optimizer(optimizer_type="adamw", lr=1e-3, weight_decay=1e-5)

        # 모델 평가 설정 
        self.evaluation_metrics = ['r2_score', 'rmse']

        # early stopping 설정 
        self.use_early_stopping = use_early_stopping
        self.use_model_checkpoint = use_model_checkpoint
        self.best_model_wts = None
        self.early_stopping_patience = early_stopping_patience

        # mixed_precision 설정 
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # 모델 디렉토리 설정 
        model_dir = "C:\\Users\\ssalt\\Documents\\ev_price_predict_project\\data\\train\\B_models\\b_model_2"
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        # 로깅 방식 설정 
        self._setup_logging(self.model_dir, continue_logging) # 로깅 설정 

        # 훈련 결과 
        self.train_loss = None
        self.val_loss = None
        self.rmse = None
        self.r2_score = None

    def _setup_logging(self, model_dir, 
                       log_level=logging.INFO, 
                       continue_logging=False, 
                       log_file_name="training.log"):
        # log 파일 주소 
        log_file = f"{model_dir}/{log_file_name}"

        # 파일 핸들러 설정 
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file, mode='a')  # 항상 'a'로 추가 모드
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 로거 불러오기 
        logger = logging.getLogger()

        # 기존 핸들러를 제거하고 새 핸들러 추가
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file:
                logger.removeHandler(handler)  # 기존 핸들러 제거

        # continue_logging이 False일때 log_file 초기화
        if not continue_logging:
            try:
                with open(log_file, 'w') as file:
                    pass  # 파일을 비우기만 함

            except PermissionError:
                print(f"Permission denied: Unable to initialize the log file at {log_file}.")
                return
            except Exception as e:
                print(f"An error occurred while trying to initialize the log file: {e}")
                return

        logger.addHandler(file_handler)  # 새 핸들러 추가
        

    def _get_optimizer(self, optimizer_type, lr, weight_decay):
        optimizers = {
            'adam': Adam,
            'adamw': AdamW,
            'sgd': SGD,
            'rmsprop': RMSprop,
            'adagrad': Adagrad,
            'adadelta': Adadelta,
            'nadam': NAdam,         
            'radam': RAdam,         
        }
        return optimizers.get(optimizer_type, Adam)(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _get_loss_function(self, loss_function):
        loss_functions = {
            'mse': nn.MSELoss,
            'l1': nn.L1Loss,
            'huber': nn.HuberLoss,
            'cross_entropy': nn.CrossEntropyLoss,
            'bce': nn.BCEWithLogitsLoss,
            'nll': nn.NLLLoss,
            'poisson': nn.PoissonNLLLoss,
            'weighted_residual_loss': WeightedResidualLoss,
            'multiquantileloss': MultiQuantileLoss
        }

        if loss_function == 'weighted_residual_loss':
            return loss_functions[loss_function](weight_factor=1.3)  # 가중치 설정
        elif loss_function == 'multiquantileloss':
            return loss_functions[loss_function](quantiles=[0.25, 0.5, 0.75], weights=[0.2, 0.6, 0.2])
        
        return loss_functions.get(loss_function, nn.MSELoss)()
    
    def _initialize_weights(self, init_method):
        init_methods = {
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'kaiming_uniform': nn.init.kaiming_uniform_,
            'kaiming_normal': nn.init.kaiming_normal_,
            'uniform': nn.init.uniform_,
            'normal': nn.init.normal_
        }
        init_func = init_methods.get(init_method, nn.init.xavier_uniform_)

        # self.model이 iterable이 아니므로, self.model.modules()를 사용하여 순회
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                init_func(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def calculate_metrics(self, y_true, y_pred):
        metrics = {}
        if 'r2_score' in self.evaluation_metrics:
            metrics['r2_score'] = r2_score(y_true, y_pred)
        if 'rmse' in self.evaluation_metrics:
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        if 'mae' in self.evaluation_metrics:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
        if 'mse' in self.evaluation_metrics:
            metrics['mse'] = mean_squared_error(y_true, y_pred)
        return metrics
    
    def move_to_device(self, features, prices):
        return features.to(self.device), prices.to(self.device)
    
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for features, prices in tqdm(dataloader, desc="Evaluating", leave=False):
                features, prices = self.move_to_device(features, prices)
                pred = self.model(features)
                loss = self.criterion(pred, prices)
                running_loss += loss.item()
                y_true.append(prices.cpu().numpy())
                y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        metrics = self.calculate_metrics(y_true, y_pred)

        return running_loss / len(dataloader), metrics
    
    def train_model(self, train_dataloader, val_dataloader, num_epochs, start_epoch=0):
        self.model.train()
        best_val_loss = float('inf')
        early_stopping_counter = 0
        scaler = GradScaler() if self.use_mixed_precision else None

        train_losses = []
        val_losses = []
        val_metrics = []

        for epoch in range(start_epoch, num_epochs):
            running_loss = 0.0
            for step, (features, prices) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)):
                features, prices = self.move_to_device(features, prices)

                # gradscaler 사용 시 
                if scaler:
                    with autocast():
                        pred = self.model(features)
                        loss = self.criterion(pred, prices)
                    scaler.scale(loss).backward()
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()
                
                # 일반 업데이트 시 
                else:
                    pred = self.model(features)
                    loss = self.criterion(pred, prices)
                    loss.backward()
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                running_loss += loss.item()

            # train_loss 기록 
            train_losses.append(running_loss / len(train_dataloader))

            # val_loss, val_metrics 기록 
            val_loss, val_metric = self.evaluate(val_dataloader)
            val_losses.append(val_loss)
            val_metrics.append(val_metric)

            # 로그 메세지(훈련 상황)
            logging.info(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}")
            logging.info(f"Epoch {epoch + 1}/{num_epochs} - Validation Metrics: {val_metric}")

            # Early stopping
            if self.use_early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    self.best_model_wts = self.model.state_dict()
                    if self.use_model_checkpoint:
                        self.save_checkpoint(epoch)
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.early_stopping_patience:
                        logging.info("Early stopping triggered!")
                        break

        # Early stopping 이후 모델에 대한 가중치 다시 가져오기 
        if self.use_early_stopping and self.best_model_wts:
            self.model.load_state_dict(self.best_model_wts)

        # 훈련 결과 저장하기 
        self.train_loss = train_losses[-1]  
        self.val_loss = val_losses[-1] 
        self.rmse = val_metrics[-1]["rmse"] 
        self.r2_score = val_metrics[-1]["r2_score"] 

        return train_losses, val_losses, val_metrics
    
    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.model_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'rmse': self.rmse,
            'r2_score': self.r2_score
        }, checkpoint_path)
        logging.info(f"Checkpoint saved at epoch {epoch + 1}.")

    def save_model(self, filename="best_model.pth", checkpoint=False, epoch=0):
        save_path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'rmse': self.rmse,
            'r2_score': self.r2_score
        }, save_path)
        print(f"Model saved to {save_path}")

        if checkpoint:
            self.save_checkpoint(epoch)

    def load_model(self, model_file_name="best_model.pth"):
        model_path = os.path.join(self.model_dir, model_file_name)
        model = torch.load(model_path, weights_only=False)
        
        # 모델 파라미터와 옵티마이저 상태 로드
        self.model.load_state_dict(model['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(model['optimizer_state_dict'])
        
        # 선택적으로 훈련 정보도 로드
        self.train_loss = model.get('train_loss', None)
        self.val_loss = model.get('val_loss', None)
        self.rmse = model.get('rmse', None)
        self.r2_score = model.get('r2_score', None)

        print(f"Model loaded from {model_path}.")
        self.model.eval()  # 평가 모드로 전환

    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for features in tqdm(dataloader, desc="Making Predictions", leave=False):  # 타겟을 받지 않음
                features = features[0].to(self.device)  # 데이터만 사용
                pred = self.model(features)
                predictions.append(pred.cpu().numpy())

        return np.concatenate(predictions)
# ----------------------------------------------------------------------------------------------------------





# ----------------------------------------------------------------------------------------------------------
class WeightedResidualLoss(nn.Module):
    def __init__(self, weight_factor=2.0):
        super(WeightedResidualLoss, self).__init__()
        self.weight_factor = weight_factor  # 범위를 벗어난 경우 부여할 가중치

    def forward(self, preds, targets):
        device = preds.device  # preds가 위치한 디바이스 설정

        residuals = preds - targets  # 예측값 - 실제값 (잔차 계산)
        errors = residuals ** 2  # MSE 계산

        weight = torch.ones_like(errors, device=device)  # 기본 가중치는 1

        # 잔차의 제곱값이 0.5보다 크면 가중치 적용
        weight[errors > 0.5] = self.weight_factor  

        weighted_loss = errors * weight  # 가중 손실 계산

        return weighted_loss.mean()  # 평균 손실 반환
# ----------------------------------------------------------------------------------------------------------





# ----------------------------------------------------------------------------------------------------------
class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9], weights=[0.2, 0.6, 0.2]):
        super(MultiQuantileLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantiles = torch.tensor(quantiles, device=self.device).view(-1, 1, 1)
        self.weights = torch.tensor(weights, device=self.device).view(-1, 1, 1)

    def forward(self, y_pred, y_true):
        if self.quantiles.device != y_pred.device:
            self.quantiles = self.quantiles.to(y_pred.device)
            self.weights = self.weights.to(y_pred.device)

        error = y_true.unsqueeze(0) - y_pred.unsqueeze(0) 
        q_losses = torch.max(self.quantiles * error, (self.quantiles - 1) * error)

        return torch.mean(torch.sum(self.weights * q_losses, dim=0)) 
# ----------------------------------------------------------------------------------------------------------