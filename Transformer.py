import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import wandb

model_path = 'Transformer.ckpt'
predict_csv_path = 'result.csv'
# 設定特徵欄位和目標欄位
feature_columns = ['LocationCode', 'WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)']
target_column = 'Power(mW)'

# 將資料轉換成序列資料
seq_len = 10  # 序列長度
input_sequences = []
target_sequences = []
power_correct = []

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# 讀取每個觀測站的資料
file_paths = glob.glob('36_TrainingData/L*_Train.csv')  # 替換成資料夾的路徑
for file_path in file_paths:
    print(file_path)
    # 讀取每個 CSV 檔案
    data = pd.read_csv(file_path)
    
    # 按照時間戳排序並解析日期和時間
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data = data.sort_values('DateTime').reset_index(drop=True)
    
    # 正規化特徵欄位
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    
    # 提取每日的序列數據
    dates = data['DateTime'].dt.date
    unique_dates = dates.unique()
    
    for date in unique_dates:
        print(date)
        # 設定時間範圍
        start_time = pd.Timestamp(date, hour=0, minute=0, second=0)
        end_time = pd.Timestamp(date, hour=0, minute=0, second=0) + pd.Timedelta(days=1) + pd.Timedelta(hours=9)  # 隔天 9:00 AM

        # 提取當天到隔天 9:00 AM 之前的數據作為輸入
        input_data = data[(data['DateTime'] >= start_time) & (data['DateTime'] < end_time)]
        
        # 提取隔天 9:00 AM 到 5:00 PM 的發電量作為目標
        target_start_time = pd.Timestamp(date, hour=0, minute=0, second=0) + pd.Timedelta(days=1) + pd.Timedelta(hours=9)
        target_end_time = pd.Timestamp(date, hour=0, minute=0, second=0) + pd.Timedelta(days=1) + pd.Timedelta(hours=17)
        target_data = data[(data['DateTime'] >= target_start_time) & (data['DateTime'] <= target_end_time)]
        
        # 檢查資料完整性
        if len(input_data) > 0 and len(target_data) > 0:
            input_seq = input_data[feature_columns].values  # 輸入特徵序列
            power_seq = target_data[target_column].values

            # 處理輸入序列補齊
            if len(input_seq) < 1500:
                # 如果資料不足，使用零填充至預期長度
                padding = np.zeros((1500 - len(input_seq), len(feature_columns)))
                input_seq = np.vstack((input_seq, padding))
            elif len(input_seq) > 1500:
                # 如果資料過多，截取到預期長度
                input_seq = input_seq[:1500]
            
            # 處理目標序列補齊
            if len(power_seq) < 1500:  # 480 分鐘為當天 9:00 AM 到 5:00 PM
                padding = np.zeros(1500 - len(power_seq))
                power_seq = np.concatenate((power_seq, padding))
            elif len(power_seq) > 1500:
                power_seq = power_seq[:1500]

            target_sequences.extend(power_seq)
            input_sequences.extend(input_seq)

# 設定分割比例
train_ratio = 0.7   # 70% 用於訓練
val_ratio = 0.15    # 15% 用於驗證
test_ratio = 0.15   # 15% 用於測試

train_dataset, val_dataset, train_targets, val_targets = train_test_split(input_sequences, target_sequences, test_size=val_ratio+test_ratio)
val_dataset, test_dataset, val_targets, test_targets = train_test_split(val_dataset, val_targets, test_size=test_ratio)

# 轉換為 PyTorch tensors
train_dataset_tensor = torch.tensor(train_dataset, dtype=torch.float32)
train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
val_dataset_tensor = torch.tensor(val_dataset, dtype=torch.float32)
val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)
test_dataset_tensor = torch.tensor(test_dataset, dtype=torch.float32)
test_targets_tensor = torch.tensor(test_targets, dtype=torch.float32)
print(train_dataset_tensor.size())
print(train_targets_tensor.size())
print(val_dataset_tensor.size())
print(val_targets_tensor.size())
print(test_dataset_tensor.size())
print(test_targets_tensor.size())

train_dataset = TensorDataset(train_dataset_tensor, train_targets_tensor)
val_dataset = TensorDataset(val_dataset_tensor, val_targets_tensor)
test_dataset = TensorDataset(test_dataset_tensor, test_targets_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定義 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, feature_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # 將原始特徵映射到 d_model 維度
        self.input_layer = nn.Linear(feature_dim, d_model)
        
        # Transformer Encoder 層
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 最終輸出層
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, src):
        if src.dim() == 2:
            src = src.unsqueeze(-1)
        src = src.permute(0, 2, 1)

        # 將輸入映射到 d_model 維度
        src = self.input_layer(src)
        
        # 調整為 (seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)
        
        # 經過 Transformer Encoder
        output = self.transformer_encoder(src)
        
        # 取最後時間步的輸出
        output = output[-1, :, :]
        
        # 映射到單一值 (batch_size, 1)
        output = self.output_layer(output)
        return output

# 模型參數設定
feature_dim = 6  # 特徵數（風速、氣壓、溫度等）
d_model = 64  # 嵌入維度
nhead = 4  # 多頭注意力頭數
num_layers = 3  # Transformer Encoder 層數
dim_feedforward = 128  # 前饋網絡維度
dropout = 0.1  # Dropout 機率

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 建立模型
model = TransformerModel(feature_dim, d_model, nhead, num_layers, dim_feedforward, dropout).to(device)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 訓練模型
epochs = 500  # 訓練輪數
best_loss = 999999
early_stop = 9999
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        
        # 模型預測
        predictions = model(inputs.to(device)).squeeze(1)
        
        # 計算損失
        loss = criterion(predictions, targets.to(device))
        
        # 反向傳播和參數更新
        loss.backward()
        optimizer.step()
        
        # 累加損失
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
    # log metrics to wandb
    wandb.log({"train_loss": epoch_loss})

    # 評估模型
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        for batch in val_loader:
            inputs, targets = batch
            predictions = model(inputs.to(device)).squeeze(1)
            loss = criterion(predictions, targets.to(device))
            valid_loss += loss.item()
        print(f"Validation Loss: {valid_loss / len(val_loader):.4f}")
        # log metrics to wandb
        wandb.log({"val_loss": valid_loss})
        
    if valid_loss < best_loss:
        early_stop = 5
        best_loss = valid_loss
        # 儲存模型
        torch.save(model.state_dict(), model_path)
        print(f"saving predict_model with loss = {valid_loss:.5f}")

    else:
        early_stop -= 1

    if early_stop < 0:
        print("Early Stopping")
        break

model.eval()
test_pred = torch.empty((0))
test_correct = torch.empty((0))
with torch.no_grad():
    test_loss = 0
    for inputs, targets in test_loader:
        predictions = model(inputs).squeeze(1)
        loss = criterion(predictions, targets)
        test_loss += loss.item()
        test_pred = torch.cat((test_pred, predictions.cpu()), 0)
        test_correct = torch.cat((test_correct, targets.cpu()), 0)
    print(f"Validation Loss: {test_loss / len(val_loader):.4f}")

with open(predict_csv_path, "w") as f:

    f.write("pred, correct\n")
    for i in range(test_pred.size(0)):
        f.write(f"{test_pred[i]}, {test_correct[i]}\n")

wandb.finish()