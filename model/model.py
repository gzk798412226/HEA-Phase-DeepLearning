import torch
import torch.nn as nn

class HEAModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 256, 128], output_size=85, dropout_rate=0.2):
        super(HEAModel, self).__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # 添加输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        # 创建顺序模型
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class HEATrainer:
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # 前向传播
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs=100, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        return train_losses, val_losses 