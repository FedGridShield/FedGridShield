import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(Block, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.linear(x)))

class MLP(nn.Module):
    def __init__(self, input_size=7, num_classes=5):
        super(MLP, self).__init__()
        self.blocks = nn.Sequential(
            Block(input_size, 64),  
            Block(64, 128, dropout=0.1),  
            Block(128, 128),  
            Block(128, 64, dropout=0.1),
        )
        self.head = nn.Linear(64, num_classes)  
        # Initialize weights
        self._init_weights()

    def forward(self, x):
        x = self.blocks(x)
        return self.head(x)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out.squeeze(0))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += self.shortcut(x.squeeze(0)) 
        return self.relu(out)

class ResNetTabular(nn.Module):
    def __init__(self, input_size=7, num_classes=5):
        super(ResNetTabular, self).__init__()
        self.layer1 = BasicBlock(input_size, 64)
        self.layer2 = BasicBlock(64, 128)
        self.layer3 = BasicBlock(128, 128)
        self.layer4 = BasicBlock(128, 64)
        self.fc = nn.Linear(64, num_classes)
        # Initialize weights
        self._init_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class TransformerTabular(nn.Module):
    def __init__(self, input_size=7, num_classes=5, d_model=128, num_heads=4, num_layers=2, dim_feedforward=256):
        super(TransformerTabular, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        x = self.embedding(x) 
        x = self.transformer_encoder(x) 
        x = x.squeeze(1) 
        return self.fc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)