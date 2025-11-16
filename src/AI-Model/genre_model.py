import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNN_LSTM_genre(nn.Module):
    def __init__(self, num_classes, additional_features_dim, hidden_size):
        super(CNN_LSTM_genre, self).__init__()

        # Bloque CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),  # Reduce a (64, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),  # Reduce a (32, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),  # Reduce a (16, 16)
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2))  # Reduce a (8, 8)
        )

        # Bloque LSTM
        self.lstm = nn.LSTM(
            input_size=128 * 8 * 8 + additional_features_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
        )

        # Capa Final
        self.fc = nn.Linear(256, num_classes)


    def forward(self, x, additional_features):

        batch_size, seq_len, channels, height, width = x.size()

        # Primero aplicamos la CNN a cada fragmento de una secuencia de tres:
        print(f"Forma de x antes de la CNN: {x.size()}")
        x = x.view(seq_len * batch_size, channels, height, width)
        x = self.cnn(x)  # Bloque Cnn
        x = x.view(batch_size, seq_len, -1)
        print(f"Forma de x después de la CNN: {x.size()}")
        # Añado las caracteristicas adicionales
        x = torch.cat((x, additional_features), dim=-1)
        print(f"Forma de x después de concatenar características adicionales: {x.size()}")
        # LSTM
        lstm_out, _ = self.lstm(x)
        # Capa final
        out = self.fc(lstm_out[:, -1, :])
        return out

#PRUEBA CRNN
class CRNN(nn.Module):
    def __init__(self, num_classes, additional_features_dim, hidden_size):
        super(CRNN, self).__init__()
        
        # Bloque CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),  # Reduce a (64, 64)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),  # Reduce a (32, 32)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),  # Reduce a (16, 16)

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2))  # Reduce a (8, 8)
        )
        
        # Bloque RNN
        self.rnn = nn.GRU(
            input_size=128 * 8 * 8 +additional_features_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, additional_features):
        batch_size, seq_len, channels, height, width = x.size()  # x: (batch_size, 3, 1, 128, 128)
        print(f"Entrada inicial: {x.shape}")

        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, -1)
        print(f"Salida de la CNN: {x.shape}")
        print(f"Características adicionales: {additional_features.shape}")
        x = torch.cat((x, additional_features), dim=-1)
        print(f"Después de concatenar características adicionales: {x.shape}")
        x, _ = self.rnn(x)
        
        out = self.fc(x[:, -1, :])
        print(f"Salida final: {out.shape}")
        return out