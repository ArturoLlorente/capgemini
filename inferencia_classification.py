import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import DTD
import torchvision.transforms as transforms

import pytorch_lightning as pl


# Especificar una seed para reproducibilidad de resultados
def set_seed(seed) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed = 42
set_seed(seed)


subscaling = 256
transform = transforms.Compose([
    transforms.Resize((subscaling, subscaling)), # Redimensionar las imágenes al valor especificado en subscaling
    transforms.ToTensor() # Convertir las imágenes a tensores y normalizar los valores de los pixeles a [0, 1]
])


# Descargar y cargar el dataset
dataset = DTD(root='data', download=True, transform=transform)

# Dividir el dataset en train, validation y test
total_samples = len(dataset)
n_train_samples = int(0.55 * total_samples)
n_val_samples = int(0.15 * total_samples)
n_test_samples = total_samples - n_train_samples - n_val_samples
train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train_samples, n_val_samples, n_test_samples])

test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# En pytorch lightning, los modelos tienen una serie de metodos que hay que definir.
# Definiendolos el entrenamiento sera automatico

class BaseModel(pl.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Sequential = None,
        loss_fn: torch.nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        optimizer_args: dict = None,
        ):
        super().__init__()
            
        self.backbone = backbone
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.loss = loss_fn
    
    def forward(self, x):
        return self.backbone(x)
    
    def compute_loss(self, prediction, truth):
        return self.loss(prediction, truth)
    
    def training_step(self, batch, batch_idx):
        train_loss = self.shared_step(batch, batch_idx)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        val_loss = self.shared_step(batch, batch_idx)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        test_loss = self.shared_step(batch, batch_idx)
        self.log('test_loss', test_loss)
        return test_loss
    
    def shared_step(self, batch, batch_idx):
        x, truth = batch
        prediction = self(x)
        loss = self.compute_loss(prediction, truth)
        return loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.backbone.parameters(), **self.optimizer_args)
        return optimizer
    
num_classes = 47
subscaling = 32
hidden_size = subscaling // 4

backbone = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    nn.Conv2d(64, 64, kernel_size=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((2, 2)),
    nn.Flatten(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(256, num_classes)
)
        
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam
optimizer_args = {'lr': 1e-3, 'weight_decay': 0.05, }
model = BaseModel(backbone=backbone, 
                  optimizer=optimizer, 
                  loss_fn=loss,
                  optimizer_args=optimizer_args)

ckpt = torch.load('./cnn_1.pth')
model.load_state_dict(ckpt)
model.eval()

# Inferencia
y_true = []
y_pred = []
for x, y in tqdm(test_dataloader):
    prediction = model(x)
    y_true.extend(y.cpu().numpy())
    y_pred.extend(prediction.argmax(1).cpu().numpy())
    
df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
df.to_csv('predictions.csv', index=False)

print('Inferencia finalizada')