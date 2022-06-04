import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import get_train_test_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS
from datetime import datetime
#import tensorflow as tf
#from tensorflow import keras



data_folder = "./data/mvtec_anomaly_detection"
subset_name = "corner"
data_folder = os.path.join(data_folder, subset_name)

batch_size = 10
target_train_accuracy = 0.90
lr = 0.0400
epochs = 15
class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heatmap_thres = 0.7

train_loader, test_loader = get_train_test_loaders(
    root=data_folder, batch_size=batch_size, test_size=0.2, random_state=42
)

model = CustomVGG()

class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=lr)

model = train(
    train_loader, model, optimizer, criterion, epochs, device, target_train_accuracy
)


model_path = f"./weights/{subset_name}_model.h5"
torch.save(model, model_path)
model = torch.load(model_path, map_location=device)
#model = keras.models.load_model(model_path)
path = "data/mvtec_anomaly_detection/corner/test/bad/0.jpg"

predict_localize(
    model, test_loader, device, path, thres=heatmap_thres, n_samples=1, show_heatmap=False
)


