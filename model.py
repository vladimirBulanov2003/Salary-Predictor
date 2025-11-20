

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from data_module_creation import FinalDataModule
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class Encoders_for_texts_with_multiple_convs(nn.Module):
  def __init__(self, embedding_matrix, PAD_IX = 1, hid_size=300, out_chanels = 15, list_of_windows = [5]):
    super().__init__()
    self.PAD_IX = PAD_IX
    self.embed = embedding_matrix
    self.list_of_convs = nn.ModuleList([ nn.Conv1d(hid_size, out_chanels, window) for window in list_of_windows])
    self.bact_norm = nn.BatchNorm1d(out_chanels)
    self.list_of_windows = list_of_windows

  def forward(self, input_idx):
    outputs_of_convs = []
    for window, conv in zip(self.list_of_windows, self.list_of_convs):
        
        input_idx_copy = input_idx
        if input_idx.shape[1] < window:
          pad_size = window - input_idx.shape[1]
          input_idx_copy = F.pad(input_idx, (0, pad_size), "constant", np.int32(self.PAD_IX))

        embedings = self.embed(input_idx_copy)
        embedings = torch.permute(embedings , (0, 2, 1))
        conv_output = F.relu(conv(embedings))
        conv_output = torch.permute(conv_output, (0,2,1))
        pooled_output = torch.sum(conv_output*torch.softmax(conv_output, dim = 1), dim = 1)
        
        outputs_of_convs.append(pooled_output)

    outputs_of_convs = torch.cat(outputs_of_convs, dim = 1)

    return outputs_of_convs
  


class  Encoders_for_texts(nn.Module):
   def __init__(self,  embedding_matrix, PAD_IX = 1, hid_size= 300, out_chanels = 15, window_size = 5):
    super().__init__()
    self.PAD_IX = PAD_IX
    self.window_sie = 5
    self.embed = embedding_matrix
    self.conv = nn.Conv1d(hid_size, out_chanels, window_size)
    self.bact_norm = nn.BatchNorm1d(out_chanels)
  
   def forward(self, input_idx):
  
      if input_idx.shape[1] < self.window_sie:
          pad_size = self.window_size - input_idx.shape[1]
          input_idx = F.pad(input_idx, (0, pad_size), "constant", np.int32(self.PAD_IX))

      embedings = self.embed(input_idx)
      embedings = torch.permute(embedings , (0, 2, 1))
      conv_output = F.relu(self.conv(embedings))
      conv_output = torch.max(conv_output, dim = -1)[0]
      return conv_output


class Encoders_for_categorical_feature(nn.Module):
  def __init__(self,  n_cat_features, hidden_dim = 64):
    super().__init__()
    self.linear_layer = nn.Linear(n_cat_features, hidden_dim)

  def forward(self, input_idx):
    return self.linear_layer(input_idx)


class SalaryPredictor(nn.Module):
    def __init__(self, n_cat_features, embedding_matrix, hid_size=300 , out_chanels = 15):
        super().__init__()

        embedding_matrix = nn.Embedding.from_pretrained(embedding_matrix, freeze= False)
        self.title_encoder = Encoders_for_texts(embedding_matrix = embedding_matrix , hid_size = 50, out_chanels = 15)
        self.list_of_windows = [4,5,6]
        self.description_encoder = Encoders_for_texts_with_multiple_convs(embedding_matrix = embedding_matrix, hid_size = 50, out_chanels = 15, list_of_windows =  self.list_of_windows)
        self.categorical_encoder = Encoders_for_categorical_feature(n_cat_features, hid_size)
        self.linear = nn.Linear(hid_size + (1 + len(self.list_of_windows))*out_chanels, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, batch):
    
        title_encoder_output = self.title_encoder(batch["Title"])
        description_encoder_output = self.description_encoder(batch["FullDescription"])
        categorical_encoder = self.categorical_encoder(batch["Categorical"])
        concat_outputs = torch.cat( (title_encoder_output, description_encoder_output, categorical_encoder), dim = -1)
        concat_outputs = self.dropout(concat_outputs)
        final_output = torch.log1p(self.linear(concat_outputs))
        return final_output[:,0]
    

class lightning_module(L.LightningModule):
   
    def __init__(self, n_cat_features, embedding_matrix):
      super().__init__()
      self.salary_predictor = SalaryPredictor(n_cat_features, embedding_matrix)

    def forward(self, batch):
       return  self.salary_predictor(batch)

    def training_step(self, batch):
      
      model_output = self(batch)
      loss = F.mse_loss(model_output, batch["Log1pSalary"])
      self.log("train_loss", loss)
      return loss 
    
    def validation_step(self, batch):
      model_output = self(batch)
      loss = F.mse_loss(model_output, batch["Log1pSalary"])
      self.log("val_loss", loss)
      return loss 
    
    def configure_optimizers(self):
       optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) 
       return optimizer


data = pd.read_csv("Train_rev1.csv", index_col=None)
data_module = FinalDataModule(data=data)
data_module.setup()
embedding_matrix = torch.tensor(data_module.create_embedding_matrix(), dtype = torch.float32)
model = lightning_module(data_module.get_the_len_of_vectorizer(), embedding_matrix)


checkpoint_callback = ModelCheckpoint(
   dirpath="checkpoints",
   filename="best-checkpoint",
   save_top_k=1,
   verbose= True,
   monitor="val_loss",
   mode = "min"
)

stopping = EarlyStopping(monitor="val_loss", mode="min", patience=2)
logger = TensorBoardLogger("ligtning_logs", name = "surface")

trainer = Trainer(logger=logger ,max_epochs=15 , accelerator="mps", enable_progress_bar=True, callbacks=[checkpoint_callback, stopping])

train = data_module.train_dataloader()
val = data_module.test_dataloader()
trainer.fit(model, train, val)