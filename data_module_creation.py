import lightning.pytorch as pl
from torch.utils.data import  DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import nltk
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class MyDataset(Dataset):
  
    def __init__(self, data):
        self.data = data
        self.TARGET_COLUMN = "Log1pSalary"
        self.categorical_columns = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):

        chosen_element = self.data.iloc[idx]
        return {
            "Title": chosen_element["Title"],
            "FullDescription": chosen_element["FullDescription"],
            "Categorical": chosen_element[self.categorical_columns],
            "Target": chosen_element[self.TARGET_COLUMN]

        }
    
class FinalDataModule(pl.LightningDataModule):

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.token_counts = Counter()
        self.categorical_vectorizer = DictVectorizer(dtype=np.float32, sparse=False)


    def counts(self, text : str):
        for element in text.split():
            self.token_counts[element] += 1

    
    def setup(self):

        data = pd.read_csv("Train_rev1.csv", index_col=None)
        data['Log1pSalary'] = np.log1p(data['SalaryNormalized']).astype('float32')

        categorical_columns = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]

        data[categorical_columns] = data[categorical_columns].fillna('NaN')

        tokenizer = nltk.tokenize.WordPunctTokenizer()
        data["FullDescription"] = data["FullDescription"].apply(lambda text: " ".join(tokenizer.tokenize(text)).lower())
        data["Title"] = data["Title"].apply(lambda text: " ".join(tokenizer.tokenize(str(text))).lower())
        data["Title"].apply(self.counts)
        data["FullDescription"].apply(self.counts)


        tokens = [key for key, value in self.token_counts.items() if value >= 8]

        UNK, PAD = "UNK", "PAD"
        self.tokens = [UNK, PAD] + tokens
        self.token_to_id = dict([(element, index) for index, element in enumerate(self.tokens)])

        top_companies, _ = zip(*Counter(data['Company']).most_common(1000))
        recognized_companies = set(top_companies)
        data["Company"] = data["Company"].apply(lambda comp: comp if comp in recognized_companies else "Other")

        self.categorical_vectorizer.fit(data[categorical_columns].apply(dict, axis=1))
        self.data_train, self.data_test = train_test_split(data, test_size=0.2, random_state=42)
    

    def as_matrix(self,sequences, max_len=None):

        UNK_IX, PAD_IX = map(self.token_to_id.get, ["UNK", "PAD"])
        if isinstance(sequences[0], str):
            sequences = list(map(str.split, sequences))

        max_len = min(max(map(len, sequences)), max_len or float('inf'))

        matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
        for i,seq in enumerate(sequences):
            row_ix = [self.token_to_id.get(word, UNK_IX) for word in seq[:max_len]]
            matrix[i, :len(row_ix)] = row_ix

        return matrix
    
    def to_tensors(self, batch, device = "mps"):

        batch_tensors = dict()
        for key, arr in batch.items():
            if key in ["FullDescription", "Title"]:
                batch_tensors[key] = torch.tensor(arr, device=device, dtype=torch.int64)
            else:
                batch_tensors[key] = torch.tensor(arr, device=device)
        return batch_tensors

    def make_batch(self, data):

        title = [element["Title"] for element in data] 
        description = [element["FullDescription"] for element in data] 
        categorical = [dict(element["Categorical"]) for element in data]
        target = [element["Target"] for element in data]

        batch = {}

        batch["Title"] = self.as_matrix(title)
        batch["FullDescription"] = self.as_matrix(description)
        batch['Categorical'] = self.categorical_vectorizer.transform(categorical)
        batch["Log1pSalary"] = target

        return self.to_tensors(batch)
    

    def create_embedding_matrix(self):

        embeddings_vectors = {}
        f = open('glove.6B.50d.txt', encoding='utf-8')
        for line in tqdm(f):
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:],dtype = 'float32')
            embeddings_vectors[word] = coef

        embedding_matrix = np.zeros((len(self.tokens), 50))
        for word, i in self.token_to_id.items():
            embedding_vector = embeddings_vectors.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_matrix[1]= np.random.rand(50)
        return embedding_matrix
            

    def train_dataloader(self):
        return DataLoader(dataset = MyDataset(self.data_train), 
                          batch_size=64, 
                          shuffle=True, 
                          collate_fn= self.make_batch,
                          )
    

    def test_dataloader(self):
        return DataLoader(dataset = MyDataset(self.data_test), 
                          batch_size=64, 
                          shuffle= False, 
                          collate_fn= self.make_batch,
                          )
    
    def get_the_len_of_array_with_tokens(self):
        return len(self.tokens)
    
    def get_the_len_of_vectorizer(self):
        return len(self.categorical_vectorizer.vocabulary_)

    

data = pd.read_csv("Train_rev1.csv", index_col=None)
data_module = FinalDataModule(data= data)
data_module.setup()

