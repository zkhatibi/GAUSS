import torch
from torch.utils.data import DataLoader, random_split, Dataset

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class CustomDatasetWithMask(CustomDataset):
    def __init__(self, X, y, z):
        super().__init__(X, y) 
        self.z = torch.tensor(z, dtype=torch.bool)

    def __getitem__(self, idx):
        X, y = super().__getitem__(idx)
        return X, y, self.z[idx]  

class DatasetSplitter:
    '''
    Takes in the SMILES and the properties that are a cobination of coordinates and first Kramers energy and split the 
    dataset into training and test using the percentile. 
    '''
    def __init__(self, data, prop, percentile, batch_size):
        self.data = data
        self.prop = prop
        self.percentile = percentile
        self.batch_size = batch_size

        # Initialize loaders
        self.train_loader, self.test_loader, self.full_loader = self.create_loaders()

    def create_loaders(self):
        # Wrap data in dataset
        dataset = CustomDataset(self.data, self.prop)

        # Compute split sizes
        train_size = int(self.percentile * len(dataset))
        test_size = len(dataset) - train_size

        torch.manual_seed(42)  # for reproducibility
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)

        # full dataset for visulization of results 
        full_dataset, _ = random_split(dataset, [len(dataset), 0])
        full_loader = DataLoader(full_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, full_loader
    

class MaskedDatasetSplitter(DatasetSplitter):
    '''
    Same as the DatasetSplitter but for a curated dataset that not all SMILES have an associated property, 
    Takes a bool tensor, named mask, and concatanate it with the dataset, later used for a semi-superwised learning (DNN).
    '''
    def __init__(self, data, prop, mask, percentile, batch_size):
        self.mask = mask 
        super().__init__(data, prop, percentile, batch_size)
        

    def create_loaders(self):
        dataset = CustomDatasetWithMask(self.data, self.prop, self.mask)
        train_size = int(self.percentile * len(dataset))
        test_size = len(dataset) - train_size

        torch.manual_seed(42)  # for reproducibility
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) # shuffle changes the batches in every epoch 
        test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)

        full_dataset, _ = random_split(dataset, [len(dataset), 0])
        full_loader = DataLoader(full_dataset, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, full_loader