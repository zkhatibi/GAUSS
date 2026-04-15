import numpy as np 
from sklearn.preprocessing import StandardScaler

def define_dict_w_mask(input_path):
    '''
    Reads in the input dataset and extract the SMILES, properties and the mask array. 
    It also normalizes the masked property array for later DNN trainig. 
    It produces the char to index and vice versa for later SMILES conversion to one-hot tensors. 
    
    :param input_path: input file containing SMILES and the properties. 
    '''
    smiles = []
    prop = []
    mask_data = []
    with open(input_path, "r") as file:
        for line in file:
            columns = line.split()
            smiles.append(columns[0])
            prop.append([float(x) for x in columns[1:]]) 
            if float(columns[1]) > -1e+5:
                mask_data.append(1)
            else:
                mask_data.append(0)
    prop = np.array(prop)
    mask = np.array(mask_data, dtype=bool)
    # Fit and transform per column
    prop_normalized = prop.copy().astype(float)  
    for col in range(prop.shape[1]):
        scaler = StandardScaler()
        scaler.fit(prop[mask, col].reshape(-1, 1))
        prop_normalized[mask, col] = scaler.transform(prop[mask, col].reshape(-1, 1)).ravel()
    #Defining char dictionary
    length = [len(smile_ent) for smile_ent in smiles]
    chars = sorted(list(set(''.join(smiles)))+[' '])
    nchars = len(chars) # number of unique chars 
    max_len = max(length) # max string length 
    char_to_idx = {ch:i  for i, ch in enumerate(chars)}    
    idx_to_char = {i:ch  for i, ch in enumerate(chars)}  
    print('SMILES and target values are extracted... ') 
    return smiles, char_to_idx, idx_to_char, nchars, max_len, prop_normalized, mask

def pad_smile(string, max_str_len):
    '''
    Adds padding to the string to make all tensors the same size.
    
    :param string: SMILES string.
    :param max_str_len: the length of the largest SMILES.
    '''
    if len(string) <= max_str_len:
            return string + " " * (max_str_len - len(string))
    else:
         print('Check the SMILES. The SMILES length is larger than the length cap')

def smiles_to_hot(smiles, seq_len, unique_chars, char_to_idx):
    no_of_examples = len(smiles)
    smiles = [pad_smile(i, seq_len) for i in smiles if pad_smile(i, seq_len)]

    X = np.zeros((no_of_examples, seq_len, unique_chars), dtype=np.float32) 
    # nr of smiles in the dataset, length of the largest string in the dataset, nr of chars in the dictionary 
    for smile_idx, smile in enumerate(smiles):
        for char_idx, char in enumerate(smile):
            try:
                X[smile_idx, char_idx, char_to_idx[char]] = 1
            except KeyError as e:
                print("Invalid SMILES - unassigned character in the string - ", smile)
                raise e
    return X

def hot_to_smiles(X, idx_to_char):
    '''
    Converts one-hot tensor into SMILES (one SMILES entry).

    :param X: SMILES string. 
    :param idx_to_char: dict for enabling the conversion. 
    '''
    string = []
    for i in X:
        # i is the each of the rows that stand for a char 
        try: 
            ind_char = np.where(i==1)[0][0]
            string.append(idx_to_char[ind_char])
        except Exception as e:
            # print('Invalid molecule -- unassigned character in the string...\nTry a new matrix')
            string = [] # discards the chars in the string so far as the generated matrix has a null char 
            break 
    if string:
        string = ''.join(string).replace(' ','')
    return string 