import tiktoken
import torch

class FileDataLoader:
    def __init__(self, file_path, batch_size, seq_length, model_type="gpt2"):
        '''
            Initialize `DataLoader` parameters.
        '''
        self.B = batch_size
        self.T = seq_length
        self.current_position = 0

        with open(file_path, 'r') as f:
            data = f.read()

        # Tokenize the data and place it on the device.
        tokenizer = tiktoken.get_encoding(model_type)
        tokens = tokenizer.encode(data)
        self.tokens = torch.tensor(tokens)
        
        print(f"Loaded {len(self.tokens)} tokens.")
        print(f"1 epoch = {len(self.tokens) // (self.B * self.T)} steps.")

    
    def next_batch(self):
        '''
            Returns next batch of data from the data source.
        '''
        B, T, curr = self.B, self.T, self.current_position

        buff = self.tokens[curr:curr+(B*T)+1]
        x = buff[:-1].view(B,T)
        labels = buff[1:].view(B,T)
        
        self.current_position += (B*T)
        
        # Reset if the next batch would be out of bounds.
        if self.current_position + (B*T) + 1 > len(self.tokens):
            self.current_position = 0
            
        return x, labels