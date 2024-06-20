import tiktoken
import torch
import math

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
    
    
    

class LRScheduler:
    '''
        Implements learning rate scheduler by factoring in warmup steps, 
        max_lr and min_lr constraints. GPT3 uses cosine scheduler for learning
        rate. Refer to the GPT3 paper for details.
    '''
    max_lr = 3e-4
    min_lr = max_lr * 0.1
    warampup_steps = 4
    max_decay_steps = 10

    @classmethod
    def get(cls):
        '''
            Returns LR scheduler.
        '''
        def lr_schedule(step):
            # Check for warmup steps.
            if step < cls.warampup_steps:
                return cls.max_lr * (step + 1) / cls.warampup_steps
            
            # Check if we have reached max_decay steps. If so, return min_lr.
            if step > cls.max_decay_steps:
                return cls.min_lr

            # In between, use cosine decay down to min learning rate.
            decay_ratio = (step - cls.warampup_steps) / (cls.max_decay_steps - cls.warampup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return cls.min_lr + coeff * (cls.max_lr - cls.min_lr)

        return lr_schedule