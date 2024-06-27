import os
import numpy as np
import tiktoken
import torch
import math
import torch
import inspect

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
    
    
class FileDataLoaderWithDDP:
    def __init__(self, file_path, batch_size, seq_length, rank, num_processes, model_type="gpt2"):
        '''
            Initialize `DataLoader` parameters.
        '''
        self.B = batch_size
        self.T = seq_length
        self.rank = rank
        self.num_processes = num_processes

        self.current_position = rank * self.B * self.T

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

        self.current_position += (B*T*self.num_processes)

        # Reset if the next batch would be out of bounds.
        if self.current_position + (B*T*self.num_processes) + 1 > len(self.tokens):
            self.current_position = self.B * self.T * self.rank

        return x, labels

class FineWebDataLoader:
    def __init__(self,  data_dir, split, batch_size, seq_length, rank, num_processes, model_type="gpt2"):
        '''
            Loads the Fineweb dataset by reading through the shards file created
            by fineweb.py. 
            `data_dir` points to the directory containing the shards.
            `split` specifies the choice between train/validation split.
        '''
        assert split in ("train", "val"), f"split must be one of train/val."

        self.B = batch_size
        self.T = seq_length
        self.rank = rank
        self.num_processes = num_processes

        shards = os.listdir(path=data_dir)
        shards = sorted([shard for shard in shards if split in shard])
        assert len(shards) > 0, f"Did not find any shards for the split: {split}"

        # Shards containing the filepaths for the given split.        
        self.shards = [os.path.join(data_dir, shard) for shard in shards]

        if rank == 0:
            print(f"Found {len(self.shards)} shards for {split} split.")

        self.reset()

    def __load_tokens(self, shard_filename):
        '''
            loads the pytorch tensor containing the tokens in the shard specified by
            `shard_filename`.
        '''
        np_tokens = np.load(shard_filename)
        np_tokens = np_tokens.astype(np.int32)
        return torch.tensor(np_tokens, dtype=torch.long)
    
    def reset(self, shard=0):
        '''
            Resets the position of the DataLoader to the starting position in shard 0
            and loads the corresponding tokens.
        '''
        self.current_shard = shard
        self.tokens = self.__load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.rank

    def next_batch(self):
        '''
            Returns next batch of data from the data source.
        '''
        B, T, curr = self.B, self.T, self.current_position

        buff = self.tokens[curr:curr+(B*T)+1]
        x = buff[:-1].view(B,T)
        labels = buff[1:].view(B,T)

        self.current_position += (B*T*self.num_processes)

        # Reset if the next batch would be out of bounds.
        if self.current_position + (B*T*self.num_processes) + 1 > len(self.tokens):
            self.reset(shard=(self.current_shard + 1) % len(self.shards))

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
    

def configure_adam_with_weight_decay(model, weight_dacay, learning_rate, device):
    '''
        Returns the AdamW optimizer with Weight Decay configured for the 
        `model` parameters. 
    '''
    # all paramters that requires gradient computation.
    param_dict = {pn:p for pn, p in model.named_parameters() if p.requires_grad}
    
    # We will only apply parameter decay to tensors with dim>2. All Linear layers,
    # Embedding layers will be weight decayed. Layer Norms and Biases won't.
    params_to_decay = [p for n, p in param_dict.items() if p.dim() >= 2]
    params_not_to_decay = [p for n, p in param_dict.items() if p.dim() < 2]
    
    # Define paramter groups for Adam optimizer with corresponding weigth decays.
    optim_param_groups = [
         {"params": params_to_decay,"weight_dacay": weight_dacay},
         {"params": params_not_to_decay,"weight_dacay": 0.0},
    ]
    
    num_decay_params = sum(p.numel() for p in params_to_decay)
    num_non_decay_params = sum(p.numel() for p in params_not_to_decay)
    print(f"Decay Stats - Tensors: {len(params_to_decay)} with Parameters: {num_decay_params}")
    print(f"Non-Decay Stats - Tensors: {len(params_not_to_decay)} with Parameters: {num_non_decay_params}")

    # Check if the fused functionality is available.
    # Uses inspect module. Looks hacky!
    # FIXME: Exception checking for fused availability.
    #use_fused = ("cuda" in device) and ("fused" in inspect.signature(torch.optim.AdamW).parameters())
    use_fused = False

    # Create an optimizer with parameter groups.
    optimizer = torch.optim.AdamW(optim_param_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
    return optimizer


def get_device():
    '''
        Returns the device to be used for training / inference.
    '''
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        #return "mps"
        return "cpu"
    return "cpu"
