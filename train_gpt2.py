from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from utils import FileDataLoader, LRScheduler, configure_adam_with_weight_decay
import time

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    '''
        Implememnts (multi head) Self Attention mechanism.
    '''
    def __init__(self, config):
        '''
            Initialize the attention parameters 
        '''
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1, config.block_size, config.block_size))
    
        # Setting a flag to indicate the residual weights need to be scaled.
        setattr(self.c_proj, "requires_residual_weight_scaling", True)

    def forward(self, x):
        '''
            Implementation of the Forward Pass for the self attention mechanism.
        '''
        # batch_size, sequence_length, embedding size.
        B, T, C = x.size()

        # Split the QKV into their own tensors.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # View k,q,v as if they have been prepared by each head separately. 
        # nh - no. of heads, hs - head size.
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(B, nh, T, hs)
        
        # attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) => (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side.

        # output projection.
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    '''
        Implements FeedForward portion of the Self Attention Block.
    '''
    def __init__(self, config: GPTConfig):
        '''
        Intializes 2 Linear Layers seperated by a GELU non-linearity in between.
        '''
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
        # Setting a flag to indicate the residual weights need to be scaled.
        setattr(self.c_proj, "requires_residual_weight_scaling", True)

    def forward(self, x):
        '''
            Forward Pass logic for the Feed Forward portion of the attention block.
        '''
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x

class Block(nn.Module):
    '''
        Implements a transformer block containing:
            - Self Attention
            - Feed Forward
            - Skip Connections
            - Layer Normalization.
    '''
    def __init__(self, config: GPTConfig):
        '''
            Initializes the block parameters: 
            LayerNorm
            Attention 
            MLP
        '''
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        '''
            Implements forward pass for the given batch of data `x`.
        '''
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        
class GPT(nn.Module):
    '''
        Implements GPT-2 Architecture.
    '''
    def __init__(self, config: GPTConfig):
        '''
            Given the config, initializes all the parameters as per GPT-2 architecture.
            The parameters names are choosen such that they match exactly with HF GPT-2
            `GPT2LMHeadModel` model.
        '''
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
                                        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
        # wte and lm_head layers share the weights.
        self.transformer.wte.weight = self.lm_head.weight

        # Apply weights initialization logic.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        '''
            Defines the weights initialization mechanism for the `module`.
        '''
        # weights and bias initialization logic for Linear layers.
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "requires_residual_weight_scaling"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # initialization for embedding layers.
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        
            

    def forward(self, batch_x, targets=None):
        '''
            Implements forward pass for the GPT model.
        '''
        B, T = batch_x.size()
        device = batch_x.device
        
        assert T <= self.config.block_size, f"Input sequence cannot be longer than max sequence length {self.config.block_size}."
        
        # Create a positional vector.
        pos = torch.arange(0, T, dtype=torch.long, device=device)
    
        # Create token and positional embeddings.
        tok_embd = self.transformer.wte(batch_x)
        pos_embd = self.transformer.wpe(pos)
        x = tok_embd + pos_embd
        
        # Step through the attention blocks.
        for block in self.transformer.h:
            x = block(x)
            
        # LayerNorm before the final FF LM head.
        x = self.transformer.ln_f(x)
        
        # Finally, pass the batch tokens to the LM_head. If targets are passed, calculate the loss.
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            #logits = self.lm_head(x[:,[-1],:])
            logits = self.lm_head(x)
            loss = None

        return logits, loss

    @classmethod
    def from_pretrained(cls,model_type, override_args={}):
        '''
            Loads the pre-trained weights of the GPT2 model from the HF's GPT2 implementation
            `GPT2LMHeadModel`
        '''
        # Model type must be from the given list.
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        # Only dropout related args could be overridden.
        assert all(k=="dropout" for k in override_args)
        
        
        from transformers import GPT2LMHeadModel
        model_config = {
            "gpt2": {"n_head": 12, "n_layer": 12, "n_embd": 768}, #124M Params.
            "gpt2-medium": {"n_head": 24, "n_layer": 16, "n_embd": 1024}, #350M Params.
            "gpt2-large": {"n_head": 36, "n_layer": 20, "n_embd": 1280}, #774M Params.
            "gpt2-xl": {"n_head": 48, "n_layer": 25, "n_embd": 1600}, #1558M Params.
        }[model_type]
        
        print(f"Loading config for the GPT model: {model_type}")
    
        # GPT2 (with Tiktoken tokenizer) uses Vocab-Size of 50257 and was trained with the sequence length of 1024.
        model_config["vocab_size"] = 50257
        model_config["block_size"] = 1024

        print(f"Fixing - Vocab Size: {model_config['vocab_size']}, Sequence Length: {model_config['block_size']}")

        # Create an instance of the GPT model based on defined model_config. 
        config = GPTConfig(**model_config)
        model = GPT(config)
        
        # Get the state_dict and the corresponding keys for the GPT model.
        sd = model.state_dict()
        #.attn.bias isn't the model parameter. We don't need to load this from the pre-trained weights.
        # as long as we have value for q, k, v loaded, .attn.bias is a simple function of the q, k.
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")] 
    
        
        # intialize a GPT2 model from hugging_face and load it's weigths.        
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()

        # Get the keys that need to be copied from the hf_model to the model.
        hf_sd_keys = [k for k in hf_sd.keys() if not (k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"))]
        
        # OpenAI uses 1D convolution in place of the LinearLayer which results in the matrix shape that require transposition.
        requires_transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys) == len(hf_sd_keys), f"Mismatch in keys. {len(sd_keys)} != {len(hf_sd_keys)}"
        
        # Copy the HF GPT model's weights to out own GPT model.
        for k in hf_sd_keys:
            if any(k.endswith(rt) for rt in requires_transpose):
                assert hf_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                assert hf_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])
                    
        return model

def get_device():
    '''
        Returns the device to be used for training / inference.
    '''
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_eval(model, prompt, batch_size, max_length, model_type):
    '''
        Runs the model in eval model and passes the prompt repeated `batch_size` times.
    '''
    model.eval()

    print(f"\n\nRunning evaluations using the model: {type(model)}")

    import tiktoken
    tokenizer = tiktoken.get_encoding(model_type)
    enc_tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(enc_tokens, dtype=torch.long) # shape: (8,)
    tokens = tokens.unsqueeze(0).repeat(batch_size, 1) # shape: (3, 8)

    x = tokens.to(device)
    
    while x.size(1) < max_length:
        with torch.no_grad():
            output = model(x)

            # Get Logits.
            if isinstance(model, GPT):
                logits = output[0]
            elif isinstance(model, GPT2LMHeadModel):
                logits = output.logits
            else:
                raise Exception("Invalid Model Type.")

            logits = logits[:, -1, :] #(B, vocab_size)
            probs = F.softmax(logits, dim=-1) #(Run softmax on vocab_dimensions)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            idx = torch.multinomial(topk_probs, 1)
            x_col = torch.gather(topk_indices, -1, idx)
            x = torch.cat((x, x_col), dim=1)

    print("\nResponse:\n")
    for i in range(batch_size):
        decoded = tokenizer.decode(x[i, :max_length].tolist())
        print(f">{decoded}")


if __name__ == "__main__":
    '''
        Load the GPT2 params from the pre-trained HF model.  
    '''
    device = get_device()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    #device = "mps"
    prompt = "Hello, I'm a language model,"
    batch_size = 3
    max_length = 30
    model_type = "gpt2"
    data_file = "data/input.txt"
    device = "cpu"
    max_steps = 10

    # Define batch size and sequence length
    batch_size, seq_length = 1, 1024

    # Parameters for gradient scaling.
    total_batch_size = 5120 # Increase this size based on available resources.
    
    assert total_batch_size % (batch_size * seq_length) == 0 #Total Batch Size is divisble by batch_size * seq_length
    grad_accm_steps = total_batch_size // (batch_size * seq_length)
    print(f"Desired batch size: {total_batch_size}; Required gradient accumulation steps: {grad_accm_steps}")
    
    # optimization1 : Set the lower precision for float32 multiplications.
    # This optimization only works for CUDA devices. For MPS, it worsens the performance.
    #torch.set_float32_matmul_precision("high")

    # Create a GPT2 model with random weights for the purposes of training.
    # optimization 5 - Use the vocab size of 50304 (a power of 2) instead of 50257.
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device=device)

    # optimization3 - Compile the model upfront and let torch perform optimizations.
    # Again, this only works for CUDA and for latest versions - V100, A100, H100.
    if device == "cuda":
        model = torch.compile(model)

    # Create a data loader.
    loader = FileDataLoader(data_file, batch_size=batch_size, seq_length=seq_length, model_type=model_type)
    
    # Define the optimizer.
    # Optimizer tuning1: Define - betas, lr and eps based on GPT3 training details.
    # Optimizer tuning4: Configuring weigth decay for AdamW.
    optimizer = configure_adam_with_weight_decay(model, weight_dacay=0.1, learning_rate=6e-4, device=device)
    
    lr_scheduler = LRScheduler.get()
    
    loop_start = time.time()
    for step in range(max_steps):
        loss_accm = 0
        s = time.time()
        optimizer.zero_grad()
        # Optimization 5: implement gradient accumulation to account for small batches.
        for micro_step in range(grad_accm_steps):
            x, labels = loader.next_batch()
            x, labels = x.to(device=device), labels.to(device=device)
            # optimization2 - Use autocast for mixed precision computation. Some of
            # the operations will run with bfloat16 precision while others main continue to
            # run with float32 precision. Only works on latest CUDA chips - Ampere onwards.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, labels)
            loss = loss / grad_accm_steps
            loss_accm += loss.detach()
            loss.backward()
        # Optimizer tuning2: Clip gradient's norm to 1. Refer GPT3 paper training details.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Optimizer tuning3: Implement the cosine learning scheduler.
        lr = lr_scheduler(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        e = time.time()
        tokens_per_sec = (loader.B * loader.T * grad_accm_steps) / (e-s)
        print(f"step: {step+1:3d} | loss: {loss_accm:9.6f} | lr: {lr:.4e} | norm: {norm:7.4f} | dt: {(e-s)*1000:.6f}ms | toks/sec: {tokens_per_sec:.5f}")
    loop_end = time.time()
    print(f'\nTotal execution time: {(loop_end-loop_start) * 1000}')