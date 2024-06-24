import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from utils import FileDataLoader, LRScheduler, configure_adam_with_weight_decay, get_device
import time
from models import GPTConfig, GPT


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
    #device = "cpu"
    max_steps = 10

    print(f"Using the device: {device}")
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
        print(f"Model compilation begins!")
        model = torch.compile(model)
        print(f"Done compiling the model.")

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
            #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
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