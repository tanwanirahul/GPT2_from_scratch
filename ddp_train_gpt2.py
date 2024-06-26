'''
A training script to be used when running in a distributed mode using torchrun.
usage: torchrun --standalone --nproc_per_node=<num_processes> ddp_train_gpt2.py

If there are multiple nodes involved in training, replace --standalone option with
rendezvous (rdzv_*) config properties. 
'''
import os
import sys
import torch
import torch.distributed
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from utils import FineWebDataLoader, LRScheduler, configure_adam_with_weight_decay
import time
from models import GPTConfig, GPT
from torch.distributed import init_process_group, destroy_process_group, all_reduce
from torch.nn.parallel import DistributedDataParallel as DDP
from hellaswag import render_example, iterate_examples, get_most_likely_row

def run_val(model, val_data_loader:FineWebDataLoader, device):
    '''
        Given the validation data loader `val_data_loader` and the `model`
        run the validation datapoints through the model and calculate the loss.
    '''
    model.eval()
    val_data_loader.reset()
    
    val_loss_accm = 0
    val_steps = 5
    with torch.no_grad():
        for _ in range(val_steps):
            x, y = val_data_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, loss = model(x, y)
            loss = loss / val_steps
            val_loss_accm += loss.detach()
    return val_loss_accm

def run_hellaswag_eval(model, rank, world_size, device):
    '''
        Run hellaswag evaluation.
    '''
    num_correct_norm = 0
    num_total = 0

    for i, example in enumerate(iterate_examples("val")):
        # process examples in the round robin fashion across num of processes.
        if i % world_size != rank:
            continue
        # convert the example into tokens and corresponding masks and labels.
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)

        # get the logits.
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, loss = model(tokens)
            
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
        
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        
        all_reduce(num_total, torch.distributed.ReduceOp.SUM)
        all_reduce(num_correct_norm, torch.distributed.ReduceOp.SUM)
        
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
        
    acc_norm = num_correct_norm / num_total
    return acc_norm

def setup_ddp():
    '''
        Initialize the process group and get the required environment details.
    '''
    init_process_group(backend="nccl")
    
    # get the execution environment details.
    world_size = int(os.environ.get("WORLD_SIZE"))
    rank = int(os.environ.get("RANK"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    
    return (world_size, rank, local_rank)

if __name__ == "__main__":
    '''
        Load the GPT2 params from the pre-trained HF model.  
    '''
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)

    # Create logs directory and define a path to logs.txt
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "logs.txt")

    # Is the script executed in ddp mode.
    ddp = int(os.environ.get('RANK', -1)) != -1
    if not ddp:
        print("This script is expecred to be executed in ddp mode using torchrun")
        sys.exit(0)
    
    # Check CUDA availability.
    assert torch.cuda.is_available(), "This script needs GPU devices to run. No CUDA devices found."

    ddp_world_size, ddp_rank, ddp_local_rank = setup_ddp()

    master_process = ddp_rank == 0

    # Pin the process to a particular GPU.
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)

    #device = "mps"
    prompt = "Hello, I'm a language model,"
    max_length = 30
    model_type = "gpt2"
    data_file = "data/input.txt"
    data_dir = "fine_web/edu_fineweb10B/"
    #device = "cpu"
    max_steps = 10
    val_steps = 5
    chkpoint_steps = 5
    hellaswag_steps = 5
    model_compilation = False
    chkpoint_save_enabled = True

    print(f"Using the device: {device}")
    # Define batch size and sequence length
    batch_size, seq_length = 16, 1024

    # Parameters for gradient scaling.
    total_batch_size = (163840 * ddp_world_size) # Increase this size based on available resources.
    #total_batch_size = 524288 # 2^19 (~0.5M as used to train GPT-2)
    
    
    assert total_batch_size % (batch_size * seq_length * ddp_world_size) == 0 #Total Batch Size is divisble by batch_size * seq_length
    grad_accm_steps = total_batch_size // (batch_size * seq_length * ddp_world_size)

    if master_process:
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
    if model_compilation:
        model = torch.compile(model)
    
    #Wrap the model into DDP.
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
    # Create a data loader.
    loader = FineWebDataLoader( data_dir, "train", batch_size=batch_size, seq_length=seq_length,
                                   rank=ddp_rank,
                                   num_processes=ddp_world_size,
                                   model_type=model_type)
    
    # Create a data loader.
    val_loader = FineWebDataLoader( data_dir, "val", batch_size=batch_size, seq_length=seq_length,
                                   rank=ddp_rank,
                                   num_processes=ddp_world_size,
                                   model_type=model_type)
    
    # Define the optimizer.
    # Optimizer tuning1: Define - betas, lr and eps based on GPT3 training details.
    # Optimizer tuning4: Configuring weigth decay for AdamW.
    optimizer = configure_adam_with_weight_decay(raw_model, weight_dacay=0.1, learning_rate=6e-4, device=device)
    
    lr_scheduler = LRScheduler.get()
    
    loop_start = time.time()
    for step in range(max_steps):
        
        last_step = step == (max_steps - 1)
        # Check val_steps to see  if we need to run validation.
        if step % val_steps == 0 or last_step:
            val_loss_accm = run_val(model, val_loader, device)
            all_reduce(val_loss_accm, op=torch.distributed.ReduceOp.AVG)
            if master_process:
                print(f"validation loss: {val_loss_accm.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accm.item():.4f}\n")

        # Check chkpoint_steps to see if the model need to be saved.
        if master_process and chkpoint_save_enabled \
            and step > 0 and (step % chkpoint_steps == 0 or last_step):

            chkpoint_path = os.path.join(logs_dir, f"model_{step:05d}.pt")
            chkpoint = {
                "model": raw_model.state_dict(),
                "config": raw_model.config,
                "step": step,
                "val_loss": val_loss_accm.item()
            }
            torch.save(chkpoint, chkpoint_path)

        # Check hellaswag steps to see if the hellaswag evaluation need to run.
        if (step % hellaswag_steps == 0 or last_step) and (not model_compilation):
            acc_norm = run_hellaswag_eval(model, ddp_rank, ddp_world_size, device)
            if master_process:
                print(f"Hellaswag accuracy: {acc_norm:.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")


        model.train()
        loss_accm = 0
        s = time.time()
        optimizer.zero_grad()
        # Optimization 5: implement gradient accumulation to account for small batches.
        for micro_step in range(grad_accm_steps):
            x, labels = loader.next_batch()
            x, labels = x.to(device=device), labels.to(device=device)
            
            ## read ddp's no_sync() context manager.
            # This instructs ddp to not sync gradients across all processes.
            # This is needed for gradient accumulation. Only at the last micro_step
            # we need the gradients to be syncronized. 
            model.require_backward_grad_sync = (micro_step == grad_accm_steps-1)
            # optimization2 - Use autocast for mixed precision computation. Some of
            # the operations will run with bfloat16 precision while others main continue to
            # run with float32 precision. Only works on latest CUDA chips - Ampere onwards.
            #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, loss = model(x, labels)
            loss = loss / grad_accm_steps
            loss_accm += loss.detach()
            loss.backward()
        
        # Accumulate the loss across all processes.
        all_reduce(loss_accm, op=torch.distributed.ReduceOp.AVG)
        # Optimizer tuning2: Clip gradient's norm to 1. Refer GPT3 paper training details.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Optimizer tuning3: Implement the cosine learning scheduler.
        lr = lr_scheduler(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        torch.cuda.synchronize() # Wait for the GPU to finish work.
        e = time.time()
        tokens_per_sec = (loader.B * loader.T * grad_accm_steps * loader.num_processes) / (e-s)
        if master_process:
            print(f"step: {step+1:3d} | loss: {loss_accm:9.6f} | lr: {lr:.4e} | norm: {norm:7.4f} | dt: {(e-s)*1000:.6f}ms | toks/sec: {tokens_per_sec:.5f}")
    loop_end = time.time()
    if master_process:
        print(f'\nTotal execution time: {(loop_end-loop_start) * 1000}')
    destroy_process_group()