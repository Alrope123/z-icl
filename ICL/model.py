# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM
# import bitsandbytes


class Model(object):

    def __init__(self, logger=None, out_dir=None, fp16=True, local_rank=-1):
        if logger is None:
            class Logger():
                def info(self, text):
                    print ("Logging from Model:\t", text)
            logger = Logger()

        self.logger = logger
        self.out_dir = out_dir
        self.fp16 = fp16
        self.local_rank = local_rank

        if self.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
            ws = 1
        else:  # distributed mode
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            ws = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
            torch.distributed.init_process_group(backend="nccl")
            n_gpu = 1

        self.n_gpu = n_gpu
        self.device = device
        if self.local_rank <= 0:
            logger.info("Setting up for local_rank=%d, world_size=%d" % (self.local_rank, ws))
        self.model_name = None
        self.model = None
        self.mode = None

    def __str__(self):
        text = "[Model]: "
        if self.model_name is None:
            text += "No model loaded yet"
        else:
            text += self.model_name
            if self.mode is None:
                text += " (no mode setted - try .train() or .eval()"
            else:
                text += " (%s mode)" % self.mode
        text += "\nusing device %s, %d gpus, local_rank=%d" % (self.device, self.n_gpu, self.local_rank)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def is_none(self):
        return self.model is None

    def train(self):
        self.model.train()
        self.mode = "train"

    def eval(self):
        self.model.eval()
        self.mode = "eval"

    def cuda(self):
        self.model.cuda()

    def to_device(self):
        self.model.to(self.device)

    def load(self, gpt="gpt2-large", use_int8=False):
        '''
        checkpoint can be either keyword of the model or path to the checkpoint file
        '''
        if gpt.startswith("gpt2"):
            model = AutoModelForCausalLM.from_pretrained(gpt)
        elif "gpt-j" in gpt:
            if use_int8:
                model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
                model = convert_model_to_int8_on_gpu(model, device='cuda')
            else:
                model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
        elif "neox" in gpt:
            if use_int8:
                model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
                model = convert_model_to_int8_on_gpu(model, device='cuda')
            else:
                model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
        elif gpt.startswith("opt"):
            model = AutoModelForCausalLM.from_pretrained("facebook/{}".format(gpt), torch_dtype=torch.float16)
        else:
            raise NotImplementedError()
        self.model_name = gpt
        self.model = model

    def save(self, step):
        if self.local_rank <= 0:
            model_state_dict = {key[7:] if key.startswith("module.") else key: value.cpu()
                                for key, value in self.model.state_dict().items()}
            torch.save(model_state_dict, os.path.join(self.out_dir, "model-{}.pt".format(step)))
            self.logger.info("Saving model parameters at step=%d" % step)


    def parallel(self):
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank)

    def do_inference(self, data, batch_size=1, verbose=False):
        dataloader = data.get_dataloader(batch_size, is_training=False)
        if verbose:
            dataloader = tqdm(dataloader)
        losses = []
        for batch in dataloader:
            input_ids=batch[0].cuda()
            attention_mask=batch[1].cuda()
            token_type_ids=batch[2].cuda()
            if len(batch)==3:
                labels=None
            else:
                labels=batch[3].cuda()
            with torch.no_grad():
                loss = self.run_model(input_ids, attention_mask, token_type_ids, labels=labels)
            losses += loss.cpu().detach().numpy().tolist()
        return losses

    def do_predict(self, data, batch_size=1, losses=None, verbose=False):
        if losses is None:
            losses = self.do_inference(data, batch_size, verbose=verbose)
        losses = np.array(losses)
        assert len(losses)==len(data)
        predictions = []
        for idx, dp in enumerate(data.metadata):
            curr_label_losses = [np.sum(losses[indices]) for indices in dp["indices"]]
            prediction_idx = sorted(enumerate(curr_label_losses), key=lambda x: x[1])[0][0]
            prediction = dp["options"][prediction_idx]
            predictions.append(prediction.strip())
        return predictions

    def run_model(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()

        if labels is None:
            labels = input_ids
        labels = labels[..., 1:].contiguous()
        label_mask = token_type_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]

        losses = losses.view(logits.size(0), logits.size(1)) * label_mask
        return torch.sum(losses, axis=1) / torch.sum(label_mask, axis=1)

def setup_fp16(model, optimizer):
    try:
        import apex
        from apex import amp
        apex.amp.register_half_function(torch, "einsum")
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    fp16_opt_level = "O1"
    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)
    return model, optimizer


def get_memory_footprint(model, return_buffers=True):
    """
    Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
    Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
    PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2
    Arguments:
        return_buffers (`bool`, *optional*, defaults to `True`):
            Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
            are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
            norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
    """
    mem = sum([param.nelement() * param.element_size() for param in model.parameters()])
    if return_buffers:
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        mem = mem + mem_bufs
    return mem


def ـreplace_linear_with_int8linear(model, modules_to_not_convert="lm_head"):
    for name, module in model.named_children():
        ـreplace_linear_with_int8linear(module, modules_to_not_convert)

        if isinstance(module, torch.nn.Linear) and name != modules_to_not_convert:
            model._modules[name] = QuantizedLinearInt8(linear_layer=module)
    return


class QuantizedLinearInt8(torch.nn.Module):
    '''
    A simple but effictive implmenetion of Int8 quantization for linear layers.
    The weights are quantized and stored as Int8, which saves ~50% of the gpu memory.
    During the forwared pass, the weights are de-quantized back to fp16 to do multiplication.
    Pros:
        - saves ~50% of the gpu memory
        - accurate quantization because only the weights are quantized, and the weights don't suffer
            from the "outliers" issue mentioned in the LLM.int8 paper; only the activations do.
        - high precision results beacuse the multiplication is done in fp16
        - much faster than LLM.int8
    Cons:
        - a bit slower because of the added computation of dequantization in each forward pass. In practice, the slowdown
            is not large because in the generation application, gpu utilization is not very high. 
    '''
    def __init__(self, linear_layer):
        super().__init__()
        self.bias = linear_layer.bias

        weight_bit_width = 8
        weight = linear_layer.weight

        self.weight_scale = torch.nn.Parameter(
            (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half(),
        )
        # print(self.weight_scale.max().item(), self.weight_scale.min().item(), self.weight_scale.mean().item())
        # if self.weight_scale.max().item() > 0.002:
            # print(self.weight_scale.max().item())
        self.weight = torch.nn.Parameter(
            torch.round(weight.float() / self.weight_scale[:, None]).char(),
            requires_grad=False
            )

    def forward(self, x):
        weight = self.weight.half() * self.weight_scale[:, None]
        return torch.nn.functional.linear(x, weight, self.bias)


def convert_model_to_int8_on_gpu(model, device):
    """
    Quantize a model to int8 and move it to GPU using a simple method.
    """
    if 'cuda' not in device:
        raise ValueError(f"Target device should be a gpu. Device {device} is not supported")

    model.half()

    memory_before_quantization = get_memory_footprint(model)  # without lm_head

    ـreplace_linear_with_int8linear(model)  # replace `Linear` with `QuantizedLinearInt8`

    model.to(device=device)
    memory_after_quantization = get_memory_footprint(model)  # without lm_head

    saving = round(100 * memory_after_quantization/memory_before_quantization)
    memory_before_quantization = round(memory_before_quantization / 2**30, 2)  # rounding for printing
    memory_after_quantization = round(memory_after_quantization / 2**30, 2)  # rounding for printing

    print(f'Quantization memory - before: {memory_before_quantization} GB, after: {memory_after_quantization} GB ({saving}% of the size before)')
    return model
