# Benchmarking QLoRA+FSDP

## Exploring training performance across different hardware configurations

NB: These benchmarks were done in February and March 2024. The exact performance numbers will quickly go out of date but the general lessons may still be of interest. 

## Introduction

We recently announced our first public project, combining [FSDP and QLoRA](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) to enable training of 70B models on consumer GPUs. Our first [follow-on post](https://www.answer.ai/posts/2024-03-14-fsdp-qlora-deep-dive.html) went deep into the technical details involved in getting it working. In this note we’ll examine the performance of this new approach to evaluate when it will make the most difference and how you can get the most out of your hardware.


## Case Study: A Dual 3090 ‘Basement Rig’

Rather than starting with a table of results, let’s look at some illustrative examples on a single setup to get a feel for how different choices might affect the memory usage and speed of training a model. Everything in this section is benchmarked on Johno’s personal machine, which features two 3090s (without NVLink), 128GB CPU RAM and an older motherboard. The 3090s are power limited to 280W each.


### Starting at 7B

We’ll use the following command as a template, training on dummy data (so we can control the context length) and logging some stats to Weights and Biases for later comparisons:

```{.bash .code-overflow-wrap}
python train.py --model_name meta-llama/Llama-2-7b-hf --batch_size 1 --context_length 512 --train_type qlora --use_gradient_checkpointing True --reentrant_checkpointing True --use_cpu_offload False --log_to wandb --dataset dummy --dataset_samples 1024
```

We’re starting out with QLoRA, and by default the script uses FSDP (that is the headline feature after all) to split the model across both GPUs. So, doing some quick napkin math, with a 7 billion parameter model we’d expect 7 billion parameters x 4 bits/parameter / 2 GPUs = ~1.75GB of weights per GPU.

It’s actually about 3.72GiB (see `reserved_after_model_wrap`). There aren’t exactly 7 billion parameters, we keep some in full precision, there are the LoRA adapter weights, memory reservation overhead… and then once we begin training there are gradients and activations to keep track of too, intermediate values that need to be stored during certain computations, optimizer state for all of the trainable (LoRA) parameters… In total, the command above shows a peak memory usage of 4.98GiB during training.

Next let’s increase the context length from 512 tokens to 2048 (`--context_length 2048`). There are internal activations for each token in the sequence, so more tokens → more GPU memory used. In this case, the peak memory per GPU goes from 4.98GiB to 5.21GiB. Training also takes longer: 800 seconds vs 550.


| Train Type | Context Length | Peak Memory (GiB) | Time (s) |
| :--------: | :------------: | :---------------: | :------: |
|   QLoRA    |      512       |       4.98        |  1,082   |
|   QLoRA    |      2048      |       5.21        |  1,564   |

*Llama-2 7B with a batch size of one*

What if we weren’t using QLoRA? Keeping the weights in 16-bit precision and doing regular LoRA means we can skip the time spent dequantizing the base weights BUT we need more memory to store the weights (~7GB per GPU) and copying parameters from one GPU to another will be slower (since there is more data to transfer). On my system, the data transfer speed outweighs the gain from avoiding quantization, and the LoRA equivalents run slower than their QLoRA counterparts in this case:


| Train Type | Context Length | Peak Memory (GiB) | Time (s) |
| :--------: | :------------: | :---------------: | :------: |
|    LoRA    |      512       |       10.24       |  2,597   |
|    LoRA    |      2048      |       10.22       |  3,090   |

*Llama-2 7B with a batch size of one*


NB: While the reported peak reserved memory for both 512 and 2048 context length is roughly the same, the peak allocated memory is 8.28 GiB vs 9.16 GiB, respectively. Which matches our intuition that a smaller context length should use less memory.

None of these runs are close to using the 24GB of VRAM I have available, so let’s scale up the batch size to fill that up a little more:


| Train Type | Batch Size | Peak Memory (GiB) | Time (s) |
| :--------: | :--------: | :---------------: | :------: |
|   QLoRA    |     4      |       11.22       |   998    |
|   QLoRA    |     10     |       20.97       |   936    |
|    LoRA    |     4      |       16.14       |  1,366   |
|    LoRA    |     6      |       21.35       |  1,199   |

*Llama-2 7B with Context Length of 2048*

Using a larger batch size results in faster training overall. You still have to do the same amount of computation per sample, but running them through in batches lets you save time by transferring the weights back and forth fewer times in total. Notice also that by using less memory for model weights QLoRA enables a larger max batch size, giving it an extra speed advantage over the standard LoRA version.

Now, we mentioned transferring the weights between GPUs was slow on my machine, with an older motherboard and slow PCI lanes. Given that, we might reasonably ask if FSDP is even required in this case since we could fit the full model (quantized OR unquantized) in the VRAM of a single GPU. This is a valid point, and we can test it out by specifying `“ddp”` as the sharding strategy[^ddp], which keeps a full copy of the weights on each GPU:

[^ddp]: This is still using FSDP, but in distributed data parallel mode. Not DistributedDataParallel.

| Train Type | DDP  | Batch Size | Peak Memory (GiB) | Time (s) |
| :--------: | :--: | :--------: | :---------------: | :------: |
|   QLoRA    | True |     8      |       20.94       |   875    |
|    LoRA    | True |     4      |       22.04       |   881    |

*Llama-2 7B with Context Length of 2048*

In the QLoRA case, we now have the full (quantized) weights on each GPU, using more VRAM than with FSDP. Because we don’t have to transfer the weights between GPUs, only gradients, training finishes a little faster than the FSDP case. Even though we use a batch size of 8 vs 10. For LoRA, each GPU has 14GB* of weights and thus much less room for everything else, necessitating a lower batch size of 4 but still finishing much faster than the FSDP version.

We have our first lesson. If the model is small enough that the weights aren’t dominating your VRAM usage, you may be better off with DDP instead of FSDP. As we move to larger models, the larger batch sizes enabled by sharding the model across multiple GPUs will outweigh the communication overhead.


### What About CPU Offloading?

Now let’s jump up to a larger model: Yi 34B. Napkin math suggests with QLoRA+FSDP we should expect ~17GB of weights per GPU, leaving enough room on my 24GB cards for a batch size of 1 or 2 at most. But there’s another option: CPU offloading (`--use_cpu_offload true`) stores the weights in CPU RAM instead, loading them into each GPU a layer at a time as needed. This leaves the GPU RAM free for activations, gradients etc and allows us to use a batch size of 4 instead. In this example, the extra communication overhead of CPU offloading is offset by the higher batch size it enables and we end up with a slightly faster training run overall:


| Train Type | CPU Offload | Batch Size | Peak Memory (GiB) | Time (s) |
| :--------: | :---------: | :--------: | :---------------: | :------: |
|   QLoRA    |    False    |     2      |       23.05       |  5,041   |
|   QLoRA    |    True     |     4      |       22.98       |  4,830   |

*Yi 34B with Context Length of 2048*

In cases where you have faster interconnect between cards (NVLink, for example) the non-offloading case may win out, but it’s interesting how comparable these are - my assumption was that having the weights on the CPU and copying them over would be *far* slower. On a cloud machine with slower RAM and a wimpy CPU we did see dramatic slowdowns where CPU offloading was many times slower, so YMMV. But the fact that it works reasonably fast on my machine is encouraging, since it does spark the inevitable question: “**can we go bigger?**”


### Llama 70B

When I first tried loading and training a 70B model the script crashed and my hopes fell. Then I spotted an issue: my 128GB of CPU RAM was completely filling up right at the start of training. I created a 10GB swapfile, which is a part of the disk that is treated like RAM when the regular system RAM gets filled. This allowed the system to get over the initial spike and start training:


| Train Type | CPU Offload | Batch Size | Peak Memory (GiB) | Time (s) |
| :--------: | :---------: | :--------: | :---------------: | :------: |
|   QLoRA    |    True     |     2      |       14.92       |  11,795  |

*Llama-2 70B with Context Length of 2048*

It’s slower than the smaller models (nearly 10x slower than the 7B model, at nearly 50 seconds per batch) but that’s not bad considering that 70 BILLION parameters are copied to the GPUs each step! And with activation offloading (`--use_activation_cpu_offload True`) the total allocated memory is low enough that training on a 16GB GPU could be possible in theory.


## Case Study: A Dual 4090 “Budget Workstation”

We ran a subset of the tests on a dual 4090 “budget workstation” with 128GB of CPU RAM[^budget]. Like the 3090 case study, the 4090s don’t have NVLink. But both GPUs have full PCIe v4 x16 lanes which should reduce the FSDP transfer overhead. The 4090s peaked at 400 watts per card[^4090-power].

[^budget]: The total workstation cost is less than a single A6000 Ada. Hence a budget workstation.

[^4090-power]: Power usage peaked at 400 watts for the 7B and 34B models, and 375 watts for the 70B model.

### Llama-2 7B

At the 7 billion parameter scale, the maximum performance difference between LoRA and FSDP methods is ~10 percent.

| Train Type | CPU Offload |  DDP  | Batch Size | Peak Memory (GiB) | Time (s) |
| :--------: | :---------: | :---: | :--------: | :---------------: | :------: |
|    LoRA    |    False    | True  |     4      |       22.04       |   437    |
|    LoRA    |    False    | False |     6      |       21.35       |   481    |
|    LoRA    |    True     | False |     10     |       22.69       |   482    |
|   QLoRA    |    False    | True  |     8      |       20.94       |   450    |
|   QLoRA    |    False    | False |     10     |       20.97       |   466    |
|   QLoRA    |    True     | False |     12     |       22.38       |   464    |

*Llama-2 7B with Context Length of 2048*

This is encouraging, as there is only a small performance hit when trading maximum training speed verses maximum tokens. It also suggests that the slowdown due to using PCIe instead of NVLink is manageable when training large enough models.

### Yi 34B

With a full PCIe lanes and FSDP’s overlapping of compute and next layer transfers, there is almost no difference between QLoRA and QLoRA with CPU Offloading. The larger batch size is ~0.5 percent faster.

| Train Type | CPU Offload | Batch Size | Peak Memory (GiB) | Time (s) |
| :--------: | :---------: | :--------: | :---------------: | :------: |
|   QLoRA    |    False    |     2      |       23.05       |   2,072  |
|   QLoRA    |    True     |     4      |       22.98       |   2,061  |

*Yi 34B with Context Length of 2048*

### Llama-2 70B

Increasing from a 34B model to a 70B model shows near linear scaling, with a ~6 percent slowdown per sample.

| Train Type | CPU Offload | Batch Size | Peak Memory (GiB) | Time (s) |
| :--------: | :---------: | :--------: | :---------------: | :------: |
|   QLoRA    |    True     |     2      |       14.92       |  4,399   |

*Llama-2 70B with Context Length of 2048*

### Bonus: Mistral 7B

Mistral 7B v0.2 Base expanded the context window of the base 7B parameter model to 32K tokens. 24GB of memory per GPU isn't quite enough to finetune at the full context length even using QLoRA, but we can manage a respectable 24K tokens.

| Train Type | CPU Offload | Batch Size | Context Length | Peak Memory (GiB) | Time (s) |
| :--------: | :---------: | :--------: | :------------: | :---------------: | :------: |
|   QLoRA    |    True     |     12     |     2,048      |       22.54       |   483    |
|   QLoRA    |    True     |     1      |     24,576     |       22.54       |  7,809   |

*Mistral 7B v0.2 Base*

While the tokens per batch is the same at 24,576, increasing the context length from 2,048 to 24,576 reduces the training speed from 2,200 tokens/second to 1,615 tokens/second.

## Case Study: Conclusions

A priori, we expected the dual 4090s to be significantly faster than our dual 3090 test case, in part due to the increased generational performance but mostly due to the faster data transfer speed from full x16 PCIe lanes.

Our results confirmed this expectation, highlighting the importance of good multi-GPU interconnect. If you have two 3090s and a non-workstation motherboard, you’ll want NVLink. If you have two 4090s, you’ll want a workstation motherboard that can provide full x16 PCIe lanes to both GPUs.

These results are exciting if you already own a dual-GPU system, but now let’s take a step back and consider whether this still makes sense given the other hardware configurations available in the cloud.

## Recommendations for Different Hardware Configurations

Let’s consider a number of different hardware configurations and see which gives the best bang-per-buck performance for fine-tuning a 70B model. For each setup we’ve tried to find the fastest possible combination of settings capable of training on context length 2048 with an effective batch size of 32 (or the closest we could get).

|  Accelerator   | GPUs | CPU+Activation Offload | Batch Size | Time (s) | Ballpark Cost |
| :------------: | :--: | :--------------------: | :--------: | :------: | ------------- |
|   A5000 24GB   |  2   |          True          |     2      |  9,688   | $2.37 - $4.14 |
|   A5000 24GB   |  4   |         False          |     1      |  4,829   | $2.36 - $4.13 |
|   A5000 24GB   |  8   |         False          |     1      |  2,613   | $2.55 - $4.47 |
| A6000 Ada 48GB |  2   |         False          |     2      |  5,867   | $3.72 - $5.22 |
| A6000 Ada 48GB |  4   |         False          |     3      |  2,904   | $3.68 - $5.16 |
| A100 40GB SMX  |  2   |         False          |     1      |  3,277   | $3.28 - $3.75 |
| A100 40GB SMX  |  4   |         False          |     4      |  1,266   | $2.53 - $2.90 |
| A100 40GB SMX  |  8   |         False          |     4      |   672    | $2.69 - $3.08 |
| H100 80GB SXM  |  4   |         False          |     8      |   667    | $3.48 - $3.53 |

*Llama-2 70B QLoRA with Context Length of 2048 on Select Accelerators*

NB: Ballpark Cost is an estimated range of training 1,024 samples at a context length of 2,048. Prices from [Cloud GPUs](https://cloud-gpus.com/) are used. Exact numbers will vary by provider and depend on availability.

On a machine with four or eight A5000s, CPU offloading was slower despite allowing us to use double the batch size. This is a different outcome to the 2x3090 example on a 34B model, where CPU offloading had a slight edge. The difference likely comes down to the different transfer speeds CPU->GPU and GPU->GPU: copying parameters between GPUs with fast interconnect is faster than transferring them from the CPU RAM to all the GPUs on these machines.

It’s interesting to compare the time here of 16 minutes on eight A5000s with the dual 3090 example from earlier. The training is ~4.6X faster, but the machine is ~6X more expensive per hour[^per-hour]. And of course if you already own the 3090s then the longer wait might look like an even better deal.

[^per-hour]: 4X-10X depending on where you find your 3090s.

This trend holds for the rest of the examples too. Using a higher number of more powerful GPUs speeds things up as you’d expect, but also costs more, such that the total training cost ends up in the same range across the different setups we tested.

One final interesting thing we noticed when testing: for the lower-end configurations QLoRA + FSDP was either the fastest option or in some cases the only option, and training speed was bandwidth-bound. Once we moved to the H100 system with fast interconnect and 80GB memory per card, we finally hit the point where compute was the limiting factor. Changing the batch size from 8 to 12 made little difference, as did switching from QLoRA to LoRA - the extra time spent transferring data didn't matter since it was happening while the computation was being done, with the latter being the bottlekneck.


## Practical Guide for Optimal Training Speed

Here is a practical step-by-step guide to find the optimal FSDP training configuration which we also followed during the experiments above. We use QLoRA which already saves a significant amount of memory by reducing the model size via quantization, and a lot more by limiting the trainable parameters (~1-2%) with LoRA. We also use backward prefetching ([BACKWARD_PRE](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.BackwardPrefetch)) by default to overlap computation and communication as much as possible, which also comes with an increased memory usage. You can also experiment with other prefetch options: BACKWARD_POST or None to tradeoff memory and speed.

It is recommended to have at least two GPUs for this guide to make sense as it leverages FSDP sharding strategies.

Follow the steps below to find the optimal configuration for your own problem and hardware:


1. **Vanilla Start**:
    * We start with a batch size of 1, sequence length of 2048 (problem dependent) and disable all the memory saving options.
    * This configuration requires the most memory but potentially the fastest/cheapest one.
    * This will use DDP (Distributed Data Parallel).

2. **Try [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html#torch-utils-checkpoint)**:
    * Next, we can try gradient checkpointing to save memory.
    * Gradient checkpointing is a technique that allows the model to avoid storing intermediate activations during the backward pass by recomputing them.

3. **Try [SHARD_GRAD_OP](https://pytorch.org/docs/stable/checkpoint.html#torch-utils-checkpoint)**:
    * If DDP with gradient checkpointing didn’t work we can try SHARD_GRAD_OP[^shard-grad] next.
    * Shard-grad-op is a technique that allows the model to split the gradients and optimizer states across multiple GPUs.
    * This can reduce memory usage on each GPU, but it can also increase communication overhead and training time.
    * You can first try without gradient checkpointing and see if it trains without OOM. If not you can set it to true as well.

4. **Try [FULL_SHARD](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy)**:
    * If SHARD_GRAD_OP with gradient checkpointing didn’t work we can try FULL_SHARD[^full-shard] next.
    * Full-sharding is a technique that allows the model to split the model parameters, gradients and optimizer states across multiple GPUs.
    * This can significantly reduce memory usage on each GPU, but it can also increase communication overhead and training time.
    * Similarly, you can first try without gradient checkpointing and see if it trains without OOM. If not you can set it to true as well.

5. **Try CPU Offloading**:
    * If FULL_SHARD with gradient checkpointing didn’t work we can try cpu offloading next.
    * FSDP’s CPU Offloading moves model parameters and gradients to the CPU when they are not involved in computation.
    * This can reduce memory usage on the GPU, but it can also increase training time due to transfers between GPU and CPU.
    * At this point you’ve so far tried both full sharding and gradient checkpointing but still faced OOM issues.

6. **Try [Activation offloading](https://github.com/pytorch/pytorch/blob/2e02e1efad957b86dbcc5b64748e03acfb8d330c/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py#L173)**:
    * Activation offloading is a technique that allows the model to move some activations from the GPU to the CPU, and transfer them back to the GPU when needed.
    * This will reduce memory usage on the GPU, increase memory usage on the CPU and have additional transfers between CPU and GPU.

[^shard-grad]: If using multi-node training you can use _HYBRID_SHARD_ZERO2 (--sharding_strategy hybrid_shard_grad_op) to apply SHARD_GRAD_OP strategy within a node and replicate it across nodes.
[^full-shard]: If using multi-node training you can use HYBRID_SHARD (--sharding_strategy hybrid_full_shard) to apply FULL_SHARD strategy within a node and replicate it across nodes.

If you are still facing out-of-memory errors after trying all the steps above then you might need to reduce the sequence length if your task allows, find more GPUs or find GPUs with more memory, and repeat the steps again.

Once a setup that can train with a batch size of 1 is found, it is recommended to increase the batch size leaving some GPU memory free to avoid memory thrashing. This can help with training speed and avoid out-of-memory errors.

After finding the optimal configuration you can give the next step command a try with a higher batch size and see if it increases the throughput and reduces the training time. For example, imagine you are able to train using DDP (step 1). You can also try with gradient checkpointing (step 2) with a larger batch size. There is a chance that this might increase the overall throughput compared to not using gradient checkpointing and result in a faster training.

## Final Thoughts

Benchmarking is always complicated: hardware varies between providers, different versions of different libraries introduce hidden optimizations or bottlenecks, and subtle differences can cause dramatic speedups.

In this post we’ve tried to give recommendations for common use-cases which we hope will be useful in informing further experimentation, especially as FSDP+QLoRA support is added to more frameworks and the community explores this frontier further. We've also shown just how many more options there are for fine-tuning these large models now that we have these techniques at our disposal.

Additional References:

* [https://pytorch.org/docs/stable/fsdp.html](https://pytorch.org/docs/stable/fsdp.html)
* [https://medium.com/pytorch/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff](https://medium.com/pytorch/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff)
