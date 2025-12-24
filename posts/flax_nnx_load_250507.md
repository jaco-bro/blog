## Loading Safetensors in NNX: A 700x Speedup

I've been playing with [NNX](https://flax.readthedocs.io/) lately, and it feels like someone finally swept away the ceremony that comes with defining models in JAX. The API maps almost 1:1 to what you'd write in Torch or MLX.

But when I tried wiring up a [TTS model](https://github.com/jaco-bro/diajax) in NNX, I ran smack into the question: *how do I load pretrained model weights from HuggingFace into this thing?* 

Since I couldn’t find any clear example on the web, I'm posting here what worked for me:

### Step 1: Trace shapes (without any FLOPs)

```python
shape_only_call = nnx.eval_shape(lambda: MyLLM(config))
```

### Step 2: Split into graphdef & state

```python
graphdef, state = nnx.split(shape_only_call)
```

### Step 3: Turn state into a flattened dict

```python
flat_state = dict(state.flat_state())
```

### Step 4: Load and assign weights

```python
for key, arr in tensors.items():
    path = tuple(key.split("."))
    flat_state[path].value = nnx.Param(jnp.array(arr))
```

### Step 5: Merge back into a model

```python
loaded_model = nnx.merge(graphdef, nnx.State.from_flat_path(flat_state))
```

### Step 6: Inference

```python
output = loaded_model(sample_input)
```

If you naively instantiate an NNX model like below, you are paying for a full round of parameter initialization that you immediately throw away.

<details><summary>Click to expand code</summary><pre>
tic = time.perf_counter()
<span style="color:red;">graphdef, state = nnx.split(model_cls(config, rngs=nnx.Rngs(0)))</span>
state = dict(state.flat_state())
for fpath in glob(f"{model_name}/model*.safetensors"):
    for path, val in ((k.replace("norm.weight", "norm.scale").replace("proj.weight", "proj.kernel").replace("mlp.weight", "mlp.kernel").replace("lm_head.weight", "lm_head.kernel").replace("embed_tokens.weight", "embed_tokens.embedding"), nnx.Param(jnp.array(v).T) if k.endswith('proj.weight') or k.endswith('mlp.weight') or k.endswith('lm_head.weight') else nnx.Param(jnp.array(v))) for k, v in load_file(fpath).items()):
        path_tuple = tuple(int(part) if part.isdigit() else part for part in path.split('.'))
        if path_tuple in state:
            state[path_tuple].value = val
        else:
            print(f'{path_tuple} missing')
model = nnx.merge(graphdef, nnx.State.from_flat_path(state))
dtype = eval(f'jnp.{config.torch_dtype}')
model.set_attributes(dtype=dtype, param_dtype=dtype)
elapsed = time.perf_counter() - tic 
print(f'Model loaded in {elapsed:.2f} seconds')
</pre></details><br>

```
Model loaded in 691.94 seconds
```

On large models, this initial allocation is expensive. For a 1.6 billion parameter model, it took **691.94 seconds** just to start up!

But there's [a better way](https://web.archive.org/web/20250507063510/https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transforms.html#flax.nnx.eval_shape). In JAX, you can trace a function to get shapes without allocating memory for actual values:

<details><summary>Click to expand code</summary><pre>
tic = time.perf_counter()
<span style="color:red;"># graphdef, state = nnx.split(model_cls(config, rngs=nnx.Rngs(0)))</span> 
<span style="color:green;">graphdef, state = nnx.split(nnx.eval_shape(lambda: model_cls(config, rngs=nnx.Rngs(0))))</span>
state = dict(state.flat_state())
for fpath in glob(f"{model_name}/model*.safetensors"):
    for path, val in ((k.replace("norm.weight", "norm.scale").replace("proj.weight", "proj.kernel").replace("mlp.weight", "mlp.kernel").replace("lm_head.weight", "lm_head.kernel").replace("embed_tokens.weight", "embed_tokens.embedding"), nnx.Param(jnp.array(v).T) if k.endswith('proj.weight') or k.endswith('mlp.weight') or k.endswith('lm_head.weight') else nnx.Param(jnp.array(v))) for k, v in load_file(fpath).items()):
        path_tuple = tuple(int(part) if part.isdigit() else part for part in path.split('.'))
        if path_tuple in state:
            state[path_tuple].value = val
        else:
            print(f'{path_tuple} missing')
model = nnx.merge(graphdef, nnx.State.from_flat_path(state))
dtype = eval(f'jnp.{config.torch_dtype}')
model.set_attributes(dtype=dtype, param_dtype=dtype)
elapsed = time.perf_counter() - tic 
print(f'Model loaded in {elapsed:.2f} seconds')
</pre></details><br>

```
Model loaded in 0.92 seconds
```

This gets the startup time down to **0.92 seconds**, a 700x improvement.

And that's all there is to it.
