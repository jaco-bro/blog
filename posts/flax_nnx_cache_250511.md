## KV Caching in NNX

It took several iterations for me to get key-value (KV) caching right. I’m writing this post to share the lessons I learned along the way.

### Version 1. Simple Concatenate

I started out by passing the caches around as a list of key and value arrays. Each attention layer would concatenate new entries with the old, and the model would collect the results into a list and return it. It worked, but it was slow. Dynamic shapes meant XLA had to recompile at every step.

```python  
class Attention(nnx.Module):
    def __init__(self, config):
        # ...

    #@nnx.jit
    def __call__(self, x, mask):
        q, k, v = self.qkv_proj(x)
        if cache is not None:
            k = jnp.concatenate([cache[0], k], axis=2) 
            v = jnp.concatenate([cache[1], v], axis=2) 
        cache = (k, v)
        return flash_mha(q, k, v, mask), cache

class Qwen3Model(nnx.Module):
    def __init__(self, config):
        # ...

    #@nnx.jit
    def __call__(self, x, mask, caches):
        x = self.embed(x)
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, new_cache = layer(x, mask, caches[i])
            new_caches.append(new_cache)
        return self.norm(x), new_caches

for _ in range(max_new_tokens):  
    logits, caches = model(token, mask, caches)
```

- Prompt processing: 39.3 tokens/sec (22 tokens in 0.6s)
- Token generation: 3.1 tokens/sec (5 tokens in 1.6s)

### Version 2. In-Place Update

To avoid having to return new array shapes on every step, I next wrapped the cache arrays in a `Module` so they can be updated in place. This may have trimmed a bit of overhead, but the cache variables still had to grow with every new token.

```python
class Cache(nnx.Module):
    def __init__(self, dtype, batch_size, num_heads, max_len, head_dim, k=None, v=None):
        self.max_len = max_len
        self.k = nnx.Variable(jnp.empty((batch_size, num_heads, 0, head_dim), dtype=dtype)) if k is None else nnx.Variable(k)
        self.v = nnx.Variable(jnp.empty((batch_size, num_heads, 0, head_dim), dtype=dtype)) if v is None else nnx.Variable(v)

    #@nnx.jit
    def __call__(self, k, v):
        self.k.value = jnp.concat([self.k.value, k], axis=2)
        self.v.value = jnp.concat([self.v.value, v], axis=2)
        return self.k.value, self.v.value

class Attention(nnx.Module):
    def __init__(self, config):
        init(config)

    #@nnx.jit
    def __call__(self, x, mask):
        q, k, v = self.qkv_proj(x)
        # if cache is not None:
        #     k = jnp.concatenate([cache[0], k], axis=2) 
        #     v = jnp.concatenate([cache[1], v], axis=2) 
        # cache = (k, v)
        # return flash_mha(q, k, v), cache
        k, v = cache(k, v)
        return flash_mha(q, k, v, mask), cache

class Qwen3Model(nnx.Module):
    def __init__(self, config):
        # ...

    #@nnx.jit
    def __call__(self, x, mask, caches):
        x = self.embed(x)
        # new_caches = []
        for i, layer in enumerate(self.layers):
            # x, new_cache = layer(x, caches[i])
            # new_caches.append(new_cache)
            x = layer(x, mask, caches[i])
        # return self.norm(x), new_caches
        return self.norm(x)

for _ in range(max_new_tokens):  
    # logits, caches = model(token, caches)
    logits = model(token, mask, caches)
```

- Prompt processing: 41.8 tokens/sec (22 tokens in 0.5s)
- Token generation: 5.2 tokens/sec (100 tokens in 19.0s)

### Version 3. Preallocated Buffers

Preallocating fixed-size cache buffer allowed jit to finally kick in and fuse operations together without recompiles every step.

```python
class Cache(nnx.Module):
    def __init__(self, dtype, batch_size, num_heads, max_len, head_dim, k=None, v=None):
        self.max_len = max_len
        self.k = nnx.Variable(jnp.zeros((batch_size, num_heads, max_len, head_dim), dtype=dtype)) if k is None else nnx.Variable(k)
        self.v = nnx.Variable(jnp.zeros((batch_size, num_heads, max_len, head_dim), dtype=dtype)) if v is None else nnx.Variable(v)

    @nnx.jit
    def __call__(self, k, v):
        self.k.value = jnp.concat([self.k.value, k], axis=2)[:,:,-self.max_len:,:]
        self.v.value = jnp.concat([self.v.value, v], axis=2)[:,:,-self.max_len:,:]
        return self.k.value, self.v.value

for _ in range(max_new_tokens):  
    logits = model(token, mask, caches)
    mask = jnp.concat([mask[1:], jnp.zeros((1,))])[-max_len:]
```

- Prompt processing: 38.1 tokens/sec (22 tokens in 0.6s)
- Token generation: 35.4 tokens/sec (100 tokens in 2.8s)

### Version 4. Full Fusion with `scan`

It also allowed the remaining Python loop to be replaced with `scan`:

```python
carry = (token, mask, caches)
_, result = nnx.scan(scan_step, in_axes=(nnx.Carry,), out_axes=(nnx.Carry, 1), length=max_new_tokens)(carry)
```

- Prompt processing: 35.3 tokens/sec (22 tokens in 0.6s)
- Token generation: 72.9 tokens/sec (100 tokens in 1.4s)

### Closing Notes

Hopefully this saves someone a few of the steps I had to stumble through. You can find the full implementation here: [github.com/jaco-bro/nnx-lm](https://pypi.org/project/nnx-lm)
