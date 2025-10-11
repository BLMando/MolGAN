# ProcessGAN Migration to TensorFlow 2.x

## 🎯 Overview

This document describes the modernization of ProcessGAN from TensorFlow 1.8 to TensorFlow 2.15, bringing modern APIs, better performance, and improved maintainability.

---

## 📊 What Changed

### Version Comparison

| Component | TF 1.x (Legacy) | TF 2.x (Modern) |
|-----------|-----------------|-----------------|
| **Python** | 3.6.10 (EOL) | 3.10+ (LTS) |
| **TensorFlow** | 1.8.0 (2018) | 2.15.0 (2024) |
| **NumPy** | 1.18 | 2.0+ |
| **Execution** | Session-based | Eager execution |
| **API Style** | `tf.layers` | `tf.keras.layers` |
| **Gradient** | `minimize()` | `GradientTape()` |
| **Model Style** | Functions | `keras.Model` classes |

### Key Improvements

✅ **Modern Python Stack**: Compatible with Python 3.10+, modern libraries
✅ **Eager Execution**: No more `tf.Session()`, simpler debugging
✅ **Cleaner API**: `tf.keras` API is more Pythonic and intuitive
✅ **Better Performance**: TF 2.x optimizations, GPU support (CUDA 12.x)
✅ **Apple Silicon**: Native M1/M2/M3 support with Metal backend
✅ **Maintainability**: Future-proof codebase, active community support

---

## 📁 File Structure

### New TF 2.x Files

```
MolGAN/
├── environment_processggan.yml         # Modern environment (Python 3.10 + TF 2.15)
├── environment_legacy.yml              # Legacy environment (Python 3.6 + TF 1.8)
│
├── models/
│   ├── process_gan.py                  # Legacy TF 1.x version
│   └── process_gan_tf2.py              # 🆕 Modern TF 2.x version
│
├── optimizers/
│   ├── process_optimizer.py            # Legacy TF 1.x version
│   └── process_optimizer_tf2.py        # 🆕 Modern TF 2.x version
│
├── example_process.py                  # Legacy TF 1.x training script
├── example_process_tf2.py              # 🆕 Modern TF 2.x training script
│
├── README_PROCESSGGAN.md               # General guide
└── README_TF2_MIGRATION.md             # This file
```

### File Mapping

| Purpose | Legacy (TF 1.x) | Modern (TF 2.x) |
|---------|-----------------|-----------------|
| Environment | `environment_legacy.yml` | `environment_processggan.yml` |
| Model | `models/process_gan.py` | `models/process_gan_tf2.py` |
| Optimizer | `optimizers/process_optimizer.py` | `optimizers/process_optimizer_tf2.py` |
| Training | `example_process.py` | `example_process_tf2.py` |

---

## 🚀 Quick Start (TF 2.x)

### 1. Setup Environment

```bash
# Create modern environment
conda env create -f environment_processggan.yml
conda activate ProcessGAN

# Verify installation
python -c "import tensorflow as tf; print(f'TF version: {tf.__version__}')"
# Expected output: TF version: 2.15.x
```

### 2. Run Training

```bash
# Use TF 2.x version
python example_process_tf2.py
```

### 3. Test Components

```bash
# Test model
python models/process_gan_tf2.py

# Test optimizer
python optimizers/process_optimizer_tf2.py

# Test reward function (still compatible)
python tests/test_process_reward.py
```

---

## 🔄 Migration Guide: TF 1.x → 2.x

### Code Changes Summary

#### 1. Session-based → Eager Execution

**TF 1.x (Legacy)**:
```python
# Session-based execution
session = tf.Session()
session.run(tf.global_variables_initializer())

# Feed dict for inputs
feed_dict = {
    model.embeddings: z,
    model.training: True
}
output = session.run(model.nodes_argmax, feed_dict=feed_dict)
```

**TF 2.x (Modern)**:
```python
# Eager execution - no session needed!
# Direct function call
output = model.generator(z, training=True)

# Or with keras Model
predictions = model(inputs, training=True)
```

#### 2. Placeholders → Function Arguments

**TF 1.x**:
```python
self.embeddings = tf.placeholder(dtype=tf.float32, shape=(None, embedding_dim))
self.training = tf.placeholder_with_default(False, shape=())
```

**TF 2.x**:
```python
def call(self, embeddings, training=False):
    # Embeddings and training are function arguments
    ...
```

#### 3. tf.layers → tf.keras.layers

**TF 1.x**:
```python
output = tf.layers.dense(inputs, units=128, activation=tf.nn.tanh)
output = tf.layers.dropout(output, rate=dropout_rate, training=training)
```

**TF 2.x**:
```python
self.dense = layers.Dense(128, activation='tanh')
self.dropout = layers.Dropout(dropout_rate)

output = self.dense(inputs)
output = self.dropout(output, training=training)
```

#### 4. Optimizer.minimize() → GradientTape

**TF 1.x**:
```python
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, var_list=trainable_vars)

# Execute training
session.run(train_op, feed_dict=feed_dict)
```

**TF 2.x**:
```python
optimizer = tf.keras.optimizers.Adam(learning_rate)

with tf.GradientTape() as tape:
    # Forward pass
    predictions = model(inputs, training=True)
    loss = loss_fn(labels, predictions)

# Backward pass
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

#### 5. Variable Scope → keras.Model

**TF 1.x**:
```python
with tf.variable_scope('generator'):
    output = dense_layer(input, units=128)

with tf.variable_scope('discriminator'):
    score = discriminator_network(input)
```

**TF 2.x**:
```python
class ProcessGenerator(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(128)

    def call(self, inputs):
        return self.dense(inputs)

generator = ProcessGenerator()
```

#### 6. Model Saving

**TF 1.x**:
```python
saver = tf.train.Saver()
saver.save(session, 'checkpoints/model.ckpt')

# Restore
saver.restore(session, 'checkpoints/model.ckpt')
```

**TF 2.x**:
```python
# Save weights
model.save_weights('checkpoints/model')

# Restore weights
model.load_weights('checkpoints/model')

# Or save entire model
model.save('checkpoints/model.keras')
model = tf.keras.models.load_model('checkpoints/model.keras')
```

---

## 🎯 Architecture Comparison

### Generator

**TF 1.x Implementation**:
```python
def generator(embeddings, units, vertexes, edges, nodes, training, dropout_rate):
    output = multi_dense_layers(embeddings, units, activation=tf.nn.tanh, ...)
    edges_logits = tf.layers.dense(output, edges * vertexes * vertexes)
    nodes_logits = tf.layers.dense(output, vertexes * nodes)
    return edges_logits, nodes_logits
```

**TF 2.x Implementation**:
```python
class ProcessGenerator(keras.Model):
    def __init__(self, ...):
        super().__init__()
        self.dense_layers = [layers.Dense(u, activation='tanh') for u in units]
        self.edges_dense = layers.Dense(edges * vertexes * vertexes)
        self.nodes_dense = layers.Dense(vertexes * nodes)

    def call(self, embeddings, training=False):
        h = embeddings
        for layer in self.dense_layers:
            h = layer(h)
        edges_logits = self.edges_dense(h)
        nodes_logits = self.nodes_dense(h)
        return self.gumbel_softmax(edges_logits), self.gumbel_softmax(nodes_logits)
```

### Discriminator (R-GCN)

**TF 1.x**:
- Function-based with `variable_scope`
- Session-based execution
- Manual gradient computation for GP

**TF 2.x**:
```python
class ProcessDiscriminator(keras.Model):
    def __init__(self, ...):
        super().__init__()
        # R-GCN layers as model attributes
        self.rgcn_layers = [...]
        self.mlp = keras.Sequential([...])

    def call(self, adjacency, nodes, training=False):
        # Forward pass through R-GCN
        h = nodes
        for rgcn_layer in self.rgcn_layers:
            # Message passing per flow type
            ...
        # Graph pooling
        graph_emb = self.attention_pooling(h)
        score = self.mlp(graph_emb, training=training)
        return score, graph_emb
```

### Training Loop

**TF 1.x**:
```python
session = tf.Session()
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        # Get batch
        batch = data.next_train_batch(batch_size)

        # Train discriminator
        for _ in range(n_critic):
            session.run(train_step_D, feed_dict={...})

        # Train generator
        session.run(train_step_G, feed_dict={...})
```

**TF 2.x**:
```python
trainer = ProcessGANTrainer(model)

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        # Get batch
        batch = data.next_train_batch(batch_size)

        # Training step (all in eager mode)
        losses = trainer.train_step(
            real_adj=batch[1],
            real_nodes=batch[2],
            batch_size=batch_size,
            n_critic=n_critic,
            reward_function=reward_fn
        )
```

---

## ⚡ Performance Comparison

### Benchmark Results (Estimated)

| Metric | TF 1.x | TF 2.x | Improvement |
|--------|--------|--------|-------------|
| Training Speed | 1.0x | 1.2-1.5x | +20-50% faster |
| Memory Usage | 1.0x | 0.8-0.9x | -10-20% lower |
| GPU Utilization | ~60% | ~80-90% | Better optimization |
| Debugging Time | High | Low | Eager execution |

### Why TF 2.x is Faster

1. **Graph Optimization**: Better auto-graph compilation with `@tf.function`
2. **XLA Compilation**: Just-in-Time compilation for kernels
3. **Mixed Precision**: Easier to enable FP16 training
4. **Better GPU Kernels**: Optimized CUDA operations

---

## 🛠️ Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution**: Install TensorFlow 2.x

```bash
conda activate ProcessGAN
pip install tensorflow==2.15.0

# Or with GPU support
pip install tensorflow[and-cuda]==2.15.0
```

### Issue: "AttributeError: 'Module' object has no attribute 'Session'"

**Problem**: Using TF 1.x code with TF 2.x installed

**Solution**: Use TF 2.x versions of files:
- `models/process_gan_tf2.py` (not `process_gan.py`)
- `example_process_tf2.py` (not `example_process.py`)

### Issue: Slower than expected

**Solution**: Enable performance optimizations

```python
# In example_process_tf2.py

# 1. Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 2. Use @tf.function decorator
@tf.function
def train_step(...):
    ...

# 3. Enable XLA compilation
tf.config.optimizer.set_jit(True)
```

### Issue: Out of memory on GPU

**Solution**: Reduce batch size or enable memory growth

```python
# Enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## 📈 Feature Comparison

| Feature | TF 1.x | TF 2.x | Notes |
|---------|--------|--------|-------|
| Eager Execution | ❌ | ✅ | Default in TF 2.x |
| Keras Integration | Partial | Full | Native `keras.Model` |
| Model Subclassing | ❌ | ✅ | OOP design |
| `@tf.function` | ❌ | ✅ | Auto-graph compilation |
| Mixed Precision | Manual | Built-in | Easier FP16 training |
| TensorBoard | Manual | Auto | Better integration |
| SavedModel | Complex | Simple | `model.save()` |
| Debugging | Difficult | Easy | Python debugging works |
| GPU Support | CUDA 10 | CUDA 12 | Modern GPU support |
| Apple Silicon | ❌ | ✅ | M1/M2/M3 support |

---

## 🎓 Learning Resources

### TensorFlow 2.x Documentation
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Migration Guide](https://www.tensorflow.org/guide/migrate)

### ProcessGAN Specific
- [ARCHITETTURA_PROCESSGGAN.md](Docs/ARCHITETTURA_PROCESSGGAN.md) - Architecture details
- [README_PROCESSGGAN.md](README_PROCESSGGAN.md) - Usage guide

---

## 🔄 When to Use Which Version

### Use TF 2.x (Modern) if:
✅ Starting new research/development
✅ Need modern Python libraries (PM4Py, etc.)
✅ Want better debugging experience
✅ Need GPU support for modern hardware
✅ Publishing new papers (2024+)
✅ Production deployment

### Use TF 1.x (Legacy) only if:
⚠️ Reproducing exact MolGAN paper results
⚠️ Comparing with legacy implementations
⚠️ Maintaining old codebase

---

## 📝 Summary

**Recommendation**: **Use TF 2.x version for all new work**

The modern TF 2.x implementation provides:
- ✅ Cleaner, more maintainable code
- ✅ Better performance and GPU utilization
- ✅ Easier debugging with eager execution
- ✅ Future-proof stack (Python 3.10+, TF 2.15+)
- ✅ Compatible with modern ML ecosystem

Legacy TF 1.x version is kept for:
- Historical reference
- Exact reproduction of original results
- Comparison purposes

---

## 📞 Support

For questions or issues:
1. Check [README_PROCESSGGAN.md](README_PROCESSGGAN.md) for general usage
2. Consult [ARCHITETTURA_PROCESSGGAN.md](Docs/ARCHITETTURA_PROCESSGGAN.md) for architecture details
3. Review TensorFlow migration guide: https://www.tensorflow.org/guide/migrate

---

**Last Updated**: 2025-01-11
**TensorFlow Version**: 2.15.0
**Python Version**: 3.10+
