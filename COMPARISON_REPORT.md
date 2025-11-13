# Report Tecnico: Comparazione Graph vs OLD ProcessGAN

**Autore**: Analisi Tecnica Comparativa  
**Data**: 2024  
**Versione**: 1.0

---

## Executive Summary

Questo documento presenta un'analisi comparativa dettagliata tra due implementazioni di ProcessGAN:
- **OLD**: Approccio originale basato su sequenze con R-GCN
- **Graph**: Nuovo approccio nativo basato su grafi con Keras API

Le differenze principali riguardano l'architettura del modello, la rappresentazione dei dati, il sistema di vincoli e la pipeline di training.

---

## 1. ARCHITETTURA DEL MODELLO

### 1.1 Generator

#### OLD (Sequence-based)
```python
# Architettura Dense → Reshape
- Input: z ∈ ℝ^128 (latent noise)
- Dense layers: (128, 256, 512)
- Output branches:
  * Adjacency: (max_act, max_act, flow_types)
  * Nodes: (max_act, activity_types)
- Gumbel-Softmax per sampling differenziabile
```

**Caratteristiche**:
- Genera matrici di adiacenza con tipi di flusso (SEQUENCE, LOOP, XOR, AND, SKIP)
- Nodi rappresentano attività in sequenza
- Dropout: 0.0 (default)
- Enforce START opzionale (hard constraint)

#### Graph (Native Graph-based)
```python
# Architettura Dense → Reshape con NO_EDGE channel
- Input: z ∈ ℝ^128 (latent noise)
- Dense layers: (256, 512, 1024) - PIÙ PROFONDO
- Output branches:
  * Adjacency: (max_nodes, max_nodes, edge_types + 1)
    → +1 per NO_EDGE channel (sparsità)
  * Nodes: (max_nodes, num_activities)
- Gumbel-Softmax con temperature scheduling
```

**Caratteristiche**:
- **NO_EDGE channel**: Permette grafi sparsi naturalmente
- Nodi rappresentano vertici di grafo (non sequenze)
- Dropout: 0.1 (regolarizzazione)
- Temperature decay: 5.0 → 0.5 (annealing)

**Differenze Chiave**:
| Aspetto | OLD | Graph |
|---------|-----|-------|
| Profondità | 3 layer (128→256→512) | 3 layer (256→512→1024) |
| Sparsità | Implicita | Esplicita (NO_EDGE) |
| Dropout | 0.0 | 0.1 |
| Temperature | Statica | Dinamica (decay) |

---

### 1.2 Discriminator

#### OLD (R-GCN con Feature Matching)
```python
# R-GCN → Attention Pooling → MLP
- R-GCN layers: (128, 64)
- Per ogni flow type: Dense layer separato
- Aggregazione: Sum over flow types
- Pooling: Attention-based weighted sum
- MLP: (128, 64, 1)
- Output: Score + Features (per feature matching)
```

**Caratteristiche**:
- R-GCN manuale con layer separati per flow type
- Attention pooling per graph-level representation
- Feature matching loss (confronto features reali vs fake)
- Wasserstein loss (WGAN-GP)

#### Graph (R-GCN Stack con Keras)
```python
# RGCNStack → Global Pooling → MLP
- R-GCN layers: (128, 64)
- RGCNStack: Implementazione modulare
- Layer normalization opzionale
- Pooling: Mean o Max (configurabile)
- MLP: (128, 64, 1)
- Kernel constraint: MaxNorm(1.0) - LIPSCHITZ
```

**Caratteristiche**:
- R-GCN modulare (RGCNStack class)
- Layer normalization per stabilità
- MaxNorm constraint per Lipschitz continuity (WGAN)
- Pooling method configurabile

**Differenze Chiave**:
| Aspetto | OLD | Graph |
|---------|-----|-------|
| R-GCN | Manuale | Modulare (RGCNStack) |
| Normalization | No | Layer Norm |
| Lipschitz | Gradient Penalty | MaxNorm + GP |
| Pooling | Attention | Mean/Max |
| Feature Matching | Sì | No |

---

### 1.3 Value Network (RL)

#### OLD
```python
# Riusa Discriminator + Value Head
- Stessa architettura del Discriminator
- Output: Value ∈ [0,1] (sigmoid)
- Loss: MSE(V_pred, reward)
```

#### Graph
```python
# Non presente nell'implementazione Graph
- Graph usa solo GAN puro
- Nessun componente RL
```

**Differenza Fondamentale**:
- **OLD**: Hybrid GAN + RL (lambda mixing)
- **Graph**: Pure GAN (solo adversarial training)

---

## 2. RAPPRESENTAZIONE DEI DATI

### 2.1 Input Format

#### OLD (XES-based)
```python
# Carica da XES con PM4Py
- Input: XES event log
- Pattern discovery: Petri Net + DFG
- Encoding:
  * Adjacency: (batch, max_act, max_act) - INDICI
  * Nodes: (batch, max_act) - INDICI
  * Features: (batch, max_act, 3) - temporal
- Flow types: 5 (SEQ, LOOP, XOR, AND, SKIP)
```

#### Graph (IG-based)
```python
# Carica da .g (Instance Graph)
- Input: .g file (pre-processato)
- Parsing: Vertices + Edges con attributi
- Encoding:
  * Adjacency: (batch, max_nodes, max_nodes, edge_types)
  * Nodes: (batch, max_nodes, num_activities)
  * Features: (batch, max_nodes, 3) - opzionale
- Edge types: 5 (SEQUENCE, AND_SPLIT, AND_JOIN, LOOP, XOR)
```

**Differenze Chiave**:
| Aspetto | OLD | Graph |
|---------|-----|-------|
| Input | XES (event log) | .g (instance graph) |
| Rappresentazione | Sequenze | Grafi nativi |
| Adjacency | Indici (sparse) | One-hot (dense) |
| Nodes | Indici | One-hot |
| Pattern Discovery | PM4Py (runtime) | Pre-processato |

---

### 2.2 Data Pipeline

#### OLD
```
XES → PM4Py → Pattern Discovery → Trace Encoding → Matrices
                ↓
         Petri Net + DFG
```

**Steps**:
1. Load XES con PM4Py
2. Discover Petri Net (Inductive Miner)
3. Build DFG (Directly-Follows Graph)
4. Detect patterns (XOR/AND splits, loops)
5. Encode traces to matrices

#### Graph
```
.g file → Parse → EdgeTypeClassifier → Matrices
                        ↓
                  AND detection
```

**Steps**:
1. Parse .g file (già processato)
2. Classify edges (AND_SPLIT/JOIN detection)
3. Build adjacency matrices
4. One-hot encoding

**Differenze Chiave**:
- **OLD**: Pattern discovery a runtime (lento ma flessibile)
- **Graph**: Pre-processing offline (veloce ma rigido)

---

## 3. SISTEMA DI VINCOLI

### 3.1 OLD Constraints (Impliciti)

**Nessun sistema di vincoli esplicito**

L'approccio OLD si affida a:
- Reward function (RL) per guidare la generazione
- Feature matching nel discriminator
- Gradient penalty (WGAN-GP)

**Reward Components**:
```python
reward = w1*validity + w2*fitness + w3*conformance + w4*diversity
```

Dove:
- **Validity**: Sintassi corretta (START/END)
- **Fitness**: Token replay (PM4Py)
- **Conformance**: Alignment (PM4Py A*)
- **Diversity**: Edit distance

---

### 3.2 Graph Constraints (Espliciti)

**Sistema di vincoli strutturato con 8 componenti**

```python
class GraphProcessConstraints:
    - start_node_constraint_loss()      # λ=3.0
    - end_node_constraint_loss()        # λ=6.0
    - activity_frequency_loss()         # λ=1.0
    - connectivity_loss()               # λ=5.0
    - structural_validity_loss()        # λ=2.0
    - degree_constraint_loss()          # λ=5.0
    - path_existence_loss()             # λ=3.0
    - nodes_on_path_loss()              # λ=5.0
```

#### 3.2.1 START Node Constraint (λ=3.0)
```python
# 4 componenti:
1. Zero in-degree (nessun arco entrante)
2. At least 1 out-degree (almeno un arco uscente)
3. Uniqueness (solo un nodo START)
4. Positioning (START deve essere primo)
```

#### 3.2.2 END Node Constraint (λ=6.0)
```python
# 5 componenti UNIFICATI:
1. Terminal positioning (END al nodo terminale)
2. Penalize END at non-terminals
3. Count constraint (sum ≈ 1)
4. Concentration (max prob ≥ 0.7)
5. Zero out-degree (nessun arco uscente)
```

**Innovazione**: Usa attention mechanism per identificare nodo terminale

#### 3.2.3 Activity Frequency (λ=1.0)
```python
# KL divergence tra distribuzione target e generata
loss = KL(target_freq || generated_freq)
```

#### 3.2.4 Connectivity (λ=5.0)
```python
# Penalizza nodi isolati
# Ogni nodo attivo deve avere ≥1 arco (in o out)
```

#### 3.2.5 Structural Validity (λ=2.0)
```python
# NO LOOPS ALLOWED (DAG enforcement)
1. Self-loops penalty (massima priorità)
2. Bidirectional edges penalty
3. Transitive closure cycles (soft)
```

#### 3.2.6 Degree Constraints (λ=5.0)
```python
# Soft constraints:
- START: in=0, out≥1
- END: in≥1, out=0
- Others: in≥1, out≥1
```

#### 3.2.7 Path Existence (λ=3.0)
```python
# Verifica reachability START → END
# Usa transitive closure
```

#### 3.2.8 Nodes on Path (λ=5.0)
```python
# Tutti i nodi devono essere su un path START → END
# Penalizza nodi "morti"
```

---

### 3.3 Comparazione Vincoli

| Vincolo | OLD | Graph | Note |
|---------|-----|-------|------|
| START/END | Reward-based | Hard constraints | Graph più rigoroso |
| Connectivity | No | Sì (λ=5.0) | Graph previene isolamento |
| DAG | No | Sì (λ=2.0) | Graph forza aciclicità |
| Degree | No | Sì (λ=5.0) | Graph controlla in/out |
| Path | No | Sì (λ=8.0) | Graph garantisce START→END |
| Frequency | Reward | Loss (λ=1.0) | Entrambi |

**Conclusione**: Graph ha vincoli **molto più stringenti** e **espliciti**

---

## 4. TRAINING PIPELINE

### 4.1 OLD Training Loop

```python
# Custom training loop con schedulers
for epoch in range(epochs):
    # Dynamic reward weights (early → late)
    weights = dynamic_rewards.get_weights(epoch)
    
    # Lambda mixing (GAN ↔ RL)
    lambda_mix = lambda_scheduler.get_lambda(epoch)
    
    # Temperature annealing
    temperature = temp_scheduler.get_temperature(epoch)
    
    # Learning rate scheduling (Cosine Annealing)
    lr = lr_scheduler.get_learning_rate(epoch)
    
    for step in range(steps_per_epoch):
        # Train discriminator (n_critic times)
        for _ in range(n_critic):
            train_discriminator_step()
        
        # Train generator (with RL if lambda < 1.0)
        train_generator_step(lambda_mix)
        
        # Train value network (if RL enabled)
        if lambda_mix < 1.0:
            train_value_network_step()
    
    # Evaluation
    metrics = evaluator.evaluate_samples()
    
    # Checkpointing (best model only)
    checkpoint_manager.save_checkpoint()
    
    # Early stopping
    if early_stopping.check(metrics):
        break
```

**Caratteristiche**:
- Custom training loop (no Keras fit)
- Dynamic reward weights (validity → fitness/conformance)
- Lambda mixing (100% GAN → 60% GAN + 40% RL)
- Cosine annealing LR con warm restarts
- Early stopping su reward
- Best model checkpointing

---

### 4.2 Graph Training Loop

```python
# Keras API con fit()
gan.compile(d_optimizer, g_optimizer)

history = gan.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[
        ModelCheckpoint,
        EarlyStopping,
        TensorBoard,
        CSVLogger,
        BackupAndRestore,
        DualLearningRateScheduler,
        SampleGraphsCallback
    ]
)
```

**train_step() override**:
```python
def train_step(self, real_data):
    # Train discriminator (n_critic times)
    for _ in range(n_critic):
        with tf.GradientTape() as tape:
            # Wasserstein loss + GP
            d_loss = wasserstein_loss() + λ_gp * gradient_penalty()
        update_discriminator()
    
    # Train generator (once)
    with tf.GradientTape() as tape:
        # Adversarial + Constraints
        g_loss = -D(fake) + λ_constraint * constraint_loss()
    update_generator()
    
    # Temperature annealing
    temperature = max(temperature * decay, temp_min)
    
    return metrics
```

**Caratteristiche**:
- Keras API nativa (fit/callbacks)
- Pure GAN (no RL)
- Constraint loss integrato
- Temperature decay automatico
- Callbacks standard (TensorBoard, CSV, etc.)
- Validation durante training

---

### 4.3 Comparazione Training

| Aspetto | OLD | Graph |
|---------|-----|-------|
| API | Custom loop | Keras fit() |
| RL | Sì (lambda mixing) | No |
| Constraints | Reward-based | Loss-based |
| LR Schedule | Cosine Annealing | Exponential Decay |
| Callbacks | Custom | Keras standard |
| Validation | Manuale | Automatica |
| Checkpointing | Best only | Best + Backup |
| Early Stopping | Reward | Constraint loss |

---

## 5. LOSS FUNCTIONS

### 5.1 OLD Losses

#### Discriminator
```python
# WGAN-GP
loss_D = E[D(fake)] - E[D(real)] + λ_gp * GP
```

#### Generator
```python
# Feature Matching + RL
loss_G_adv = ||E[features_real] - E[features_fake]||²
loss_RL = -E[V(fake)]  # Value network prediction

# Mixing
loss_G = λ_mix * loss_G_adv + (1 - λ_mix) * loss_RL
```

#### Value Network
```python
loss_V = MSE(V(real), reward_real) + MSE(V(fake), reward_fake)
```

**Total Losses**: 4 (D, G, V, RL)

---

### 5.2 Graph Losses

#### Discriminator
```python
# WGAN-GP (identico)
loss_D = E[D(fake)] - E[D(real)] + λ_gp * GP
```

#### Generator
```python
# Adversarial + Constraints
loss_G = -E[D(fake)] + λ_constraint * constraint_loss

# Constraint loss (8 componenti)
constraint_loss = (
    λ_start * start_loss +
    λ_end * end_loss +
    λ_freq * frequency_loss +
    λ_conn * connectivity_loss +
    λ_struct * structural_loss +
    λ_degree * degree_loss +
    λ_path * path_loss +
    λ_nodes_path * nodes_on_path_loss
)
```

**Total Losses**: 2 (D, G) + 8 constraint components

---

### 5.3 Comparazione Losses

| Loss Type | OLD | Graph |
|-----------|-----|-------|
| Discriminator | WGAN-GP | WGAN-GP |
| Generator Adv | Feature Matching | Standard WGAN |
| RL Component | Sì (Value Network) | No |
| Constraints | Reward (external) | Loss (internal) |
| Total Networks | 3 (G, D, V) | 2 (G, D) |

**Differenza Chiave**: 
- OLD: Constraints via RL reward (soft, learned)
- Graph: Constraints via loss (hard, explicit)

---

## 6. HYPERPARAMETERS

### 6.1 OLD Default Config
```yaml
# Model
z_dim: 128
max_activities: 15
decoder_units: [128, 256, 512]
discriminator_units: [128, 64]
mlp_units: 128
dropout_rate: 0.0

# Training
batch_size: 32
epochs: 100
n_critic: 5
learning_rate: 1e-4
learning_rate_D: 2e-4
learning_rate_V: 1e-4

# GAN/RL
lambda_start: 1.0  # Pure GAN
lambda_end: 0.6    # 60% GAN + 40% RL
lambda_decay_start: 10

# Temperature
temperature_start: 5.0
temperature_end: 0.5
temperature_decay: 0.95

# Reward weights (dynamic)
early: {validity: 0.4, fitness: 0.2, conformance: 0.2, diversity: 0.2}
late: {validity: 0.2, fitness: 0.3, conformance: 0.3, diversity: 0.2}
```

---

### 6.2 Graph Default Config
```python
# Model
noise_dim: 128
max_nodes: 20  # Più grande
generator_hidden_dims: [256, 512, 1024]  # Più profondo
rgcn_hidden_dims: [128, 64]
mlp_hidden_dims: [128, 64]
generator_dropout: 0.1
discriminator_dropout: 0.3

# Training
batch_size: 32
epochs: 500  # Molto più lungo
n_critic: 3  # Meno critic steps
d_lr: 1e-4
g_lr: 1e-4  # Stesso LR per G e D

# Constraints (λ weights)
lambda_gp: 10.0
lambda_constraint: 0.5
lambda_start: 3.0
lambda_end: 6.0
lambda_frequency: 1.0
lambda_connectivity: 5.0
lambda_structure: 2.0
lambda_degree: 5.0
lambda_path: 3.0
lambda_node_on_path: 5.0

# Temperature
temp_start: 5.0
temp_min: 0.5
temp_decay: 0.99995  # Molto più lento
```

---

### 6.3 Comparazione Hyperparameters

| Parameter | OLD | Graph | Ratio |
|-----------|-----|-------|-------|
| Epochs | 100 | 500 | 5x |
| n_critic | 5 | 3 | 0.6x |
| Generator depth | 512 | 1024 | 2x |
| Dropout G | 0.0 | 0.1 | - |
| Dropout D | 0.0 | 0.3 | - |
| Temp decay | 0.95 | 0.99995 | Molto più lento |
| LR schedule | Cosine | Exponential | - |

**Osservazioni**:
- Graph richiede **5x più epoche** (vincoli più stringenti)
- Graph usa **dropout** (regolarizzazione)
- Graph ha **temperature decay più lento** (convergenza graduale)

---

## 7. EVALUATION METRICS

### 7.1 OLD Metrics

**Training Metrics**:
- loss_D, loss_G, loss_V, loss_RL
- grad_penalty
- reward_mean

**Evaluation Metrics**:
```python
# Reward components
- validity_mean
- fitness_mean (PM4Py token replay)
- conformance_mean (PM4Py alignment)
- diversity_mean (edit distance)
- reward_mean (weighted sum)

# Quality metrics
- valid_rate (% traces valide)
- unique_rate (% traces uniche)
- novel_rate (% traces non in training)
- high_quality_rate (reward ≥ 0.7)
```

**Validation**: Su validation set durante training

---

### 7.2 Graph Metrics

**Training Metrics** (Keras):
- d_loss
- g_loss
- gradient_penalty
- constraint_loss
- d_real_score
- d_fake_score
- temperature

**Evaluation Metrics**:
```python
# Non specificato nel codice
# Probabilmente:
- Constraint satisfaction rate
- Graph validity (DAG, connectivity)
- Activity distribution matching
```

**Validation**: Automatica con Keras (val_loss)

---

### 7.3 Comparazione Metrics

| Metric Type | OLD | Graph |
|-------------|-----|-------|
| PM4Py Fitness | Sì | No |
| PM4Py Conformance | Sì | No |
| Constraint Loss | No | Sì |
| Diversity | Sì | No |
| Validation | Manual | Automatic |

**Differenza Chiave**: 
- OLD: Focus su **process mining quality** (fitness, conformance)
- Graph: Focus su **structural validity** (constraints)

---

## 8. VANTAGGI E SVANTAGGI

### 8.1 OLD Approach

#### Vantaggi ✅
1. **Process Mining Integration**: PM4Py nativo (fitness, conformance)
2. **Flexible Constraints**: Reward-based (soft, learnable)
3. **RL Component**: Può apprendere reward complessi
4. **Dynamic Weights**: Adatta focus durante training
5. **Proven Metrics**: Metriche standard process mining

#### Svantaggi ❌
1. **Slow Pattern Discovery**: PM4Py a runtime
2. **Complex Training**: Custom loop, molti scheduler
3. **RL Instability**: Lambda mixing può essere instabile
4. **No Hard Constraints**: Può generare grafi invalidi
5. **Reward Engineering**: Richiede tuning pesi

---

### 8.2 Graph Approach

#### Vantaggi ✅
1. **Hard Constraints**: Garantisce validità strutturale
2. **Keras API**: Training semplice, callbacks standard
3. **Fast Data Loading**: Pre-processing offline
4. **Explicit Sparsity**: NO_EDGE channel
5. **Modular Design**: RGCNStack, EdgeTypeClassifier
6. **Stable Training**: Pure GAN (no RL complexity)

#### Svantaggi ❌
1. **No Process Mining**: Nessuna integrazione PM4Py
2. **Rigid Constraints**: 8 loss components con pesi fissi
3. **Long Training**: 5x più epoche
4. **Pre-processing Required**: .g file format
5. **No Conformance**: Non valuta conformance a modello
6. **Overfitting Risk**: Vincoli troppo stringenti?

---

## 9. USE CASES

### 9.1 Quando usare OLD

✅ **Ideale per**:
- Generazione di event log realistici
- Conformance checking
- Process discovery benchmarking
- Ricerca in process mining
- Dataset con Petri Net di riferimento

❌ **Non ideale per**:
- Grafi generici (non process-specific)
- Training veloce
- Vincoli strutturali rigidi

---

### 9.2 Quando usare Graph

✅ **Ideale per**:
- Generazione di grafi strutturati (DAG)
- Vincoli topologici rigidi
- Training con Keras ecosystem
- Grafi con parallelismo (AND-split/join)
- Applicazioni non process mining

❌ **Non ideale per**:
- Process mining con conformance
- Vincoli soft/learnable
- Training rapido (poche epoche)

---

## 10. RACCOMANDAZIONI

### 10.1 Per Migliorare OLD

1. **Modularizzare Constraints**: Separare reward components
2. **Keras Integration**: Migrare a Keras API
3. **Cache Pattern Discovery**: Salvare Petri Net
4. **Simplify RL**: Rimuovere lambda mixing (troppo complesso)
5. **Add Hard Constraints**: Integrare alcuni vincoli Graph

### 10.2 Per Migliorare Graph

1. **Add PM4Py**: Integrare fitness/conformance
2. **Soft Constraints**: Rendere alcuni vincoli learnable
3. **Reduce Epochs**: Ottimizzare convergenza
4. **XES Support**: Supportare input XES diretto
5. **Adaptive Weights**: Lambda weights dinamici

### 10.3 Hybrid Approach

**Proposta**: Combinare i punti di forza

```python
class HybridProcessGAN:
    # Da Graph
    - Keras API
    - Hard structural constraints (DAG, connectivity)
    - NO_EDGE channel
    - RGCNStack modulare
    
    # Da OLD
    - PM4Py integration (fitness, conformance)
    - Reward-based soft constraints
    - Dynamic weight scheduling
    - XES input support
    
    # Nuovo
    - Constraint loss + Reward loss
    - Adaptive lambda weights
    - Fast convergence (< 100 epochs)
```

---

## 11. CONCLUSIONI

### Sintesi Comparativa

| Dimensione | OLD | Graph | Winner |
|------------|-----|-------|--------|
| **Architettura** | Sequence-based | Graph-native | Graph |
| **Constraints** | Soft (reward) | Hard (loss) | Dipende |
| **Training** | Custom loop | Keras API | Graph |
| **Process Mining** | PM4Py full | Nessuno | OLD |
| **Convergenza** | 100 epochs | 500 epochs | OLD |
| **Validità** | Soft | Garantita | Graph |
| **Flessibilità** | Alta | Bassa | OLD |
| **Manutenibilità** | Media | Alta | Graph |

### Raccomandazione Finale

**Per ricerca in Process Mining**: Usare **OLD** (PM4Py integration essenziale)

**Per applicazioni generiche di graph generation**: Usare **Graph** (vincoli strutturali migliori)

**Per produzione**: Sviluppare **Hybrid** (best of both worlds)

---

## APPENDICE: Codice Chiave

### A.1 OLD Generator Call
```python
def call(self, embeddings, training=False, temperature=1.0):
    h = embeddings
    for layer in self.dense_layers:
        h = layer(h, training=training)
    
    edges_logits = self.edges_dense(h)
    nodes_logits = self.nodes_dense(h)
    
    edges = self.gumbel_softmax(edges_logits, temperature, training)
    nodes = self.gumbel_softmax(nodes_logits, temperature, training)
    
    return edges, nodes
```

### A.2 Graph Generator Call
```python
def call(self, z, temperature=1.0, hard=False, training=True):
    h = z
    for layer in self.dense_layers:
        h = layer(h, training=training)
    
    adj_logits = self.adjacency_head(h)
    adj_logits = tf.reshape(adj_logits, (..., edge_types + 1))  # +1 for NO_EDGE
    
    node_logits = self.node_head(h)
    
    adjacency_with_no_edge = gumbel_softmax(adj_logits, temperature, hard)
    adjacency = adjacency_with_no_edge[:, :, :, :-1]  # Remove NO_EDGE
    
    nodes = gumbel_softmax(node_logits, temperature, hard)
    
    return adjacency, nodes
```

### A.3 Graph Constraint Example
```python
def end_node_constraint_loss(self, nodes, adjacency):
    # 1. Terminal positioning
    out_degrees = tf.reduce_sum(adjacency, axis=2)
    terminal_score = active_mask * tf.nn.sigmoid(-(out_degrees - 0.1) * 20.0)
    attention_weights = tf.nn.softmax(terminal_score, axis=1)
    
    # 2. Penalize END at non-terminals
    non_terminal_mask = tf.nn.sigmoid((out_degrees - 0.1) * 10.0)
    end_at_non_terminal = end_probs * non_terminal_mask
    
    # 3. Count constraint
    end_count = tf.reduce_sum(end_probs, axis=1)
    loss_count = tf.reduce_mean(tf.square(end_count - 1.0))
    
    # 4. Concentration
    max_end_prob = tf.reduce_max(end_probs, axis=1)
    loss_concentration = tf.reduce_mean(tf.nn.relu(0.7 - max_end_prob))
    
    # 5. Zero out-degree
    end_out_penalty = end_probs * out_degrees
    
    return combined_loss
```

---

**Fine Report**
