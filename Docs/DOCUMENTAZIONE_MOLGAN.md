# MolGAN: Documentazione Completa del Progetto

## Panoramica del Progetto

**MolGAN** è un'implementazione TensorFlow di un modello generativo implicito per piccoli grafi molecolari, basato sull'articolo di ricerca "MolGAN: An implicit generative model for small molecular graphs" (De Cao & Kipf, 2018).

### Scopo del Progetto

Il progetto implementa reti neurali generative (GAN e VAE) in grado di generare molecole rappresentate come grafi. Le molecole vengono rappresentate attraverso:

- **Matrici di adiacenza**: che codificano i legami tra atomi
- **Matrici di nodi**: che codificano i tipi di atomi e le loro proprietà
- **Rappresentazioni latenti**: spazi dimensionali ridotti che catturano le caratteristiche essenziali delle molecole

## Struttura del Progetto

```
MolGAN/
├── data/                           # Dataset molecolari
│   └── download_dataset.sh         # Script per scaricare i dati
├── models/                         # Architetture dei modelli
│   ├── __init__.py                 # Encoder/decoder e funzioni di utilità
│   ├── gan.py                      # Implementazione del modello GAN
│   └── vae.py                      # Implementazione del modello VAE
├── optimizers/                     # Ottimizzatori per l'addestramento
│   ├── __init__.py
│   ├── gan.py                      # Ottimizzatore per GAN
│   └── vae.py                      # Ottimizzatore per VAE
├── utils/                          # Utilità e moduli di supporto
│   ├── __init__.py
│   ├── layers.py                   # Layer personalizzati per grafi
│   ├── molecular_metrics.py        # Metriche per valutare molecole
│   ├── progress_bar.py             # Barra di progresso
│   ├── sparse_molecular_dataset.py # Gestione dataset molecolari
│   ├── trainer.py                  # Framework di addestramento
│   └── utils.py                    # Funzioni di utilità generali
├── environment.yml                 # Dipendenze conda
├── example.py                      # Esempio d'uso del framework
├── LICENSE                         # Licenza MIT
└── README.md                       # Documentazione base
```

## Dipendenze Principali

- **TensorFlow >= 1.7.0**: Framework per deep learning
- **RDKit**: Libreria per cheminformatica e manipolazione molecolare
- **NumPy**: Calcoli numerici
- **Scikit-learn**: Algoritmi di machine learning

## Architettura del Sistema

### 1. Rappresentazione dei Dati

#### Strutture Dati Fondamentali

```
Molecola → Tripla di Matrici:
├── Adjacency Tensor (A): [batch, vertexes, vertexes, edge_types]
├── Node Tensor (X): [batch, vertexes, node_types]
└── Feature Tensor (F): [batch, vertexes, features]
```

**Dettagli delle Rappresentazioni:**

- **Adjacency Tensor**: Codifica i legami tra atomi

  - Dimensioni: `(N, 9, 9, 4)` dove N=batch, 9=max atomi, 4=tipi di legame
  - Tipi di legame: [no_bond, single, double, triple]
  - Matrice simmetrica per legami non direzionali

- **Node Tensor**: Codifica i tipi di atomi

  - Dimensioni: `(N, 9, atom_types)`
  - One-hot encoding per tipo di atomo (C, N, O, F, etc.)
  - Posizione nella matrice = posizione dell'atomo nel grafo

- **Feature Tensor**: Proprietà aggiuntive degli atomi
  - Valenza, carica formale, aromaticità, etc.

#### Encoding e Preprocessing

```python
# Workflow di encoding (utils/sparse_molecular_dataset.py)
Molecola SMILES → RDKit Mol → Matrici Sparse → One-hot Encoding
```

### 2. Architettura GAN

#### Generator (models/gan.py)

```
Noise Vector (z) → Decoder → (Adjacency_logits, Node_logits)
                           ↓
                   Gumbel-Softmax → (Adjacency_soft, Node_soft)
```

**Componenti del Generator:**

- **Input**: Vector latente `z` di dimensione `z_dim`
- **Decoder**: Multi-layer dense network che mappa `z` → grafi
- **Output**: Logits per adjacency matrix e node matrix
- **Gumbel-Softmax**: Rende differenziabile la discretizzazione

#### Discriminator (models/gan.py)

```
(Adjacency, Nodes) → Encoder (RGCN) → Features → Dense → Real/Fake Score
```

**Componenti del Discriminator:**

- **Input**: Coppia (Adjacency, Node matrices)
- **Encoder**: Relational Graph Convolutional Network (RGCN)
- **Aggregation**: Graph pooling per ottenere rappresentazione fissa
- **Output**: Score di autenticità [0,1]

#### Value Network (per Reinforcement Learning)

```
(Adjacency, Nodes) → Encoder → Value Estimate
```

### 3. Architettura VAE

#### Encoder (models/vae.py)

```
(Adjacency, Nodes, Features) → RGCN → Graph_Pooling → μ, σ
                                                        ↓
                                               Reparameterization → z
```

#### Decoder (condiviso con GAN)

```
z → Dense_Layers → (Adjacency_logits, Node_logits)
```

**Differenze VAE vs GAN:**

- **VAE**: Ricostruzione + regolarizzazione KL
- **GAN**: Competizione adversarial + feature matching

## Componenti Chiave del Codice

### 1. Layer Personalizzati (utils/layers.py)

#### Graph Convolution Layer

```python
def graph_convolution_layer(inputs, units):
    adjacency, hidden, nodes = inputs
    # 1. Messaggio da nodi vicini
    neighbor_messages = matmul(adjacency, node_features)
    # 2. Self-message
    self_message = dense(node_features)
    # 3. Aggregazione
    output = neighbor_messages + self_message
    return activation(output)
```

#### Graph Aggregation Layer

```python
def graph_aggregation_layer(inputs, units):
    # Attention-based pooling per grafo di dimensione variabile
    attention_weights = sigmoid(dense(inputs))
    weighted_features = attention_weights * tanh(dense(inputs))
    graph_embedding = sum(weighted_features, axis=1)  # Pool over nodes
    return graph_embedding
```

### 2. Decoder Variants (models/**init**.py)

#### Adjacency Decoder

```python
def decoder_adj(embeddings, units, vertexes, edges, nodes):
    # 1. Embedding → Hidden representation
    hidden = multi_dense_layers(embeddings, units)

    # 2. Adjacency matrix generation
    adj_logits = dense(hidden, edges * vertexes * vertexes)
    adj_logits = reshape(adj_logits, [-1, edges, vertexes, vertexes])
    adj_logits = (adj_logits + transpose(adj_logits)) / 2  # Simmetria

    # 3. Node matrix generation
    node_logits = dense(hidden, vertexes * nodes)
    node_logits = reshape(node_logits, [-1, vertexes, nodes])

    return adj_logits, node_logits
```

**Altri Decoder Disponibili:**

- `decoder_dot()`: Decoder con prodotto scalare
- `decoder_rnn()`: Decoder con RNN per generazione sequenziale

### 3. Postprocessing (models/**init**.py)

#### Gumbel-Softmax

```python
def postprocess_logits(logits, temperature):
    # 1. Standard softmax
    softmax = tf.nn.softmax(logits / temperature)

    # 2. Gumbel noise
    gumbel_noise = -log(-log(uniform_random))
    gumbel_logits = logits + gumbel_noise

    # 3. Gumbel-softmax (differenziabile)
    gumbel_softmax = tf.nn.softmax(gumbel_logits / temperature)

    # 4. Hard sampling (non differenziabile)
    gumbel_argmax = one_hot(argmax(gumbel_logits))

    return softmax, argmax, gumbel_logits, gumbel_softmax, gumbel_argmax
```

## Flusso di Addestramento

### 1. Preprocessing dei Dati

```python
# sparse_molecular_dataset.py
for molecule in dataset:
    # 1. Parsing SMILES/SDF
    mol = Chem.MolFromSmiles(smiles)

    # 2. Estrazione features
    adjacency = extract_adjacency_matrix(mol)
    nodes = extract_node_features(mol)

    # 3. Padding a dimensione fissa
    adjacency_padded = pad_matrix(adjacency, max_size=9)

    # 4. One-hot encoding
    adjacency_onehot = to_categorical(adjacency_padded)
```

### 2. Training Loop GAN

```python
# example.py - Training principale
for epoch in epochs:
    for batch in dataloader:
        # 1. Train Discriminator
        real_data = batch_real
        fake_data = generator(noise)
        loss_D = discriminator_loss(real_data, fake_data)

        # 2. Train Generator
        fake_data = generator(noise)
        loss_G = generator_loss(discriminator(fake_data))

        # 3. Reinforcement Learning (opzionale)
        reward = compute_molecular_reward(fake_data)
        loss_RL = -value_network(fake_data) * reward
```

### 3. Loss Functions

#### GAN Losses (optimizers/gan.py)

```python
# Discriminator: Wasserstein + Gradient Penalty
loss_D = -logits_real + logits_fake + λ * gradient_penalty

# Generator: Adversarial + Feature Matching
loss_G = -logits_fake + α * ||features_real - features_fake||²

# Value Network: Prediction Error
loss_V = ||value_pred - reward||²
```

#### VAE Losses (optimizers/vae.py)

```python
# Reconstruction
loss_recon = CrossEntropy(real_adjacency, pred_adjacency) +
             CrossEntropy(real_nodes, pred_nodes)

# KL Divergence
loss_kl = KL(q(z|x) || p(z))

# Total ELBO
loss_VAE = loss_recon + β * loss_kl
```

## Tecniche Avanzate

### 1. Reinforcement Learning Integration

- **Reward Function**: Valuta proprietà molecolari (validità, QED, SA score)
- **Policy Gradient**: Ottimizza generatore verso reward alti
- **Value Network**: Stima valore atteso di una molecola

### 2. Feature Matching

- **Obiettivo**: Matcher features intermedi invece che output finale
- **Vantaggio**: Stabilizza training, migliora diversità
- **Implementazione**: `||E[φ(x_real)] - E[φ(x_fake)]||²`

### 3. Gradient Penalty

- **Problema**: Training instabile in WGAN
- **Soluzione**: Penalizza gradient norm lungo interpolazioni
- **Formula**: `λ * (||∇_x D(x)|| - 1)²`

## Metriche di Valutazione

### 1. Validità Chimica (molecular_metrics.py)

```python
def valid_scores(molecules):
    # Verifica usando RDKit
    valid_mols = [mol for mol in molecules if Chem.MolToSmiles(mol)]
    return len(valid_mols) / len(molecules)
```

### 2. Proprietà Molecolari

- **QED**: Quantitative Estimate of Drug-likeness
- **SA Score**: Synthetic Accessibility Score
- **LogP**: Partition coefficient (solubilità)
- **Tanimoto Similarity**: Similarità strutturale

### 3. Diversità e Novità

- **Uniqueness**: Percentuale di molecole uniche generate
- **Novelty**: Percentuale non presente nel training set
- **Diversity**: Dissimilarità media tra molecole generate

## Workflow Completo

### Step 1: Preparazione Dataset

```python
# 1. Carica dataset molecolari
data = SparseMolecularDataset()
data.load('data/gdb9_9nodes.sparsedataset')

# 2. Configurazione modello
model = GraphGANModel(
    data.vertexes,           # Numero massimo di vertici (9)
    data.bond_num_types,     # Tipi di legami
    data.atom_num_types,     # Tipi di atomi
    z_dim,                   # Dimensione spazio latente (8)
    decoder_units=(128, 256, 512),
    discriminator_units=((128, 64), 128, (128, 64)),
    decoder=decoder_adj,
    discriminator=encoder_rgcn
)
```

### Step 2: Training

```python
# Ottimizzatore e trainer
optimizer = GraphGANOptimizer(model)
trainer = Trainer(model, optimizer, session)

# Addestramento
trainer.train(
    batch_dim=batch_dim,
    epochs=epochs,
    steps=steps,
    train_fetch_dict=train_fetch_dict,
    train_feed_dict=train_feed_dict,
    eval_fetch_dict=eval_fetch_dict,
    eval_feed_dict=eval_feed_dict
)
```

### Step 3: Generazione e Valutazione

```python
# Generazione di nuove molecole
embeddings = model.sample_z(batch_size)
nodes, edges = session.run([model.nodes_argmax, model.edges_argmax],
                          feed_dict={model.embeddings: embeddings})

# Conversione in molecole
mols = [data.matrices2mol(n, e) for n, e in zip(nodes, edges)]

# Valutazione con metriche
validity = MolecularMetrics.valid_total_score(mols)
uniqueness = MolecularMetrics.unique_total_score(mols)
```

## Limitazioni e Vincoli

### 1. Dimensioni Fisse

- **Massimo 9 atomi** per molecola
- **Padding necessario** per dimensioni variabili
- **Memory usage** cresce quadraticamente

### 2. Rappresentazione Discreta

- **Gumbel-softmax** per differenziabilità
- **Quantization error** nella ricostruzione
- **Mode collapse** possibile

### 3. Vincoli Chimici

- **Valenza atomica** non sempre rispettata
- **Regole di legame** possono essere violate
- **Post-processing** necessario per validità

## Estensibilità del Framework

### 1. Nuovi Encoder/Decoder

- **Interface standardizzato** per componenti intercambiabili
- **Graph attention networks** supportati
- **RNN decoder** per generazione sequenziale

### 2. Diverse Loss Functions

- **Modular optimizer design**
- **Multi-objective optimization**
- **Custom reward functions**

### 3. Scale-up Capabilities

- **Batch processing** efficiente
- **GPU acceleration** con TensorFlow
- **Distributed training** possibile

## Citazione

```bibtex
@article{de2018molgan,
  title={{MolGAN: An implicit generative model for small molecular graphs}},
  author={De Cao, Nicola and Kipf, Thomas},
  journal={ICML 2018 workshop on Theoretical Foundations and Applications of Deep Generative Models},
  year={2018}
}
```

## Licenza

Questo progetto è rilasciato sotto licenza MIT.

---

Questa architettura fornisce una base solida per l'adattamento a domini diversi dalle molecole, mantenendo i principi fondamentali della generazione di grafi strutturati.
