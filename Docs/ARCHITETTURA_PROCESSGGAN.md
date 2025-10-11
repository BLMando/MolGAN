# Architettura ProcessGAN: Da Molecole a Event Logs

## 📚 Panoramica

**ProcessGAN** è un adattamento di MolGAN per la generazione di event logs nel contesto del process mining. Mantiene l'architettura core basata su **Generative Adversarial Networks (GAN)** ma adatta rappresentazioni, metriche e reward function al dominio dei processi aziendali.

### Obiettivo

Generare **event logs sintetici** plausibili per data augmentation, partendo da un dataset limitato di tracce reali, con garanzie di:
- **Validità**: Conformità a regole sintattiche e semantiche
- **Fitness**: Aderenza al modello di processo sottostante
- **Diversità**: Varianti nuove ma realistiche
- **Plausibilità**: Score [0,1] che quantifica quanto una trace è probabile

---

## 🔄 Mappatura Concettuale: Chimica → Process Mining

| **Dominio Molecolare** | **Dominio Process Mining** | **Rappresentazione** |
|------------------------|----------------------------|----------------------|
| **Atomo** (C, N, O, F) | **Attività** (Start, Submit, Review, Approve) | Nodo del grafo |
| **Legame chimico** (Single, Double, Triple) | **Flusso di controllo** (Sequence, XOR, AND, Loop) | Arco del grafo |
| **Molecola** (H₂O, C₆H₆) | **Trace/Case** (sequenza di attività) | Grafo completo |
| **Validità chimica** (valenze, aromaticità) | **Conformità processo** (start/end, ordine logico) | Vincoli strutturali |
| **Proprietà molecolari** (LogP, QED, SA) | **KPI processo** (cycle time, cost, conformance) | Metriche di qualità |

### Esempio Concreto

```
MOLECOLA: H₂O (Acqua)
├─ Atomi: H, H, O
├─ Legami: H-O (single), H-O (single)
└─ Proprietà: Polare, dipolo, alta tensione superficiale

EVENT LOG: Case_123 (Purchase Order Process)
├─ Attività: Start, Create_PO, Approve_PO, Send_PO, Receive_Goods, End
├─ Flussi: Start→Create (seq), Create→Approve (seq), Approve→Send (seq)
└─ Proprietà: Cycle time 3 giorni, Cost 150€, Conformance 0.95
```

---

## 🏗️ Architettura Tripartita ProcessGAN

```
┌─────────────────────────────────────────────────────────────────┐
│                         PROCESSGGAN                              │
│                                                                  │
│  ┌────────────────────┐           ┌───────────────────────┐    │
│  │    GENERATOR       │           │   DISCRIMINATOR       │    │
│  │                    │           │                       │    │
│  │  Input: z ~ N(0,I) │           │  Input: (Adj, Nodes)  │    │
│  │  Dimension: 16     │           │                       │    │
│  │        ↓           │           │    ┌─────────────┐    │    │
│  │  Dense Layers:     │           │    │   R-GCN     │    │    │
│  │   z → 128 → 256 →  │           │    │  Layers     │    │    │
│  │       → 512        │           │    │  (128, 64)  │    │    │
│  │        ↓           │           │    └──────┬──────┘    │    │
│  │  ┌──────────────┐  │           │           ↓           │    │
│  │  │ Adjacency    │  │           │  Graph Pooling        │    │
│  │  │ (15×15×5)    │  │──────────▶│  (Attention-based)    │    │
│  │  │              │  │           │           ↓           │    │
│  │  │ Nodes        │  │           │  MLP Classifier       │    │
│  │  │ (15×12)      │  │           │           ↓           │    │
│  │  └──────────────┘  │           │  Score ∈ [0,1]        │    │
│  │        ↓           │           │  (Real/Fake)          │    │
│  │  Gumbel-Softmax    │           └───────────────────────┘    │
│  │        ↓           │                       ↓                 │
│  │  Generated Trace   │                  Feedback               │
│  └─────────┬──────────┘                                         │
│            │                                                    │
│            ↓                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │            REWARD FUNCTION [0,1]                        │   │
│  │                                                          │   │
│  │  R(trace) = α·Validity + β·Fitness +                   │   │
│  │             γ·Conformance + δ·Diversity                 │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐                    │   │
│  │  │  VALIDITY    │  │   FITNESS    │                    │   │
│  │  │              │  │              │                    │   │
│  │  │ • Start/End  │  │ • Token      │                    │   │
│  │  │ • Length     │  │   Replay     │                    │   │
│  │  │ • No loops   │  │ • Missing    │                    │   │
│  │  │              │  │   tokens     │                    │   │
│  │  │ Score: [0,1] │  │ Score: [0,1] │                    │   │
│  │  └──────────────┘  └──────────────┘                    │   │
│  │                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐                    │   │
│  │  │ CONFORMANCE  │  │  DIVERSITY   │                    │   │
│  │  │              │  │              │                    │   │
│  │  │ • Alignment  │  │ • Edit       │                    │   │
│  │  │   cost       │  │   distance   │                    │   │
│  │  │ • A* algo    │  │ • From train │                    │   │
│  │  │              │  │              │                    │   │
│  │  │ Score: [0,1] │  │ Score: [0,1] │                    │   │
│  │  └──────────────┘  └──────────────┘                    │   │
│  │                                                          │   │
│  │  Total Reward = Weighted Sum ∈ [0,1]                   │   │
│  │  (High reward = plausible trace)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎨 Componente 1: GENERATOR

### 1.1 Funzione

Trasforma un **vettore latente casuale z** in una **trace di processo** rappresentata come grafo.

```
Input:  z ∈ ℝ^16  (Gaussian noise)
Output: (Adjacency Matrix, Node Vector)
        ├─ Adjacency: (15×15×5) - Tipi di flusso tra attività
        └─ Nodes: (15×12) - Tipi di attività
```

### 1.2 Architettura

```python
class ProcessGenerator:
    def __init__(self):
        self.z_dim = 16              # Dimensione spazio latente
        self.max_activities = 15     # Max attività per trace
        self.flow_types = 5          # Tipi di flusso
        self.activity_types = 12     # Tipi di attività

    def forward(self, z):
        # Layer 1: Espansione
        h1 = Dense(128, activation='tanh')(z)      # 16 → 128
        h2 = Dense(256, activation='tanh')(h1)     # 128 → 256
        h3 = Dense(512, activation='tanh')(h2)     # 256 → 512

        # Branch A: Adjacency Matrix (flussi tra attività)
        edges_logits = Dense(5 * 15 * 15)(h3)     # 512 → 1125
        edges_logits = Reshape((5, 15, 15))(edges_logits)
        edges_logits = Transpose((0, 2, 3, 1))(edges_logits)  # (batch, 15, 15, 5)

        # Branch B: Node Vector (attività)
        nodes_logits = Dense(15 * 12)(h3)         # 512 → 180
        nodes_logits = Reshape((15, 12))(nodes_logits)  # (batch, 15, 12)

        # Gumbel-Softmax per differenziabilità
        edges = GumbelSoftmax(tau=0.5)(edges_logits)
        nodes = GumbelSoftmax(tau=0.5)(nodes_logits)

        return edges, nodes
```

### 1.3 Gumbel-Softmax Trick

**Problema**: Sampling discreto non è differenziabile → impossibile fare backpropagation.

**Soluzione**: Gumbel-Softmax approximation

```python
def gumbel_softmax(logits, temperature=0.5):
    # Aggiungi rumore Gumbel
    U = tf.random.uniform(tf.shape(logits))
    gumbel_noise = -tf.log(-tf.log(U + 1e-20) + 1e-20)

    # Softmax con temperatura
    y = tf.nn.softmax((logits + gumbel_noise) / temperature)

    return y  # Differenziabile!
```

**Temperatura τ**:
- τ → 0: Output discreto (one-hot)
- τ → ∞: Output uniforme
- τ = 0.5: Bilanciamento ottimale

### 1.4 Output: Rappresentazione Grafo

**Adjacency Matrix** (15×15×5):

```python
# Esempio: Trace "Start → Submit → Review → End"

adjacency[0] = [  # Flow type 0: SEQUENCE
    [0, 1, 0, 0, 0, ...],  # Start → Submit
    [0, 0, 1, 0, 0, ...],  # Submit → Review
    [0, 0, 0, 1, 0, ...],  # Review → End
    [0, 0, 0, 0, 0, ...],  # End → (none)
    ...
]

adjacency[1] = [  # Flow type 1: XOR-split (alternative paths)
    [...],
]

adjacency[2] = [  # Flow type 2: AND-split (parallel execution)
    [...],
]

# ... altri tipi di flusso
```

**Node Vector** (15×12):

```python
# One-hot encoding delle attività
nodes = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Position 0: Start
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Position 1: Submit
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Position 2: Review
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Position 3: End
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Position 4-14: Padding
    ...
]
```

### 1.5 Tipi di Flusso di Controllo

```python
flow_types = {
    0: 'SEQUENCE',      # A → B (esecuzione sequenziale)
    1: 'XOR_SPLIT',     # A → B OR C (scelta esclusiva)
    2: 'AND_SPLIT',     # A → B & C (fork parallelo)
    3: 'LOOP',          # B → A (iterazione)
    4: 'SKIP',          # A → C (B opzionale)
}
```

**Esempio di Trace Complessa**:

```
Start → Submit → [Review & Check] → Approve → End
      ↑                                  ↓
      └──────────── Rework ──────────────┘

Flow encoding:
├─ Start → Submit: SEQUENCE
├─ Submit → Review: AND_SPLIT (fork)
├─ Submit → Check: AND_SPLIT (fork)
├─ Review → Approve: XOR_SPLIT (join parallelo)
├─ Check → Approve: XOR_SPLIT (join parallelo)
├─ Approve → End: SEQUENCE
└─ Approve → Rework → Submit: LOOP (rework cycle)
```

---

## 🔍 Componente 2: DISCRIMINATOR

### 2.1 Funzione

Valuta se una trace (rappresentata come grafo) è **reale** (dal dataset) o **fake** (generata).

```
Input:  (Adjacency Matrix, Node Vector)
Output: Score ∈ [0,1]
        ├─ 1.0 = sicuramente reale
        ├─ 0.5 = incertezza
        └─ 0.0 = sicuramente fake
```

### 2.2 Architettura: Relational Graph Convolutional Network (R-GCN)

**Idea chiave**: Messaggi aggregati **separatamente per tipo di edge**.

```python
class ProcessDiscriminator:
    def __init__(self):
        self.rgcn_layers = [(128, 64)]  # Hidden dimensions
        self.mlp_units = 128

    def forward(self, adjacency, nodes):
        h = nodes  # (batch, 15, 12) - features iniziali

        # R-GCN Layers
        for units_out in [128, 64]:
            messages_by_type = []

            # Per ogni tipo di flusso, trasformazione separata
            for flow_type in range(5):
                adj_slice = adjacency[:, :, :, flow_type]  # (batch, 15, 15)

                # Trasforma features nodi
                transformed = Dense(units_out)(h)  # (batch, 15, units_out)

                # Aggrega messaggi dai vicini
                neighbor_msg = tf.matmul(adj_slice, transformed)
                messages_by_type.append(neighbor_msg)

            # Combina tutti i tipi di flusso
            aggregated = tf.reduce_sum(tf.stack(messages_by_type), axis=0)

            # Self-connection
            self_msg = Dense(units_out)(h)

            # Update con attivazione
            h = tf.nn.tanh(aggregated + self_msg)
            h = Dropout(0.2)(h)

        # Graph Pooling: (batch, 15, 64) → (batch, 128)
        attention = Dense(1, activation='sigmoid')(h)  # (batch, 15, 1)
        pooled = Dense(self.mlp_units, activation='tanh')(h)
        graph_embedding = tf.reduce_sum(attention * pooled, axis=1)

        # MLP Classifier
        x = Dense(128, activation='tanh')(graph_embedding)
        x = Dense(64, activation='tanh')(x)
        score = Dense(1, activation='sigmoid')(x)  # [0,1]

        return score
```

### 2.3 R-GCN: Intuizione Matematica

**Standard GCN** (single edge type):
```
h_i^(l+1) = σ(∑_{j∈N(i)} W^(l) h_j^(l))
```

**R-GCN** (multiple edge types):
```
h_i^(l+1) = σ(∑_{r∈R} ∑_{j∈N_r(i)} W_r^(l) h_j^(l) + W_0^(l) h_i^(l))
                ↑                 ↑                      ↑
         edge types        neighbors of     self-connection
                            type r
```

**Vantaggi**:
- Cattura pattern specifici per tipo di flusso (es: loops, paralleli)
- Peso separato per ogni relazione
- Permette di distinguere "Submit → Review" da "Approve → Rework"

### 2.4 Graph Pooling Attenzionale

**Problema**: Traces di lunghezza variabile → serve dimensione fissa per classifier.

**Soluzione**: Weighted sum con attention mechanism.

```python
# Per ogni nodo, calcola importance weight
attention_weights = sigmoid(Dense(1)(h))  # (batch, 15, 1) ∈ [0,1]

# Feature transformation
features = tanh(Dense(128)(h))  # (batch, 15, 128)

# Weighted sum → embedding globale
graph_emb = sum(attention_weights * features, axis=nodes)  # (batch, 128)
```

**Interpretazione**:
- Nodi importanti (Start, End, decision points) → alto weight
- Nodi padding o opzionali → basso weight
- Output indipendente dalla lunghezza della trace

---

## 🎁 Componente 3: REWARD FUNCTION

### 3.1 Funzione

Quantifica la **plausibilità** di una trace generata con score [0,1].

```
Input:  Trace = ['Start', 'Submit', 'Review', 'Approve', 'End']
Output: Reward ∈ [0,1]
        ├─ 1.0 = trace perfettamente plausibile
        ├─ 0.5 = trace mediocre
        └─ 0.0 = trace invalida/impossibile
```

### 3.2 Architettura Multi-Objective

```python
R(trace) = α·R_validity + β·R_fitness + γ·R_conformance + δ·R_diversity

Dove:
├─ α + β + γ + δ = 1.0 (pesi normalizzati)
└─ Ogni R_i ∈ [0,1]
```

**Configurazione Default**:
```python
weights = {
    'validity': 0.30,      # α: Regole sintattiche
    'fitness': 0.25,       # β: Token replay fitness
    'conformance': 0.25,   # γ: Alignment con modello
    'diversity': 0.20      # δ: Diversità da training
}
```

### 3.3 Sub-Reward 1: VALIDITY [0,1]

**Obiettivo**: Verifica regole sintattiche base.

```python
def validity_score(trace):
    score = 1.0

    # Regola 1: Trace non vuota
    if len(trace) == 0:
        return 0.0

    # Regola 2: Deve iniziare con 'Start'
    if trace[0] != 'Start':
        score *= 0.5

    # Regola 3: Deve finire con 'End'
    if trace[-1] != 'End':
        score *= 0.5

    # Regola 4: Lunghezza ragionevole (3-20 attività)
    if not (3 <= len(trace) <= 20):
        score *= 0.7

    # Regola 5: No attività duplicate consecutive
    for i in range(len(trace) - 1):
        if trace[i] == trace[i+1]:
            score *= 0.8
            break

    # Regola 6: 'End' non può apparire prima dell'ultima posizione
    if 'End' in trace[:-1]:
        score *= 0.3

    return score
```

**Esempi**:
```
Trace: ['Start', 'Submit', 'Review', 'Approve', 'End']
→ Validity = 1.0 ✓

Trace: ['Submit', 'Review', 'End']  # Manca Start
→ Validity = 0.5

Trace: ['Start', 'Review', 'Review', 'End']  # Duplicate consecutive
→ Validity = 0.8

Trace: []  # Empty
→ Validity = 0.0
```

### 3.4 Sub-Reward 2: FITNESS [0,1]

**Obiettivo**: Token replay fitness rispetto a Petri net.

**Algoritmo Token-Based Replay**:

```
1. Inizializza: 1 token in place "Start"
2. Per ogni attività nella trace:
   a. Se transizione abilitata → fire (consuma/produce token)
   b. Altrimenti → missing token (penalità)
3. Fine: se rimangono token → remaining tokens (penalità)
```

**Formula Fitness (van der Aalst et al.)**:

```python
def fitness_score(trace, petri_net):
    produced_tokens = 0
    consumed_tokens = 0
    missing_tokens = 0
    remaining_tokens = 0

    # Token replay simulation
    marking = initial_marking(petri_net)

    for activity in trace:
        transition = petri_net.get_transition(activity)

        if is_enabled(transition, marking):
            # Fire transition
            marking = fire_transition(transition, marking)
            produced_tokens += len(transition.output_places)
            consumed_tokens += len(transition.input_places)
        else:
            # Missing token → force fire
            missing_tokens += 1
            marking = force_fire(transition, marking)

    remaining_tokens = sum(marking.values())

    # Fitness calculation
    if consumed_tokens + missing_tokens + remaining_tokens > 0:
        fitness = 0.5 * (1 - missing_tokens / (consumed_tokens + missing_tokens)) + \
                  0.5 * (1 - remaining_tokens / (consumed_tokens + remaining_tokens))
    else:
        fitness = 0.0

    return fitness
```

**Interpretazione**:
- **fitness = 1.0**: Trace perfectly fits (no missing/remaining tokens)
- **fitness = 0.5**: Half of moves are incorrect
- **fitness = 0.0**: Completely non-fitting trace

**Esempio**:

```
Petri Net: Start → Submit → Review → Approve → End

Trace: ['Start', 'Submit', 'Review', 'Approve', 'End']
├─ Missing tokens: 0
├─ Remaining tokens: 0
└─ Fitness = 1.0 ✓

Trace: ['Start', 'Approve', 'End']  # Skip Submit, Review
├─ Missing tokens: 2 (for Approve)
├─ Remaining tokens: 2 (from Submit, Review)
└─ Fitness ≈ 0.5
```

### 3.5 Sub-Reward 3: CONFORMANCE [0,1]

**Obiettivo**: Alignment cost rispetto al modello di riferimento.

**Algoritmo A* Alignment**:

```python
def conformance_score(trace, reference_model):
    # Calcola optimal alignment con A*
    alignment = compute_alignment(trace, reference_model)

    # Conta move types
    sync_moves = count(alignment, 'sync')        # Matching moves
    model_moves = count(alignment, 'model')      # Model-only moves (skips)
    log_moves = count(alignment, 'log')          # Log-only moves (deviations)

    # Alignment cost
    cost = model_moves + log_moves
    max_cost = len(trace) + len(reference_model.activities)

    # Normalizza
    conformance = 1.0 - (cost / max_cost)

    return max(0.0, conformance)
```

**Tipi di Move**:
```
Sync move:   (a, a) - Attività matcha perfettamente
Model move:  (-, a) - Attività nel modello ma non nella trace
Log move:    (a, -) - Attività nella trace ma non nel modello
```

**Esempio**:

```
Model:  Start → Submit → Review → Approve → End
Trace:  Start → Submit → Check → Approve → End

Alignment:
├─ (Start, Start)     → sync move ✓
├─ (Submit, Submit)   → sync move ✓
├─ (-, Review)        → model move (skipped) ⚠
├─ (Check, -)         → log move (deviation) ⚠
├─ (Approve, Approve) → sync move ✓
└─ (End, End)         → sync move ✓

Cost = 2 (1 model move + 1 log move)
Conformance = 1 - (2 / 10) = 0.8
```

### 3.6 Sub-Reward 4: DIVERSITY [0,1]

**Obiettivo**: Traces generate devono essere **diverse** dal training set (per augmentation).

```python
def diversity_score(trace, training_traces):
    # Calcola edit distance minimo da training
    min_distance = float('inf')

    for train_trace in training_traces:
        distance = levenshtein_distance(trace, train_trace)
        min_distance = min(min_distance, distance)

    # Normalizza
    max_len = max(len(trace), max(len(t) for t in training_traces))
    diversity = min_distance / max_len if max_len > 0 else 0.0

    # Clip: troppa diversità (>0.8) può essere irrealistica
    diversity = min(diversity, 0.8) / 0.8

    return diversity
```

**Levenshtein Distance**:

```python
def levenshtein_distance(trace1, trace2):
    """
    Numero minimo di operazioni (insert, delete, substitute)
    per trasformare trace1 in trace2
    """
    m, n = len(trace1), len(trace2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    # Inizializza
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    # Dynamic programming
    for i in range(1, m+1):
        for j in range(1, n+1):
            if trace1[i-1] == trace2[j-1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i-1][j] + 1,      # Deletion
                dp[i][j-1] + 1,      # Insertion
                dp[i-1][j-1] + cost  # Substitution
            )

    return dp[m][n]
```

**Esempio**:

```
Training trace: ['Start', 'Submit', 'Review', 'Approve', 'End']

Generated trace: ['Start', 'Submit', 'Check', 'Approve', 'End']
Edit distance: 1 (substitute Review → Check)
Diversity: 1/5 = 0.2

Generated trace: ['Start', 'Process', 'Validate', 'End']
Edit distance: 3 (multiple edits)
Diversity: 3/5 = 0.6 (good for augmentation!)

Generated trace: ['Start', 'Submit', 'Review', 'Approve', 'End']
Edit distance: 0 (identical)
Diversity: 0.0 (bad for augmentation - duplicate!)
```

### 3.7 Reward Composito: Esempio Completo

```python
trace = ['Start', 'Submit', 'Review', 'Approve', 'End']

# Calcola sub-rewards
R_validity = 1.0      # Perfetto: Start/End presenti, lunghezza OK
R_fitness = 0.92      # Ottimo: solo 1 missing token
R_conformance = 0.85  # Buono: alignment cost basso
R_diversity = 0.55    # Moderato: abbastanza diverso da training

# Weighted sum
R_total = 0.30 * 1.0 + 0.25 * 0.92 + 0.25 * 0.85 + 0.20 * 0.55
R_total = 0.30 + 0.23 + 0.21 + 0.11
R_total = 0.85

# Interpretazione
if R_total >= 0.8:
    print("Trace altamente plausibile ✓")
elif R_total >= 0.6:
    print("Trace accettabile ⚠")
else:
    print("Trace di bassa qualità ✗")
```

---

## 🔄 Training Loop

### 4.1 Adversarial Training + Reinforcement Learning

```python
def train_step(generator, discriminator, reward_fn, batch_real):
    """
    Training step ProcessGAN
    """
    batch_size = batch_real[0].shape[0]

    # ════════════════════════════════════════════════════════
    # PHASE 1: Train Discriminator (5 steps)
    # ════════════════════════════════════════════════════════

    for _ in range(5):
        # Real traces
        adj_real, nodes_real = batch_real

        # Generate fake traces
        z = np.random.normal(0, 1, (batch_size, 16))
        adj_fake, nodes_fake = generator(z)

        # Discriminator predictions
        D_real = discriminator(adj_real, nodes_real)
        D_fake = discriminator(adj_fake, nodes_fake)

        # Wasserstein loss
        loss_D = -tf.reduce_mean(D_real) + tf.reduce_mean(D_fake)

        # Gradient penalty (stabilità training)
        alpha = tf.random.uniform((batch_size, 1, 1, 1))
        interpolated_adj = alpha * adj_real + (1-alpha) * adj_fake
        interpolated_nodes = alpha[:,:,:,0] * nodes_real + (1-alpha[:,:,:,0]) * nodes_fake

        with tf.GradientTape() as gp_tape:
            gp_tape.watch([interpolated_adj, interpolated_nodes])
            D_inter = discriminator(interpolated_adj, interpolated_nodes)

        grads = gp_tape.gradient(D_inter, [interpolated_adj, interpolated_nodes])
        grad_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in grads]))
        gradient_penalty = 10.0 * (grad_norm - 1.0)**2

        loss_D_total = loss_D + gradient_penalty

        # Update discriminator
        discriminator.optimizer.minimize(
            loss_D_total,
            discriminator.trainable_variables
        )

    # ════════════════════════════════════════════════════════
    # PHASE 2: Train Generator (1 step)
    # ════════════════════════════════════════════════════════

    # Generate fake traces
    z = np.random.normal(0, 1, (batch_size, 16))
    adj_fake, nodes_fake = generator(z)

    # Convert to traces for reward
    traces_fake = matrices_to_traces(adj_fake, nodes_fake)

    # Compute rewards
    rewards = reward_fn.compute_reward(traces_fake)  # (batch, 1) ∈ [0,1]

    # Generator losses
    D_fake = discriminator(adj_fake, nodes_fake)

    loss_G_adv = -tf.reduce_mean(D_fake)           # Fool discriminator
    loss_G_reward = -tf.reduce_mean(rewards)       # Maximize reward

    # Combined loss
    lambda_adv = 0.6
    lambda_reward = 0.4
    loss_G_total = lambda_adv * loss_G_adv + lambda_reward * loss_G_reward

    # Update generator
    generator.optimizer.minimize(
        loss_G_total,
        generator.trainable_variables
    )

    return {
        'loss_D': loss_D.numpy(),
        'loss_G': loss_G_total.numpy(),
        'reward_mean': rewards.mean().numpy(),
        'reward_std': rewards.std().numpy(),
        'D_real_mean': D_real.mean().numpy(),
        'D_fake_mean': D_fake.mean().numpy()
    }
```

### 4.2 Loss Functions Explained

**Discriminator Loss (Wasserstein)**:
```
L_D = -E[D(x_real)] + E[D(x_fake)] + λ·GP
      ↑               ↑                ↑
   Maximize score  Minimize score  Gradient penalty
   for real        for fake        (stabilità)
```

**Generator Loss (Adversarial + RL)**:
```
L_G = -E[D(x_fake)] - E[R(x_fake)]
      ↑               ↑
   Fool            Maximize reward
   discriminator   (process conformance)
```

**Gradient Penalty**:
```
GP = E[(||∇_x D(x)||_2 - 1)²]
     ↑
  Impone Lipschitz constraint: gradient norm ≈ 1
```

### 4.3 Hyperparameter Tuning

```python
config = {
    # Model architecture
    'z_dim': 16,                    # Latent dimension
    'max_activities': 15,           # Max trace length
    'flow_types': 5,                # Control flow types
    'activity_types': 12,           # Activity types
    'decoder_units': (128, 256, 512),
    'discriminator_units': (128, 64),

    # Training
    'batch_size': 32,
    'learning_rate': 1e-4,
    'n_critic': 5,                  # D steps per G step
    'epochs': 200,
    'gradient_penalty_weight': 10.0,

    # Loss weights
    'lambda_adversarial': 0.6,
    'lambda_reward': 0.4,

    # Reward weights
    'alpha_validity': 0.30,
    'beta_fitness': 0.25,
    'gamma_conformance': 0.25,
    'delta_diversity': 0.20,

    # Gumbel-Softmax
    'temperature_start': 5.0,       # Initial temperature
    'temperature_end': 0.5,         # Final temperature
    'temperature_decay': 0.95,      # Decay rate per epoch
}
```

---

## 📊 Valutazione e Metriche

### 5.1 Metriche di Qualità

```python
def evaluate_generation_quality(generated_traces, reference_model, training_traces):
    """
    Valuta qualità traces generate
    """
    metrics = {}

    # 1. VALIDITY RATE
    validity_scores = [validity_score(t) for t in generated_traces]
    metrics['validity_rate'] = np.mean([s >= 0.9 for s in validity_scores])
    metrics['validity_mean'] = np.mean(validity_scores)

    # 2. FITNESS DISTRIBUTION
    fitness_scores = [fitness_score(t, reference_model) for t in generated_traces]
    metrics['fitness_mean'] = np.mean(fitness_scores)
    metrics['fitness_std'] = np.std(fitness_scores)

    # 3. CONFORMANCE
    conformance_scores = [conformance_score(t, reference_model) for t in generated_traces]
    metrics['conformance_mean'] = np.mean(conformance_scores)

    # 4. DIVERSITY
    diversity_scores = [diversity_score(t, training_traces) for t in generated_traces]
    metrics['diversity_mean'] = np.mean(diversity_scores)

    # 5. UNIQUENESS
    unique_traces = len(set([tuple(t) for t in generated_traces]))
    metrics['uniqueness'] = unique_traces / len(generated_traces)

    # 6. NOVELTY
    novel_traces = [t for t in generated_traces if tuple(t) not in
                   set([tuple(tt) for tt in training_traces])]
    metrics['novelty_rate'] = len(novel_traces) / len(generated_traces)

    return metrics
```

### 5.2 Metriche di Successo Target

```
┌─────────────────────┬──────────┬────────────────────┐
│ Metrica             │ Target   │ Interpretazione    │
├─────────────────────┼──────────┼────────────────────┤
│ Validity Rate       │ > 90%    │ Syntactically OK   │
│ Fitness Mean        │ > 0.80   │ Process-conforming │
│ Conformance Mean    │ > 0.75   │ Aligns with model  │
│ Diversity Mean      │ 0.5-0.7  │ Novel but realistic│
│ Uniqueness          │ > 95%    │ No duplicates      │
│ Novelty Rate        │ > 80%    │ Not in training    │
└─────────────────────┴──────────┴────────────────────┘
```

---

## 🎯 Caso d'Uso: Purchase Order Process

### Esempio Completo

**Training Data** (5 traces):
```
Case 1: Start → Create_PO → Approve_PO → Send_PO → Receive → End
Case 2: Start → Create_PO → Reject_PO → Rework → Create_PO → Approve_PO → Send_PO → Receive → End
Case 3: Start → Create_PO → Approve_PO → Send_PO → Receive → Invoice → End
Case 4: Start → Create_PO → Approve_PO → Cancel_PO → End
Case 5: Start → Create_PO → Approve_PO → Send_PO → Receive → Return → End
```

**Generated Traces** (dopo training):
```
Gen 1: Start → Create_PO → Approve_PO → Send_PO → Receive → Invoice → Pay → End
       ├─ Validity: 1.0 (Start/End OK, length OK)
       ├─ Fitness: 0.88 (good fit)
       ├─ Conformance: 0.82 (slight deviation on Pay)
       ├─ Diversity: 0.65 (novel: adds Pay activity)
       └─ Total Reward: 0.84 ✓

Gen 2: Start → Create_PO → Approve_PO → Send_PO → Receive → Return → Rework → Create_PO → Send_PO → Receive → End
       ├─ Validity: 1.0
       ├─ Fitness: 0.75 (loop handled correctly)
       ├─ Conformance: 0.70 (complex path)
       ├─ Diversity: 0.70 (very novel: combines Return + Rework)
       └─ Total Reward: 0.79 ✓

Gen 3: Start → Create_PO → Urgent_Approve → Send_PO → Receive → End
       ├─ Validity: 1.0
       ├─ Fitness: 0.65 (Urgent_Approve not in model)
       ├─ Conformance: 0.55 (significant deviation)
       ├─ Diversity: 0.80 (very different)
       └─ Total Reward: 0.67 (acceptable, but low conformance)

Gen 4: Start → Create_PO → End  # Too simple
       ├─ Validity: 0.7 (missing core activities)
       ├─ Fitness: 0.40
       ├─ Conformance: 0.30
       ├─ Diversity: 0.90
       └─ Total Reward: 0.47 ✗ (rejected)
```

**Data Augmentation Result**:
- Original dataset: 5 traces
- Generated: 50 traces
- Valid (reward > 0.7): 41 traces
- **Final augmented dataset: 46 traces (9.2x augmentation)**

---

## 💡 Vantaggi dell'Approccio

### 1. **Architettura Generale**
✅ GAN funziona per qualsiasi tipo di grafo
✅ R-GCN cattura pattern complessi
✅ Gumbel-Softmax permette differenziabilità

### 2. **Domain-Specific Reward**
✅ Multi-objective: bilancia validità, fitness, conformance, diversity
✅ Interpretabile: ogni score ∈ [0,1] ha significato chiaro
✅ Configurabile: pesi adattabili al contesto

### 3. **Data Augmentation Efficace**
✅ Genera varianti realistiche ma diverse
✅ Mantiene conformance al processo
✅ Scale-up: da pochi esempi a molti

### 4. **Integrazione Process Mining**
✅ Compatibile con Petri nets, BPMN
✅ Usa algoritmi standard (token replay, alignment)
✅ Output utilizzabile direttamente in PM4Py, ProM

---

## 📚 Riferimenti Bibliografici

1. **MolGAN**: De Cao & Kipf (2018). "MolGAN: An implicit generative model for small molecular graphs"
2. **R-GCN**: Schlichtkrull et al. (2018). "Modeling Relational Data with Graph Convolutional Networks"
3. **Gumbel-Softmax**: Jang et al. (2017). "Categorical Reparameterization with Gumbel-Softmax"
4. **Token Replay**: van der Aalst et al. (2012). "Replaying history on process models for conformance checking"
5. **Alignment**: Adriansyah et al. (2011). "Conformance checking using cost-based fitness analysis"
6. **WGAN-GP**: Gulrajani et al. (2017). "Improved Training of Wasserstein GANs"

---

## 🚀 Prossimi Passi

1. **Implementazione**: Codificare i 5 componenti (dataset, metrics, model, optimizer, training)
2. **Testing**: Validare su dataset sintetico
3. **Tuning**: Ottimizzare hyperparameter
4. **Validation**: Confrontare traces generate con esperti dominio
5. **Scale-up**: Applicare a processi reali complessi

---

**Autore**: ProcessGAN Architecture Design
**Data**: 2025-10-11
**Versione**: 1.0
