# Guida Completa: Adattamento MolGAN per Data Augmentation di Event Logs

## Obiettivo dell'Adattamento

Trasformare MolGAN da un sistema per generare molecole a un sistema per **data augmentation di event logs** nel process mining, mantenendo l'architettura core ma adattando rappresentazioni e metriche.

**Problema**: Hai pochi event logs e vuoi generarne di nuovi per:

- Aumentare il dataset per training di modelli di process mining
- Testare robustezza di algoritmi di process discovery
- Simulare varianti di processo rare
- Bilanciare dataset sbilanciati

**Soluzione**: Adattare MolGAN per imparare la distribuzione degli event logs esistenti e generarne di nuovi plausibili.

## Analogie Fondamentali

| Dominio Molecolare       | Dominio Process Mining  | Mapping                |
| ------------------------ | ----------------------- | ---------------------- |
| **Atomi**                | **Attività/Eventi**     | Nodi del grafo         |
| **Legami chimici**       | **Flussi di controllo** | Archi del grafo        |
| **Molecola**             | **Trace/Case**          | Grafo completo         |
| **Validità chimica**     | **Conformità processo** | Vincoli di validazione |
| **Proprietà molecolari** | **KPI processo**        | Metriche di qualità    |

### Vantaggi di MolGAN per Data Augmentation

1. **Apprendimento della distribuzione**: Cattura pattern complessi nei processi
2. **Controllo della generazione**: Può generare varianti specifiche
3. **Qualità garantita**: Metriche di validazione per event logs
4. **Scalabilità**: Genera migliaia di nuovi logs da pochi esempi

## Analisi Dettagliata delle Modifiche Necessarie

### 1. Trasformazione delle Rappresentazioni

#### 1.1 Da Atomi ad Attività

```
PRIMA (Molecole):
atom_types = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br']
atom_encoder = {atom: idx for idx, atom in enumerate(atom_types)}

DOPO (Event Logs):
activity_types = ['Start', 'End', 'Submit', 'Review', 'Approve', 'Reject', 'Process']
activity_encoder = {activity: idx for idx, activity in enumerate(activity_types)}
```

**Modifiche in `sparse_molecular_dataset.py`:**

- **Linea ~85**: Sostituire `atom_labels` con `activity_labels`
- **Funzione `_generate_encoders_decoders()`**: Creare encoder per attività invece di atomi
- **Dimensione encoding**: Da ~20 tipi di atomi a ~10-15 tipi di attività

#### 1.2 Da Legami a Flussi di Controllo

```
PRIMA (Molecole):
bond_types = [SINGLE, DOUBLE, TRIPLE, AROMATIC]
rappresentano: intensità del legame chimico

DOPO (Event Logs):
flow_types = [SEQUENCE, PARALLEL, CHOICE, LOOP, SKIP]
rappresentano: relazioni temporali/logiche tra attività
```

**Interpretazione dei Flussi:**

- **SEQUENCE**: A → B (esecuzione sequenziale)
- **PARALLEL**: A → B & C (fork parallelo)
- **CHOICE**: A → B OR C (scelta esclusiva)
- **LOOP**: B → A (ciclo/iterazione)
- **SKIP**: A → C (B opzionale)

#### 1.3 Da Molecole a Traces

```
PRIMA (Molecole):
data_structure = {
    'adjacency': [9, 9, 4],  # max 9 atomi, 4 tipi legame
    'nodes': [9, 7],         # max 9 atomi, 7 tipi atomo
    'features': [9, 5]       # 5 features per atomo
}

DOPO (Event Logs):
data_structure = {
    'adjacency': [15, 15, 5],  # max 15 attività, 5 tipi flusso
    'nodes': [15, 12],         # max 15 attività, 12 tipi attività
    'features': [15, 8]        # 8 features per attività
}
```

### 2. Adattamento del Data Pipeline

#### 2.1 Input Data Format

```python
# PRIMA: Formato molecolare (SMILES/SDF)
"CCO"  # Etanolo
"C1=CC=CC=C1"  # Benzene

# DOPO: Formato event log (CSV/XES)
case_id,activity,timestamp,resource
1,Start,2023-01-01 09:00,User1
1,Submit,2023-01-01 09:15,User1
1,Review,2023-01-01 10:00,Manager
1,Approve,2023-01-01 10:30,Manager
1,End,2023-01-01 10:45,System
```

#### 2.2 Parsing e Conversione

```python
# PRIMA: RDKit per parsing molecole
def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    adjacency = Chem.GetAdjacencyMatrix(mol)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return adjacency, atoms

# DOPO: Pandas per parsing event logs
def trace_to_graph(case_events):
    activities = case_events['activity'].tolist()
    # Crea adjacency matrix per flusso sequenziale
    adjacency = create_sequence_adjacency(activities)
    # Encode attività come nodi
    nodes = [activity_encoder[act] for act in activities]
    return adjacency, nodes
```

#### 2.3 Features Engineering

```python
# PRIMA: Features chimiche
molecular_features = [
    'atomic_number', 'valence', 'formal_charge',
    'hybridization', 'aromaticity'
]

# DOPO: Features di processo
process_features = [
    'activity_duration',     # Durata attività
    'resource_type',         # Tipo risorsa
    'criticality',          # Criticità nel processo
    'automation_level',     # Livello automazione
    'cost',                 # Costo attività
    'frequency',            # Frequenza esecuzione
    'variability',          # Variabilità temporale
    'dependency_count'      # Numero dipendenze
]
```

## File da Modificare (Roadmap)

### 1. 📊 **Dataset Management** - `utils/sparse_molecular_dataset.py`

**Modifiche Necessarie:**

```python
class SparseProcessDataset(SparseMolecularDataset):  # Eredita da classe base

    def __init__(self):
        # Sostituire encoding atomi/legami con encoding attività/flussi
        self.activity_encoder = {}  # Mappa attività -> indici
        self.flow_encoder = {}      # Mappa tipi di flusso -> indici

    def load_event_logs(self, filename):
        # Carica file XES, CSV o altri formati process mining
        # Invece di molecole, lavora con trace/case
        pass

    def trace_to_graph(self, trace):
        # Converte una trace in matrice di adiacenza + nodi
        # trace = [A, B, C, D] -> matrice NxN + features
        pass
```

**Dove agire:**

- **Linea ~35-70**: Sostituire `_generate_encoders_decoders()` per attività
- **Linea ~85-120**: Modificare `_generate_AX()` per grafi di processo
- **Linea ~250-290**: Cambiare `matrices2mol()` in `matrices2trace()`

### 2. 🧠 **Model Architecture** - `models/gan.py` e `models/vae.py`

**Modifiche Minori:** (Le architetture GAN/VAE rimangono valide)

```python
class ProcessGANModel(GraphGANModel):  # Eredita architettura
    def __init__(self, max_activities, flow_types, activity_types, ...):
        # max_activities: invece di max 9 atomi, max N attività
        # flow_types: tipi di flusso tra attività
        # activity_types: tipi di attività del processo
        super().__init__(max_activities, flow_types, activity_types, ...)
```

**Dove agire:**

- **Parametri costruttore**: Cambiare `vertexes`, `edges`, `nodes` con nomi appropriati
- **Mantenere**: Tutta la logica GAN/VAE (funziona per qualsiasi grafo)

### 3. 📏 **Metrics** - `utils/molecular_metrics.py`

**Sostituire Completamente:**

```python
class ProcessMetrics:

    @staticmethod
    def conformance_score(traces, reference_model):
        """Aderenza al modello di riferimento"""
        scores = []
        for trace in traces:
            # Algoritmi di conformance checking (es. token replay)
            score = token_based_replay(trace, reference_model)
            scores.append(score)
        return np.array(scores)

    @staticmethod
    def diversity_score(traces):
        """Diversità tra traces usando edit distance"""
        diversity_scores = []
        for i, trace1 in enumerate(traces):
            distances = []
            for j, trace2 in enumerate(traces):
                if i != j:
                    dist = edit_distance(trace1, trace2)
                    distances.append(dist)
            avg_distance = np.mean(distances) if distances else 0
            diversity_scores.append(avg_distance)
        return np.array(diversity_scores)

    @staticmethod
    def business_value_score(traces):
        """Valore business del processo"""
        scores = []
        for trace in traces:
            value = 0
            # Esempio: bonus per automation
            if 'AutoApprove' in trace:
                value += 0.3
            # Penalità per rework
            if 'Rework' in trace:
                value -= 0.2
            # Bonus per completezza
            if trace[0] == 'Start' and trace[-1] == 'End':
                value += 0.5
            scores.append(max(0, min(1, value)))
        return np.array(scores)

    @staticmethod
    def valid_scores(traces):
        """Validità dei traces secondo regole business"""
        scores = []
        for trace in traces:
            if not trace:
                scores.append(0.0)
                continue

            # Regole di validità
            is_valid = (
                len(trace) >= 3 and
                trace[0] == 'Start' and
                trace[-1] == 'End'
            )
            scores.append(1.0 if is_valid else 0.0)
        return np.array(scores)

    @staticmethod
    def fitness_scores(traces):
        """Fitness business del processo"""
        scores = []
        for trace in traces:
            if not trace:
                scores.append(0.0)
                continue

            score = 1.0

            # Penalizza se mancano attività core
            if 'Register' not in trace:
                score *= 0.8
            if 'Submit' not in trace:
                score *= 0.7

            # Bonus per flussi logici
            if 'Review' in trace and any(x in trace for x in ['Approve', 'Reject']):
                score *= 1.2

            scores.append(min(score, 1.0))
        return np.array(scores)
```

**Dove agire:**

- **Tutto il file**: Sostituire metriche chimiche con metriche di process mining
- **Rimuovere**: Dipendenze da RDKit
- **Aggiungere**: Dipendenze da PM4Py o ProM

### 4. 🔧 **Example Usage** - `example.py`

**Modificare Setup:**

```python
# Invece di:
data = SparseMolecularDataset()
data.load('data/gdb9_9nodes.sparsedataset')

# Usare:
data = SparseProcessDataset()
data.load('data/event_logs.csv')  # o .xes

# Modificare reward function
def process_reward(traces):
    reward = 1.0
    for metric in ['fitness', 'precision', 'conformance']:
        if metric == 'fitness':
            reward *= ProcessMetrics.fitness_score(traces)
        elif metric == 'precision':
            reward *= ProcessMetrics.precision_score(traces, reference_model)
        # ...
    return reward
```

## Reward Function per Data Augmentation

### Multi-Objective Reward

```python
def compute_reward(traces, original_traces, weights={'validity': 0.4, 'diversity': 0.3, 'fitness': 0.3}):
    """
    Reward function bilanciata per data augmentation
    """
    # Validità: traces devono essere sintatticamente corretti
    validity = ProcessMetrics.valid_scores(traces)

    # Diversità: traces devono essere diversi dagli originali
    diversity = ProcessMetrics.diversity_score(traces, original_traces)

    # Fitness: traces devono rispettare logica business
    fitness = ProcessMetrics.business_value_score(traces)

    # Reward pesato
    total_reward = (weights['validity'] * validity +
                   weights['diversity'] * diversity +
                   weights['fitness'] * fitness)

    return total_reward.reshape(-1, 1)
```

### Reward Adaptativi

```python
def adaptive_reward(epoch, traces, target_metrics):
    """
    Reward che si adatta durante il training
    """
    current_validity = np.mean(ProcessMetrics.valid_scores(traces))
    current_diversity = np.mean(ProcessMetrics.diversity_score(traces))

    # Aumenta peso diversità se validità è buona
    if current_validity > 0.8:
        diversity_weight = 0.5
        validity_weight = 0.3
    else:
        diversity_weight = 0.2
        validity_weight = 0.6

    return compute_reward(traces, weights={
        'validity': validity_weight,
        'diversity': diversity_weight,
        'fitness': 0.2
    })
```

## Configurazioni Specifiche per Data Augmentation

### Hyperparameter Tuning

```python
# Configurazione ottimizzata per data augmentation
augmentation_config = {
    # Modello
    'max_activities': 20,           # Processi più lunghi delle molecole
    'embedding_dim': 24,            # Spazio latente più ricco
    'decoder_units': (128, 256, 512),  # Capacità maggiore

    # Training
    'batch_size': 32,               # Batch più grandi per stabilità
    'learning_rate': 5e-4,          # LR più conservativo
    'n_critic': 3,                  # Meno passi discriminator
    'epochs': 200,                  # Più epoche per convergenza

    # Data Augmentation
    'augmentation_factor': 5,       # 5x più dati
    'diversity_threshold': 0.7,     # Soglia diversità minima
    'validity_threshold': 0.9,      # Soglia validità minima

    # Reward weights
    'validity_weight': 0.5,         # Priorità alla validità
    'diversity_weight': 0.3,        # Importante per augmentation
    'fitness_weight': 0.2           # Business value
}
```

### Early Stopping Criteri

```python
def augmentation_early_stopping(metrics_history, patience=10):
    """
    Early stopping specifico per data augmentation
    """
    # Stop se validità non migliora
    validity_trend = metrics_history['validity'][-patience:]
    if len(validity_trend) == patience and max(validity_trend) < 0.8:
        return True, "Low validity"

    # Stop se diversità collassa
    diversity_trend = metrics_history['diversity'][-patience:]
    if len(diversity_trend) == patience and max(diversity_trend) < 0.5:
        return True, "Diversity collapse"

    # Stop se convergenza
    if len(validity_trend) == patience:
        validity_std = np.std(validity_trend)
        if validity_std < 0.01:  # Variazione < 1%
            return True, "Converged"

    return False, "Continue"
```

## Validazione e Quality Assurance

### Validation Pipeline

```python
def validate_augmented_traces(generated_traces, original_traces, domain_knowledge):
    """
    Pipeline di validazione completa
    """
    validation_results = {}

    # 1. Validazione sintattica
    syntax_valid = [validate_syntax(trace) for trace in generated_traces]
    validation_results['syntax_validity'] = np.mean(syntax_valid)

    # 2. Validazione semantica
    semantic_valid = [validate_semantics(trace, domain_knowledge) for trace in generated_traces]
    validation_results['semantic_validity'] = np.mean(semantic_valid)

    # 3. Diversità da originali
    diversity_scores = compute_diversity_matrix(generated_traces, original_traces)
    validation_results['diversity_mean'] = np.mean(diversity_scores)
    validation_results['diversity_std'] = np.std(diversity_scores)

    # 4. Copertura pattern originali
    original_patterns = extract_patterns(original_traces)
    generated_patterns = extract_patterns(generated_traces)
    pattern_coverage = len(original_patterns & generated_patterns) / len(original_patterns)
    validation_results['pattern_coverage'] = pattern_coverage

    # 5. Distribuzione lunghezze
    orig_lengths = [len(trace) for trace in original_traces]
    gen_lengths = [len(trace) for trace in generated_traces]
    length_ks_stat = ks_test(orig_lengths, gen_lengths)
    validation_results['length_distribution_similarity'] = 1 - length_ks_stat

    return validation_results
```

### Quality Gates

```python
def quality_gates_check(validation_results, thresholds):
    """
    Quality gates per assicurare qualità augmentation
    """
    gates_passed = {}

    # Gate 1: Validità minima
    gates_passed['validity'] = validation_results['syntax_validity'] >= thresholds['min_validity']

    # Gate 2: Diversità sufficiente
    gates_passed['diversity'] = validation_results['diversity_mean'] >= thresholds['min_diversity']

    # Gate 3: Coverage pattern
    gates_passed['coverage'] = validation_results['pattern_coverage'] >= thresholds['min_coverage']

    # Gate 4: Distribuzione realistica
    gates_passed['distribution'] = validation_results['length_distribution_similarity'] >= thresholds['min_similarity']

    all_passed = all(gates_passed.values())

    return all_passed, gates_passed
```

## Workflow di Training Adattato

### Data Loading

```python
# PRIMA: Caricamento molecole
data = SparseMolecularDataset()
data.load('data/molecules.sparsedataset')

# DOPO: Caricamento event logs
data = SparseProcessDataset()  # Nuova classe
data.load_from_csv('data/event_logs.csv')
# Oppure
data.load_from_xes('data/event_logs.xes')
```

### Training Loop Modificato

```python
def train_step_augmentation(model, optimizer, data, batch_size):
    """
    Training step ottimizzato per data augmentation
    """
    # 1. Batch di traces reali
    real_traces, _, _, adj_real, nodes_real, _, _, _, _ = data.next_train_batch(batch_size)

    # 2. Genera traces fake
    embeddings = model.sample_z(batch_size)
    adj_fake, nodes_fake = model.generate(embeddings)
    fake_traces = [data.matrices2trace(nodes_fake[i], adj_fake[i]) for i in range(batch_size)]

    # 3. Calcola rewards
    reward_real = compute_reward(real_traces, data.original_traces)
    reward_fake = compute_reward(fake_traces, data.original_traces)

    # 4. Training discriminator
    loss_D = optimizer.discriminator_step(adj_real, nodes_real, adj_fake, nodes_fake)

    # 5. Training generator con reward
    loss_G = optimizer.generator_step(adj_fake, nodes_fake, reward_fake)

    return loss_D, loss_G, reward_fake.mean()
```

## Roadmap di Implementazione

### Fase 1: Setup Base (1-2 giorni)

1. **Copia e rinomina** `sparse_molecular_dataset.py` → `sparse_process_dataset.py`
2. **Sostituisci encoding** atomi/legami con attività/flussi
3. **Testa caricamento** di un piccolo event log CSV

### Fase 2: Adattamento Metriche (2-3 giorni)

1. **Crea** `process_metrics.py` con metriche di processo
2. **Implementa** funzioni di validazione business
3. **Testa** calcolo reward su traces generate casualmente

### Fase 3: Integrazione Modello (3-4 giorni)

1. **Modifica** `example.py` per usare process dataset
2. **Adatta** reward function per data augmentation
3. **Testa** training su dataset piccolo

### Fase 4: Ottimizzazione (2-3 giorni)

1. **Fine-tune** hyperparameter per event logs
2. **Implementa** early stopping specifico
3. **Valida** qualità augmentation con esperti dominio

### Fase 5: Production (1-2 giorni)

1. **Crea** pipeline end-to-end
2. **Documenta** best practices
3. **Prepara** esempi d'uso

## Esempi Pratici

### Input: Event Log Limitato

```
Case 1: Start → ProcessA → Decision → End
Case 2: Start → ProcessB → Decision → Approve → End
Case 3: Start → ProcessA → Decision → Reject → End
```

### Output: Event Logs Aumentati

```
Generated Case 1: Start → ProcessA → Decision → Approve → End
Generated Case 2: Start → ProcessB → Decision → End
Generated Case 3: Start → ProcessA → ProcessB → Decision → End
Generated Case 4: Start → ProcessA → Review → Decision → Approve → End
Generated Case 5: Start → ProcessB → Check → Decision → Reject → Rework → Decision → Approve → End
...
```

### Encoding Strategy

```python
# Attività frequenti nel process mining
activity_types = [
    'Start', 'End',                    # Attività standard
    'Register', 'Submit', 'Review',    # Attività core
    'Approve', 'Reject', 'Rework',     # Decisioni
    'Send', 'Receive', 'Process',      # Comunicazioni
    'Check', 'Validate', 'Confirm'    # Verifiche
]

# Tipi di flusso tra attività
flow_types = [
    'direct_sequence',     # A → B
    'parallel_split',      # A → B & C
    'parallel_join',       # B & C → D
    'choice',             # A → B OR C
    'loop_back',          # B → A (ciclo)
    'skip'                # A → C (salta B)
]
```

## Conclusioni e Vantaggi

### Vantaggi dell'Approccio

1. **Riuso Architettura**: L'architettura GAN/VAE è ottimale per grafi
2. **Flessibilità**: Facilmente adattabile a diversi domini di processo
3. **Scalabilità**: Può gestire processi complessi con molte attività
4. **Quality Control**: Metriche integrate per validazione automatica

### Limitazioni da Considerare

1. **Dimensioni Fisse**: Limitato a processi di lunghezza massima fissa
2. **Pattern Complessi**: Difficoltà con pattern very complex (e.g., nested loops)
3. **Domain Knowledge**: Richiede encoding accurato delle regole business
4. **Training Data**: Necessita di dataset sufficientemente grandi per convergenza

### Metriche di Successo per Data Augmentation

- **Validity Rate**: % di traces generati validi (target: >90%)
- **Diversity Score**: Diversità rispetto originali (target: 0.6-0.8)
- **Coverage**: Pattern originali preservati (target: >95%)
- **Augmentation Ratio**: Moltiplicatore dataset (target: 3-5x)

Questo approccio fornisce una soluzione robusta e scalabile per data augmentation di event logs, sfruttando la potenza delle reti generative mantenendo la specificità del dominio process mining.

# Modificare reward function

def process_reward(traces):
reward = 1.0
for metric in ['fitness', 'precision', 'conformance']:
if metric == 'fitness':
reward _= ProcessMetrics.fitness_score(traces)
elif metric == 'precision':
reward _= ProcessMetrics.precision_score(traces, reference_model) # ...
return reward

````

## Piano di Implementazione Passo-Passo

### Phase 1: Preparazione Dataset 🗃️

1. **Creare `utils/process_dataset.py`**:

   ```bash
   cp utils/sparse_molecular_dataset.py utils/process_dataset.py
````

2. **Modificare encoding**:

   - Sostituire atomi con attività
   - Sostituire legami con tipi di flusso
   - Adattare matrici per process graph

3. **Testare caricamento**:
   ```python
   data = ProcessDataset()
   data.load_from_csv('sample_log.csv')
   print(f"Loaded {len(data)} process instances")
   ```

### Phase 2: Adattamento Metriche 📊

1. **Creare `utils/process_metrics.py`**:

   ```bash
   touch utils/process_metrics.py
   ```

2. **Implementare metriche base**:
   - Valid trace (sintassi corretta)
   - Fitness (aderenza al modello)
   - Uniqueness (diversità)

### Phase 3: Modifiche Modello 🧠

1. **Mantenere architetture GAN/VAE** (funzionano per qualsiasi grafo)
2. **Cambiare solo parametri**:
   - `vertexes` = max_activities
   - `edges` = flow_types
   - `nodes` = activity_types

### Phase 4: Testing e Validation ✅

1. **Testare generazione**:

   ```python
   # Generare nuove trace
   embeddings = model.sample_z(100)
   traces = generate_traces(embeddings)

   # Valutare qualità
   fitness = ProcessMetrics.fitness_score(traces)
   print(f"Generated traces fitness: {fitness}")
   ```

## File di Input Suggeriti

### Formato CSV Event Log

```csv
case_id,activity,timestamp,resource
1,Start,2023-01-01 10:00,User1
1,Process,2023-01-01 10:15,System
1,Approve,2023-01-01 10:30,Manager
1,End,2023-01-01 10:45,System
2,Start,2023-01-01 11:00,User2
...
```

### Formato Graph Adjacency

```python
# Trace: Start -> Process -> Approve -> End
adjacency_matrix = [
    [0, 1, 0, 0],  # Start -> Process
    [0, 0, 1, 0],  # Process -> Approve
    [0, 0, 0, 1],  # Approve -> End
    [0, 0, 0, 0]   # End -> nothing
]

node_features = [
    [1, 0, 0, 0],  # Start activity
    [0, 1, 0, 0],  # Process activity
    [0, 0, 1, 0],  # Approve activity
    [0, 0, 0, 1]   # End activity
]
```

## Vantaggi dell'Approccio

1. **Riuso Architettura**: GAN/VAE funzionano per qualsiasi grafo
2. **Scalabilità**: Può gestire processi complessi
3. **Metriche Specifiche**: Adattate al domain del process mining
4. **Generazione Controllata**: Reinforcement learning per vincoli di processo

## Prossimi Passi Concreti

1. **Inizia con Phase 1**: Adatta il dataset management
2. **Crea un piccolo test**: Carica 5-10 process instances
3. **Testa la pipeline**: Verifica che le matrici siano generate correttamente
4. **Implementa metriche base**: Valid trace percentage
5. **Adatta gradualmente**: Un componente alla volta

Vuoi che iniziamo subito con la modifica di un file specifico?
