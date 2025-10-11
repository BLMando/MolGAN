# ProcessGAN: Data Augmentation per Event Logs

**ProcessGAN** è un adattamento di MolGAN per generare event logs sintetici nel contesto del process mining. Utilizza Generative Adversarial Networks (GAN) con Reinforcement Learning per generare traces di processo plausibili per data augmentation.

---

## 🎯 Obiettivo

Generare event logs sintetici di alta qualità per:
- **Data Augmentation**: Aumentare dataset limitati di process mining
- **Testing**: Creare varianti di processo per validazione algoritmi
- **Simulazione**: Generare scenari rari o edge cases
- **Bias Mitigation**: Bilanciare dataset sbilanciati

---

## 📁 Struttura del Progetto

```
MolGAN/
├── Docs/
│   ├── ARCHITETTURA_PROCESSGGAN.md          # Documentazione dettagliata architettura
│   ├── DOCUMENTAZIONE_MOLGAN.md             # Documentazione MolGAN originale
│   └── GUIDA_ADATTAMENTO_PROCESS_MINING.md  # Guida adattamento
│
├── utils/
│   ├── process_dataset.py                   # Dataset management per event logs
│   └── process_metrics.py                   # Reward function multi-objective
│
├── models/
│   └── process_gan.py                       # Generator + Discriminator + Value Network
│
├── optimizers/
│   └── process_optimizer.py                 # Optimizer con WGAN-GP + RL
│
├── data/
│   └── sample_event_log.csv                 # Dataset di esempio (20 cases)
│
├── tests/
│   └── test_process_reward.py               # Unit tests (30 test cases)
│
├── example_process.py                       # Script training completo
└── README_PROCESSGGAN.md                    # Questo file
```

---

## 🚀 Quick Start

### 1. Installazione Dipendenze

```bash
# Creare ambiente conda (richiede Python 3.6 per compatibilità TF 1.8)
conda env create -f environment.yml
conda activate MolGAN

# Oppure con pip (per testing senza TensorFlow)
pip install numpy pandas scikit-learn
```

### 2. Preparare Dataset

Il formato richiesto è CSV con colonne: `case_id, activity, timestamp, resource`

```csv
case_id,activity,timestamp,resource
1,Start,2023-01-01 09:00:00,System
1,Submit,2023-01-01 09:15:00,User1
1,Review,2023-01-01 10:00:00,Manager
1,Approve,2023-01-01 10:30:00,Manager
1,End,2023-01-01 10:45:00,System
...
```

Un dataset di esempio è fornito in `data/sample_event_log.csv`.

### 3. Training

```bash
python example_process.py
```

Il training salverà checkpoints in `checkpoints/process_gan/` ogni 10 epoche.

### 4. Generazione

Dopo il training, il modello genera automaticamente nuove traces. Per generare traces da un checkpoint salvato, modificare `example_process.py` per caricare il modello.

---

## 📊 Architettura

### Componente 1: Generator

Trasforma vettore latente **z ∈ ℝ^16** in trace di processo:

```
z (16 dim) → Dense(128) → Dense(256) → Dense(512)
                ↓
          ┌─────────────┐
          │ Adjacency   │ (15×15×5)  Flow control tra attività
          │ Matrix      │
          └─────────────┘
                ↓
          ┌─────────────┐
          │ Node Vector │ (15×12)    Tipi di attività
          └─────────────┘
                ↓
          Gumbel-Softmax (differenziabile)
                ↓
          Generated Trace: ['Start', 'Submit', 'Review', 'Approve', 'End']
```

### Componente 2: Discriminator

Relational Graph Convolutional Network (R-GCN) che valuta se una trace è reale o generata:

```
(Adjacency, Nodes)
       ↓
   R-GCN Layers (128, 64)
       ↓
   Graph Pooling (attention-based)
       ↓
   MLP Classifier
       ↓
   Score ∈ [0,1]
```

### Componente 3: Reward Function

Multi-objective reward **R(trace) ∈ [0,1]**:

```
R = α·Validity + β·Fitness + γ·Conformance + δ·Diversity

Dove:
├─ Validity:     Sintassi corretta (Start/End, length, no loops)
├─ Fitness:      Token replay fitness (aderenza al modello)
├─ Conformance:  Alignment cost (similarità al modello di riferimento)
└─ Diversity:    Edit distance da training set (per augmentation)

Default weights: α=0.30, β=0.25, γ=0.25, δ=0.20
```

---

## 🔧 Configurazione

Modificare il dict `config` in `example_process.py`:

```python
config = {
    # Data
    'data_file': 'data/sample_event_log.csv',
    'max_activities': 15,         # Max trace length

    # Model
    'z_dim': 16,                  # Latent dimension
    'decoder_units': (128, 256, 512),
    'discriminator_units': ((128, 64), 128, (128, 64)),

    # Training
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'n_critic': 5,                # D steps per G step

    # Reward weights
    'reward_weights': {
        'validity': 0.30,
        'fitness': 0.25,
        'conformance': 0.25,
        'diversity': 0.20
    },

    # Generation
    'n_samples_eval': 100,        # Samples per evaluation
}
```

---

## 📈 Metriche di Valutazione

### Metriche di Qualità

```python
from utils.process_metrics import ProcessRewardFunction

reward_fn = ProcessRewardFunction(training_traces=training_data)
metrics = reward_fn.evaluate_batch(generated_traces)

# Output:
# {
#   'validity_mean': 0.92,      # 92% traces sintatticamente valide
#   'fitness_mean': 0.85,       # Buon fit al modello
#   'conformance_mean': 0.78,   # Buon allineamento
#   'diversity_mean': 0.65,     # Sufficientemente diverse
#   'reward_mean': 0.80,        # Overall quality
#   'high_quality_rate': 0.75   # 75% con reward >= 0.7
# }
```

### Target di Successo

| Metrica | Target | Interpretazione |
|---------|--------|-----------------|
| Validity Rate | > 90% | Traces sintatticamente corrette |
| Fitness Mean | > 0.80 | Aderenza al processo |
| Conformance Mean | > 0.75 | Allineamento con modello |
| Diversity Mean | 0.5-0.7 | Novità ma realistica |
| High Quality Rate | > 70% | % con reward >= 0.7 |

---

## 🧪 Testing

### Unit Tests

```bash
# Esegui tutti i test
python tests/test_process_reward.py

# Output atteso:
# ======================================================================
# Tests run: 30
# Successes: 30
# Failures: 0
# Errors: 0
# ✓ All tests passed!
# ======================================================================
```

### Test Componenti Singoli

```bash
# Test dataset
python utils/process_dataset.py

# Test reward function
python utils/process_metrics.py

# Test modello (richiede TensorFlow)
python models/process_gan.py
```

---

## 📖 Esempi d'Uso

### Esempio 1: Caricamento Dataset

```python
from utils.process_dataset import ProcessDataset

# Carica da CSV
data = ProcessDataset(max_activities=15)
data.load_from_csv('data/sample_event_log.csv', validation=0.1, test=0.1)

# Stampa statistiche
stats = data.get_stats()
print(f"Total traces: {stats['total_traces']}")
print(f"Activities: {stats['activity_types']}")
```

### Esempio 2: Valutazione Trace

```python
from utils.process_metrics import ProcessRewardFunction

# Crea reward function
reward_fn = ProcessRewardFunction(training_traces=training_data)

# Valuta singola trace
trace = ['Start', 'Submit', 'Review', 'Approve', 'End']
reward_fn.print_trace_analysis(trace)

# Output:
# ============================================================
# Trace Analysis: Start → Submit → Review → Approve → End
# ============================================================
# Validity:     1.000 (α=0.30)
# Fitness:      0.850 (β=0.25)
# Conformance:  0.800 (γ=0.25)
# Diversity:    0.600 (δ=0.20)
# ------------------------------------------------------------
# Total Reward: 0.813
# Quality:      Excellent ✓✓
# ============================================================
```

### Esempio 3: Generazione Traces

```python
import tensorflow as tf
from models.process_gan import ProcessGANModel, matrices_to_traces

# Carica modello (dopo training)
model = ProcessGANModel(...)
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'checkpoints/process_gan/model_final.ckpt')

    # Genera samples
    z = model.sample_z(100)
    nodes, edges = sess.run(
        [model.nodes_argmax, model.edges_argmax],
        feed_dict={model.embeddings: z, model.training: False}
    )

    # Converti in traces
    traces = matrices_to_traces(edges, np.argmax(nodes, axis=-1), dataset)

    # Filtra traces di alta qualità
    high_quality = [t for t, r in zip(traces, rewards) if r >= 0.7]
```

---

## 🎓 Per Approfondire

### Documentazione Completa

- **[ARCHITETTURA_PROCESSGGAN.md](Docs/ARCHITETTURA_PROCESSGGAN.md)**: Spiegazione dettagliata di ogni componente con formule matematiche, esempi e intuizioni
- **[GUIDA_ADATTAMENTO_PROCESS_MINING.md](Docs/GUIDA_ADATTAMENTO_PROCESS_MINING.md)**: Guida step-by-step per adattare MolGAN al process mining

### Componenti Chiave

1. **ProcessDataset** ([process_dataset.py](utils/process_dataset.py)):
   - Carica event logs da CSV/XES
   - Converte traces in grafi (adjacency + nodes)
   - Gestisce encoding/decoding attività

2. **ProcessRewardFunction** ([process_metrics.py](utils/process_metrics.py)):
   - 4 sub-rewards: Validity, Fitness, Conformance, Diversity
   - Configurabile tramite pesi
   - Output [0,1] interpretabile

3. **ProcessGANModel** ([process_gan.py](models/process_gan.py)):
   - Generator: z → (adjacency, nodes)
   - Discriminator: R-GCN → score [0,1]
   - Value Network: per RL reward prediction

4. **ProcessGANOptimizer** ([process_optimizer.py](optimizers/process_optimizer.py)):
   - Wasserstein GAN + Gradient Penalty
   - Integrazione Reinforcement Learning
   - Alternanza discriminator/generator (5:1)

---

## 🔬 Approccio Scientifico

### Architettura GAN + RL

```
PHASE 1: Adversarial Training
├─ Discriminator impara a distinguere traces reali/fake
└─ Generator impara a ingannare discriminator

PHASE 2: Reinforcement Learning
├─ Reward function valuta qualità traces generate
├─ Generator ottimizza per massimizzare reward
└─ Value network stima reward attesi

LOSS TOTALE:
L_G = λ_adv · L_adversarial + λ_reward · L_RL
```

### Vantaggi Rispetto ad Altri Approcci

✅ **vs Random Sampling**: Genera traces realistiche, non casuali
✅ **vs Rule-Based**: Apprende pattern complessi dai dati
✅ **vs VAE**: Maggiore diversità grazie a adversarial training
✅ **vs Pure GAN**: Controllo qualità tramite reward function
✅ **vs Simulazione**: Non richiede modello esplicito del processo

---

## ⚠️ Limitazioni

1. **Dimensione Fissa**: Max 15 attività per trace (configurabile)
2. **TensorFlow 1.x**: Richiede versione obsoleta (compatibilità MolGAN)
3. **Training Data**: Necessita almeno 50-100 traces per convergenza
4. **Pattern Complessi**: Difficoltà con loop annidati o strutture molto complesse
5. **Domain Knowledge**: Reward function richiede tuning per dominio specifico

---

## 🛠️ Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'tensorflow'"

**Soluzione**: Installare TensorFlow 1.8 (compatibile con Python 3.6)

```bash
conda create -n molgan python=3.6
conda activate molgan
pip install tensorflow==1.8.0
```

### Problema: "Reward sempre < 0.5"

**Soluzione**: Regolare pesi reward function o aumentare epoche training

```python
config['reward_weights'] = {
    'validity': 0.40,  # Aumenta peso validità
    'fitness': 0.30,
    'conformance': 0.20,
    'diversity': 0.10   # Riduci peso diversità
}
```

### Problema: "Mode collapse (tutte traces identiche)"

**Soluzione**:
- Aumentare `n_critic` (più step discriminator)
- Ridurre `learning_rate`
- Aumentare peso `diversity` nella reward function

---

## 📚 Riferimenti

1. **MolGAN**: De Cao & Kipf (2018). "MolGAN: An implicit generative model for small molecular graphs"
2. **R-GCN**: Schlichtkrull et al. (2018). "Modeling Relational Data with Graph Convolutional Networks"
3. **WGAN-GP**: Gulrajani et al. (2017). "Improved Training of Wasserstein GANs"
4. **Process Mining**: van der Aalst (2016). "Process Mining: Data Science in Action"

---

## 👥 Contributi

Questo progetto è un adattamento di [MolGAN](https://github.com/nicola-decao/MolGAN) per il process mining.

**Modifiche principali**:
- Sostituzione encoding atomi/legami con attività/flussi
- Implementazione reward function multi-objective per process mining
- Adattamento dataset loader per event logs (CSV/XES)
- Unit tests completi per validazione

---

## 📄 Licenza

MIT License (come progetto originale MolGAN)

---

## 🎯 Next Steps

1. **Training**: Esegui `python example_process.py` per vedere il sistema in azione
2. **Customizzazione**: Modifica `config` in `example_process.py` per il tuo dominio
3. **Tuning**: Sperimenta con diversi pesi reward e hyperparameter
4. **Evaluation**: Valuta traces generate con esperti di dominio
5. **Production**: Integra con pipeline PM4Py/ProM per analisi avanzate

---

**Domande? Consulta [ARCHITETTURA_PROCESSGGAN.md](Docs/ARCHITETTURA_PROCESSGGAN.md) per dettagli tecnici completi!**
