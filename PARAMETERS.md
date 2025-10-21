# Spiegazione Dettagliata dei Parametri Config - ProcessGAN

## 📊 PARAMETRI DATI (Data Parameters)

### `data_file: 'data/helpdesk_parsed.xes'`
**Cosa fa:** Specifica il percorso del file di input contenente i log degli eventi.

**Modifica comporta:**
- Cambiare dataset di training
- Diversi pattern di processo da apprendere
- Diverse dimensioni del dataset (numero tracce)

**Perché:** Il modello apprende i pattern specifici del processo contenuto in questo file.

---

### `max_activities: 10`
**Cosa fa:** Definisce la lunghezza massima delle tracce (numero massimo di attività per sequenza).

**Modifica comporta:**
- **Aumentare (es. 15, 20):**
  - ✅ Copre tracce più lunghe del dataset
  - ❌ Matrici più grandi (max×max per adiacenza)
  - ❌ Più parametri nel modello (crescita quadratica)
  - ❌ Più padding waste (se tracce medie sono corte)
  - ❌ Training più lento
  - ❌ Maggior rischio di overfitting
  
- **Diminuire (es. 5, 8):**
  - ✅ Training più veloce
  - ✅ Meno parametri (meno overfitting)
  - ✅ Meno memoria richiesta
  - ❌ Tracce lunghe vengono troncate/escluse
  - ❌ Perdita di informazione

**Perché:** Le matrici di adiacenza sono max_activities × max_activities. Se max=15, hai 15²=225 elementi per matrice. Se max=10, hai 10²=100 elementi (56% in meno).

**Regola empirica:** Usa il 95° percentile della distribuzione delle lunghezze delle tracce nel dataset.

---

### `validation_split: 0.1` (10%)
**Cosa fa:** Percentuale del dataset riservata alla validazione.

**Modifica comporta:**
- **Aumentare (es. 0.2):**
  - ✅ Validazione più robusta
  - ❌ Meno dati per training
  
- **Diminuire (es. 0.05):**
  - ✅ Più dati per training
  - ❌ Validazione meno affidabile

**Perché:** Split standard è 80/10/10 (train/val/test). Con dataset piccoli (<1000 tracce), considera 0.15-0.2 per validazione più stabile.

---

### `test_split: 0.1` (10%)
**Cosa fa:** Percentuale del dataset riservata al test finale.

**Modifica comporta:** Stesso ragionamento di `validation_split`.

**Perché:** Il test set valuta la performance finale su dati mai visti durante training/validazione.

---

## 🏗️ PARAMETRI ARCHITETTURA MODELLO (Model Architecture)

### `z_dim: 64`
**Cosa fa:** Dimensione dello spazio latente (vettore di rumore casuale input al Generator).

**Modifica comporta:**
- **Aumentare (es. 128, 256):**
  - ✅ Maggiore capacità rappresentativa
  - ✅ Può catturare pattern più complessi
  - ❌ Più parametri nel Generator
  - ❌ Rischio overfitting con dataset piccoli
  - ❌ Training più lento
  
- **Diminuire (es. 16, 32):**
  - ✅ Modello più compatto
  - ✅ Training più veloce
  - ❌ Capacità rappresentativa limitata
  - ❌ Difficoltà a catturare variabilità complessa

**Perché:** z_dim controlla la "ricchezza" dello spazio latente da cui il Generator campiona. Più alto = più variazioni possibili, ma serve più dati per evitare overfitting.

**Regola empirica:** 
- Dataset piccoli (<5000 tracce): 16-64
- Dataset medi (5000-50000): 64-128
- Dataset grandi (>50000): 128-256

---

### `decoder_units: (128, 256, 256)`
**Cosa fa:** Architettura del Generator (numero neuroni per layer Dense).

**Modifica comporta:**
- **Aumentare (es. (256, 512, 512)):**
  - ✅ Maggiore capacità di modellazione
  - ❌ Molti più parametri (rischio overfitting)
  - ❌ Training molto più lento
  - ❌ Richiede dataset grande
  
- **Diminuire (es. (64, 128, 128)):**
  - ✅ Modello più leggero
  - ✅ Training più veloce
  - ❌ Potrebbe non catturare pattern complessi

**Perché:** Questi layer trasformano il vettore latente z in matrici di adiacenza e nodi. Più neuroni = più espressività, ma serve bilanciare con dimensione dataset.

**Calcolo parametri:** 
- Layer 1: z_dim × 128 = 64 × 128 = 8,192 parametri
- Layer 2: 128 × 256 = 32,768 parametri
- Layer 3: 256 × 256 = 65,536 parametri
- **Totale: ~106K parametri solo nel decoder**

---

### `discriminator_units: (128, 128)`
**Cosa fa:** Architettura del Discriminator (dopo R-GCN layers).

**Modifica comporta:**
- **Aumentare (es. (256, 256)):**
  - ✅ Discriminator più potente (critica migliore)
  - ❌ Rischio di dominare il Generator (mode collapse)
  - ❌ Training instabile
  
- **Diminuire (es. (64, 64)):**
  - ✅ Bilanciamento migliore con Generator debole
  - ❌ Critica meno accurata

**Perché:** Il Discriminator deve essere abbastanza forte da guidare il Generator, ma non troppo da "schiacciarlo". Regola generale: discriminator_units ≤ decoder_units.

---

### `mlp_units: 128`
**Cosa fa:** Dimensione hidden layer nel Value Network (RL component).

**Modifica comporta:**
- **Aumentare (es. 256):**
  - ✅ Stima reward più accurata
  - ❌ Più parametri
  
- **Diminuire (es. 64):**
  - ✅ Più leggero
  - ❌ Stima reward meno precisa

**Perché:** Il Value Network stima il reward atteso. Più complesso = stime migliori, ma richiede più dati.

---

### `dropout_rate: 0.1` (10%)
**Cosa fa:** Percentuale di neuroni "spenti" casualmente durante training (regolarizzazione).

**Modifica comporta:**
- **Aumentare (es. 0.3, 0.5):**
  - ✅ Maggiore regolarizzazione (meno overfitting)
  - ❌ Training più lento (convergenza più difficile)
  - ❌ Potrebbe underfittare
  
- **Diminuire (es. 0.0):**
  - ✅ Convergenza più veloce
  - ❌ Rischio overfitting alto

**Perché:** Dropout previene co-adattamento dei neuroni, forzando il modello a imparare rappresentazioni robuste.

**Regola empirica:**
- Dataset grande: 0.1-0.2
- Dataset piccolo: 0.2-0.5
- Overfitting evidente: aumenta a 0.3-0.5

---

## 🎓 PARAMETRI TRAINING (Training Parameters)

### `batch_size: 32`
**Cosa fa:** Numero di tracce processate insieme in ogni step di training.

**Modifica comporta:**
- **Aumentare (es. 64, 128):**
  - ✅ Gradienti più stabili (meno rumore)
  - ✅ Training più veloce (meno step per epoca)
  - ✅ Migliore utilizzo GPU
  - ❌ Richiede più memoria RAM/VRAM
  - ❌ Potrebbe convergere a minimi locali peggiori
  
- **Diminuire (es. 8, 16):**
  - ✅ Meno memoria richiesta
  - ✅ Gradienti più "rumorosi" (può aiutare esplorazione)
  - ❌ Training più lento (più step per epoca)
  - ❌ Gradienti instabili

**Perché:** Batch size bilancia stabilità vs velocità. Batch grandi = gradienti più accurati ma meno esplorazione. Batch piccoli = più rumore ma migliore generalizzazione.

**Regola empirica:**
- GPU con 8GB VRAM: 32-64
- GPU con 16GB VRAM: 64-128
- CPU only: 16-32

---

### `epochs: 50`
**Cosa fa:** Numero di passaggi completi sul dataset di training.

**Modifica comporta:**
- **Aumentare (es. 100, 200):**
  - ✅ Più tempo per convergere
  - ✅ Risultati migliori (se non overfit)
  - ❌ Training molto più lungo
  - ❌ Rischio overfitting
  
- **Diminuire (es. 20, 30):**
  - ✅ Training veloce
  - ❌ Modello non converge (underfitting)

**Perché:** Le GAN richiedono molte iterazioni per stabilizzarsi. 50 epoch è un compromesso. Per risultati ottimali: 100-200 epoch.

**Timeline:**
- 5 epoch: ~2 min (test only, risultati pessimi)
- 50 epoch: ~20 min (buoni risultati)
- 100 epoch: ~40 min (risultati ottimali)

---

### `learning_rate: 3e-4` (0.0003)
**Cosa fa:** Dimensione del passo di aggiornamento dei pesi durante training.

**Modifica comporta:**
- **Aumentare (es. 1e-3, 5e-3):**
  - ✅ Convergenza più veloce inizialmente
  - ❌ Rischio instabilità (loss diverge, NaN)
  - ❌ Oscillazioni attorno al minimo
  
- **Diminuire (es. 1e-4, 5e-5):**
  - ✅ Training più stabile
  - ✅ Convergenza più fine
  - ❌ Convergenza molto lenta
  - ❌ Può rimanere bloccato in minimi locali

**Perché:** Learning rate controlla quanto "aggressivamente" il modello impara. Troppo alto = instabile, troppo basso = lento.

**Regola empirica:**
- GAN standard: 1e-4 (conservativo)
- WGAN-GP: 1e-4 - 5e-4 (più stabile)
- Se loss diverge (NaN): dimezza learning rate

---

### `n_critic: 3`
**Cosa fa:** Numero di step di training del Discriminator per ogni step del Generator.

**Modifica comporta:**
- **Aumentare (es. 5, 10):**
  - ✅ Discriminator più forte (critica migliore)
  - ✅ Training più stabile (WGAN-GP theory)
  - ❌ Training più lento (~40% più lento con n=5)
  
- **Diminuire (es. 1, 2):**
  - ✅ Training più veloce
  - ❌ Discriminator debole (Generator non guidato bene)
  - ❌ Rischio mode collapse

**Perché:** WGAN-GP richiede Discriminator ben trainato per fornire gradienti utili al Generator. n_critic=5 è teoricamente ottimale, ma n_critic=3 è un buon compromesso velocità/qualità.

**Impatto velocità:**
- n_critic=1: 100% velocità (sconsigliato)
- n_critic=3: ~60% velocità (raccomandato)
- n_critic=5: ~40% velocità (ottimale teoria)

---

## 🎯 PARAMETRI GAN/RL MIXING

### `lambda_start: 0.8` (80% GAN, 20% RL)
**Cosa fa:** Peso iniziale della loss GAN vs RL all'inizio del training.

**Modifica comporta:**
- **Aumentare (es. 1.0):**
  - ✅ Inizio con solo GAN (più stabile)
  - ❌ RL entra in gioco tardi
  
- **Diminuire (es. 0.5):**
  - ✅ RL attivo da subito
  - ❌ Potrebbe destabilizzare training iniziale

**Perché:** All'inizio il Generator produce tracce casuali. Meglio usare GAN loss per stabilizzare, poi introdurre RL gradualmente.

---

### `lambda_end: 0.5` (50% GAN, 50% RL)
**Cosa fa:** Peso finale della loss GAN vs RL alla fine del training.

**Modifica comporta:**
- **Aumentare (es. 0.8):**
  - ✅ GAN rimane dominante
  - ❌ Meno ottimizzazione basata su reward
  
- **Diminuire (es. 0.2):**
  - ✅ RL diventa dominante (ottimizza reward)
  - ❌ Potrebbe ignorare realismo (GAN loss)

**Perché:** Bilanciamento finale tra "sembrare reale" (GAN) e "avere buon reward" (RL). 50/50 è equilibrato.

**Formula loss totale:**
```
loss_total = lambda * loss_GAN + (1 - lambda) * loss_RL
```

---

### `lambda_decay_start: 5`
**Cosa fa:** Epoca da cui inizia il decay lineare di lambda (da lambda_start a lambda_end).

**Modifica comporta:**
- **Aumentare (es. 10, 20):**
  - ✅ Più tempo con GAN puro (stabilità)
  - ❌ RL entra tardi (meno ottimizzazione reward)
  
- **Diminuire (es. 0, 2):**
  - ✅ RL attivo prima
  - ❌ Potrebbe destabilizzare training iniziale

**Perché:** Serve tempo per stabilizzare il Generator con GAN prima di introdurre RL. 5 epoch è un buon warm-up.

**Esempio decay:**
```
Epoch 0-4:   lambda = 0.8 (80% GAN)
Epoch 5:     lambda = 0.8 → 0.5 (inizio decay)
Epoch 10:    lambda = 0.73
Epoch 25:    lambda = 0.63
Epoch 50:    lambda = 0.5 (50% GAN)
```

---

## 🎁 PARAMETRI REWARD FUNCTION

### `use_rl: True`
**Cosa fa:** Abilita/disabilita componente Reinforcement Learning.

**Modifica comporta:**
- **False:**
  - ✅ Solo GAN puro (più semplice)
  - ❌ Nessuna ottimizzazione basata su reward
  - ❌ Nessun controllo su validity/fitness/conformance
  
- **True:**
  - ✅ Ottimizzazione multi-obiettivo (reward)
  - ✅ Controllo su qualità tracce
  - ❌ Training più complesso

**Perché:** RL permette di guidare il Generator verso tracce con proprietà desiderate (valide, conformi, diverse).

---

### `reward_weights`
**Cosa fa:** Pesi per le 4 componenti del reward totale.

```python
'reward_weights': {
    'validity': 0.35,      # 35%
    'fitness': 0.35,       # 35%
    'conformance': 0.20,   # 20%
    'diversity': 0.10      # 10%
}
```

**Somma deve essere 1.0 (100%)**

#### `validity: 0.35` (35%)
**Cosa fa:** Peso per tracce sintatticamente corrette (START presente, no self-loop, lunghezza ragionevole).

**Aumentare (es. 0.5):**
- ✅ Più tracce valide generate
- ❌ Meno focus su fitness/conformance

**Perché:** Validity è fondamentale. Senza tracce valide, fitness/conformance sono inutili.

#### `fitness: 0.35` (35%)
**Cosa fa:** Peso per allineamento con reference model (token replay PM4Py).

**Aumentare (es. 0.4):**
- ✅ Tracce più aderenti al modello di riferimento
- ❌ Meno diversità

**Perché:** Fitness misura quanto le tracce "seguono" il processo del reference model.

#### `conformance: 0.20` (20%)
**Cosa fa:** Peso per conformance checking (A* alignment PM4Py).

**Aumentare (es. 0.3):**
- ✅ Tracce più conformi al processo
- ❌ Meno esplorazione di varianti

**Perché:** Conformance è più rigorosa di fitness (penalizza deviazioni).

#### `diversity: 0.10` (10%)
**Cosa fa:** Peso per diversità rispetto al training set (edit distance).

**Aumentare (es. 0.2):**
- ✅ Più tracce nuove/originali
- ❌ Rischio di tracce irrealistiche

**Perché:** Diversity previene mode collapse (generare sempre le stesse tracce).

**Configurazioni comuni:**
```python
# Bilanciato (default)
{'validity': 0.35, 'fitness': 0.35, 'conformance': 0.20, 'diversity': 0.10}

# Focus su validità
{'validity': 0.50, 'fitness': 0.25, 'conformance': 0.15, 'diversity': 0.10}

# Focus su conformance
{'validity': 0.30, 'fitness': 0.30, 'conformance': 0.30, 'diversity': 0.10}

# Focus su diversità (data augmentation)
{'validity': 0.30, 'fitness': 0.25, 'conformance': 0.20, 'diversity': 0.25}
```

---

## 📊 PARAMETRI GENERATION

### `n_samples_eval: 500`
**Cosa fa:** Numero di tracce generate per valutazione ad ogni epoca.

**Modifica comporta:**
- **Aumentare (es. 1000, 2000):**
  - ✅ Statistiche più robuste
  - ❌ Valutazione più lenta
  
- **Diminuire (es. 100, 200):**
  - ✅ Valutazione veloce
  - ❌ Statistiche meno affidabili

**Perché:** Serve campione grande per stimare accuratamente valid_rate, unique_rate, novel_rate.

---

## 💾 PARAMETRI CHECKPOINTING

### `save_dir: 'checkpoints/process_gan'`
**Cosa fa:** Directory dove salvare i checkpoint del modello.

**Modifica comporta:** Cambia solo la posizione dei file salvati.

---

### `save_every: 10`
**Cosa fa:** Frequenza di salvataggio checkpoint (ogni N epoch).

**Nota:** Attualmente il codice salva SOLO il best model (non usa questo parametro).

---

## 🌡️ PARAMETRI GUMBEL-SOFTMAX

### `temperature_start: 5.0`
**Cosa fa:** Temperatura iniziale per Gumbel-Softmax (sampling discreto differenziabile).

**Modifica comporta:**
- **Aumentare (es. 10.0):**
  - ✅ Sampling più "soft" (distribuzioni smooth)
  - ❌ Meno discreto (meno realistico)
  
- **Diminuire (es. 2.0):**
  - ✅ Più discreto da subito
  - ❌ Gradienti più difficili all'inizio

**Perché:** Alta temperatura all'inizio = gradienti più smooth = training più stabile.

---

### `temperature_end: 0.5`
**Cosa fa:** Temperatura finale per Gumbel-Softmax.

**Modifica comporta:**
- **Aumentare (es. 1.0):**
  - ✅ Output meno discreto (più smooth)
  - ❌ Tracce meno "sharp"
  
- **Diminuire (es. 0.1):**
  - ✅ Output molto discreto (quasi argmax)
  - ❌ Gradienti quasi zero (problemi training)

**Perché:** Bassa temperatura finale = output discreto = tracce realistiche.

---

### `temperature_decay: 0.95`
**Cosa fa:** Fattore di decay esponenziale della temperatura.

**Formula:** `temp(epoch) = temp_start * (decay ^ epoch)`

**Modifica comporta:**
- **Aumentare (es. 0.98):**
  - ✅ Decay più lento (più tempo con temperature alte)
  
- **Diminuire (es. 0.90):**
  - ✅ Decay più veloce (discretizzazione rapida)

**Esempio decay con 0.95:**
```
Epoch 0:  temp = 5.0
Epoch 10: temp = 3.0
Epoch 20: temp = 1.8
Epoch 30: temp = 1.1
Epoch 50: temp = 0.5 (min)
```

---

## 📝 PARAMETRI LOGGING

### `log_every: 1`
**Cosa fa:** Frequenza di logging dettagliato (ogni N epoch).

**Modifica comporta:**
- **Aumentare (es. 5, 10):**
  - ✅ Meno output verboso
  - ❌ Meno visibilità su progressi
  
- **Diminuire (es. 1):**
  - ✅ Monitoraggio dettagliato
  - ❌ Output molto verboso

**Perché:** log_every=1 mostra progressi ad ogni epoca (utile per training brevi).

---

## 🎯 CONFIGURAZIONI RACCOMANDATE

### 🚀 Quick Test (5 min)
```python
config = {
    'max_activities': 10,
    'z_dim': 32,
    'decoder_units': (64, 128, 128),
    'discriminator_units': (64, 64),
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 3e-4,
    'n_critic': 3,
}
```

### ⚖️ Bilanciato (20 min)
```python
config = {
    'max_activities': 10,
    'z_dim': 64,
    'decoder_units': (128, 256, 256),
    'discriminator_units': (128, 128),
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 3e-4,
    'n_critic': 3,
}
```

### 🏆 Ottimale (40 min)
```python
config = {
    'max_activities': 15,
    'z_dim': 128,
    'decoder_units': (256, 512, 512),
    'discriminator_units': (256, 128),
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 1e-4,
    'n_critic': 5,
}
```

### 💾 Dataset Piccolo (<1000 tracce)
```python
config = {
    'max_activities': 10,
    'z_dim': 32,
    'decoder_units': (64, 128, 128),
    'discriminator_units': (64, 64),
    'dropout_rate': 0.3,  # Più regolarizzazione
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 1e-4,
}
```

### 🎲 Focus Diversità (Data Augmentation)
```python
config = {
    'reward_weights': {
        'validity': 0.30,
        'fitness': 0.25,
        'conformance': 0.20,
        'diversity': 0.25  # Aumentato
    },
    'lambda_end': 0.3,  # Più RL (ottimizza reward)
}
```

---

## ⚠️ TROUBLESHOOTING

### Problema: Loss diverge (NaN)
**Soluzione:**
- Dimezza `learning_rate` (3e-4 → 1.5e-4)
- Aumenta `n_critic` (3 → 5)
- Riduci `batch_size` (32 → 16)

### Problema: Valid Rate basso (<30%)
**Soluzione:**
- Aumenta `reward_weights['validity']` (0.35 → 0.50)
- Aumenta `lambda_end` (0.5 → 0.7) per più RL
- Aumenta `epochs` (50 → 100)

### Problema: Overfitting
**Soluzione:**
- Aumenta `dropout_rate` (0.1 → 0.3)
- Riduci `decoder_units` (256 → 128)
- Riduci `z_dim` (64 → 32)
- Aumenta `validation_split` (0.1 → 0.2)

### Problema: Training troppo lento
**Soluzione:**
- Riduci `max_activities` (15 → 10)
- Riduci `n_critic` (5 → 3)
- Aumenta `batch_size` (32 → 64)
- Riduci `n_samples_eval` (500 → 200)

### Problema: Out of Memory
**Soluzione:**
- Riduci `batch_size` (32 → 16)
- Riduci `max_activities` (15 → 10)
- Riduci `decoder_units` (256 → 128)

---

## 📚 RIFERIMENTI TEORICI

### Perché questi valori di default?

1. **max_activities=10**: Copre 95° percentile helpdesk dataset
2. **z_dim=64**: Bilanciamento capacità/overfitting per 3804 tracce
3. **learning_rate=3e-4**: Ottimale per WGAN-GP (Gulrajani et al. 2017)
4. **n_critic=3**: Compromesso teoria (5) vs velocità
5. **lambda mixing**: Warm-up GAN poi graduale RL (best practice)
6. **reward_weights**: Validity+Fitness prioritari (70%), poi Conformance+Diversity (30%)
7. **temperature decay**: Annealing esponenziale standard per Gumbel-Softmax

---

**Autore:** ProcessGAN Team  
**Versione:** 1.0  
**Data:** 2025
