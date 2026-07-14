# Guida all'Uso — Modulo Autoencoder FFNN (Anomaly Detection)

Questo modulo implementa un **Autoencoder Feed-Forward Neural Network (FFNN)** progettato per il rilevamento delle frodi (Anomaly Detection) su carte di credito. 

Il modello impara a ricostruire esclusivamente le transazioni normali (Classe 0). Durante la fase di test, le frodi produrranno un errore di ricostruzione sensibilmente maggiore rispetto alle transazioni lecite, permettendoci di identificarle impostando una soglia ottimale.

---

## 📂 Struttura dei File

La parte del progetto relativa alla FFNN è organizzata come segue:

```text
CreditCard_FraudDetection/
├── configs/
│   └── ffnn_config.yaml          # Configurazione centrale (architettura e iperparametri)
├── src/
│   ├── models/
│   │   ├── __init__.py           # Package initializer
│   │   └── ffnn_autoencoder.py   # Definizione del modello PyTorch (Encoder, Decoder, Autoencoder)
│   ├── trainer.py                # Pipeline di addestramento con Early Stopping e Checkpoint
│   ├── evaluator.py              # Calcolo delle metriche, ricerca soglia ottima e plot
│   └── utils.py                  # Caricamento config, seed per riproducibilità, device manager
├── train_ffnn.py                 # Script CLI per avviare il training
├── predict_ffnn.py               # Script CLI per valutare il modello e fare predizioni
└── README_FFNN.md                # Questa guida
```

---

## ⚙️ Configurazione (`configs/ffnn_config.yaml`)

Il comportamento dell'intero modulo è controllato tramite il file di configurazione YAML. **Non è necessario modificare il codice Python per cambiare l'architettura o gli iperparametri.**

Esempio di configurazione:
```yaml
model:
  input_dim: 30                      # Numero di feature in ingresso
  encoder_layers: [64, 32, 16]       # Dimensioni dei layer nascosti dell'Encoder (decrescenti)
  latent_dim: 8                      # Dimensione del bottleneck (spazio latente)
  decoder_layers: [16, 32, 64]       # Dimensioni dei layer del Decoder (crescenti)
  activation: "relu"                 # Attivazione: relu | leaky_relu | gelu | selu
  dropout: 0.2                       # Dropout rate (0.0 per disabilitarlo)
  batch_norm: true                   # Applica Batch Normalization ad ogni blocco

training:
  epochs: 100                        # Epoche massime di addestramento
  batch_size: 256                    # Dimensione dei mini-batch
  learning_rate: 0.001               
  optimizer: "adam"                  # Ottimizzatore: adam | adamw | sgd
  early_stopping:
    enabled: true
    patience: 10                     # Arresto dopo 10 epoche senza miglioramenti

anomaly:
  threshold_method: "f1_optimal"     # Metodo soglia: f1_optimal | percentile | mean_std
  percentile: 95                     # Usato solo con metodo "percentile"
```

---

## 🚀 Come Utilizzare il Modulo (Guida Rapida)

### 1. Addestramento del Modello
Per avviare l'addestramento dell'Autoencoder sulle transazioni normali:

```bash
python train_ffnn.py --config configs/ffnn_config.yaml
```

**Cosa fa lo script:**
1. Rileva se ci sono dati pre-elaborati in `data/preprocessed/`. Se non ci sono, carica il file `creditcard.csv` applicando uno split standard temporaneo (80% train, 20% test).
2. Filtra il set di training tenendo **solo le transazioni normali** (Classe 0).
3. Esegue il training salvando ad ogni epoca il miglior checkpoint in `checkpoints/ffnn_autoencoder_best.pt`.
4. Salva il grafico dell'andamento delle loss in `logs/ffnn_training_history.png`.

---

### 2. Valutazione e Predizione
Una volta addestrato il modello, puoi valutare le performance sul test set e calcolare la soglia di anomalia ottimale:

```bash
python predict_ffnn.py --config configs/ffnn_config.yaml --checkpoint checkpoints/ffnn_autoencoder_best.pt
```

**Cosa fa lo script:**
1. Carica il modello addestrato e le transazioni del test set (sia normali che frodi).
2. Calcola l'errore di ricostruzione (MSE) per ciascuna transazione.
3. Determina la soglia di errore ideale basandosi sul metodo scelto (es. `f1_optimal` cerca la soglia che massimizza la metrica F1-score).
4. Genera e mostra a terminale il report di classificazione completo:
   *   **Precision** (quanti allarmi del modello sono frodi reali)
   *   **Recall** (quante frodi reali sono state intercettate)
   *   **F1-score** (bilanciamento geometrico delle prime due)
   *   **AUPRC** (Area Under the Precision-Recall Curve)
5. Salva tre grafici diagnostici nella cartella `plots/`:
   *   `ffnn_error_distribution.png`: Distribuzione dell'errore di ricostruzione per normali vs frodi, con linea della soglia.
   *   `ffnn_precision_recall_curve.png`: Curva Precision-Recall con indicazione dell'AUPRC.
   *   `ffnn_confusion_matrix.png`: Matrice di confusione (True Positives, False Positives, ecc.).

---

## 🤝 Integrazione con il Lavoro dei Collaboratori

Il modulo è pronto per integrarsi con gli altri componenti del progetto in modo trasparente:

### A. Integrazione con il modulo di Preprocessing
Il caricamento dei dati in `train_ffnn.py` e `predict_ffnn.py` cerca in via prioritaria i file tensore PyTorch pronti. Il collaboratore incaricato del preprocessing dovrà semplicemente salvare i set pronti come file `.pt` in `data/preprocessed/`:
*   `data/preprocessed/X_train.pt` e `data/preprocessed/y_train.pt`
*   `data/preprocessed/X_val.pt` e `data/preprocessed/y_val.pt`
*   `data/preprocessed/X_test.pt` e `data/preprocessed/y_test.pt`

Il tuo modulo caricherà questi file automaticamente ed escluderà le frodi in addestramento in totale autonomia.

### B. Confronto con il modulo Self-Attention
L'architettura FFNN è totalmente isolata. Chi svilupperà il modello con Self-Attention scriverà un modello parallelo in `src/models/`. Avendo a disposizione lo stesso file di configurazione e lo stesso split di dati preprocessati, potrete confrontare le curve di loss e i punteggi AUPRC finali dei due modelli in modo pulito e comparativo.
