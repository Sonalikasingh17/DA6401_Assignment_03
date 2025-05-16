# DA6401_Assignment_03
# Link for the Wandb Report:

https://wandb.ai/sonalikasingh299-iit-madras/DA6401-Assignment_03/reports/DA6401-Assignment-03--VmlldzoxMjcwNzE2OA


# Sequence-to-Sequence Translation Model with Attention

This project implements a sequence-to-sequence (Seq2Seq) model for language translation using TensorFlow/Keras. It supports different RNN cell types (LSTM, GRU, SimpleRNN), Bahdanau attention, and optional beam search for decoding.

## Project Structure

### `dataprocessing.py`

This module handles all data preprocessing tasks:
- Tokenization of input and target sequences
- Padding and truncating sentences
- Vocabulary building
- Train-validation split
- Sentence encoding and decoding functions

### `modelclass.py`

This contains all model-related class definitions:
- `Encoder`: Encodes the input sequence using an RNN cell.
- `Decoder`: Decodes the encoded input sequence and generates the output.
- `BahdanauAttention`: Implements attention mechanism to dynamically focus on relevant input tokens.
- Utility functions to build and initialize model components.

### `train.py`

Handles the training process:
- Model initialization
- Training loop using `tf.GradientTape`
- Masked loss and accuracy
- WandB integration for logging
- Evaluation functions (greedy decoding and beam search)
- BLEU score computation
- Test inference with visualization (if attention is used)

---

Since, I worked with only one python Google Colab notebook for the whole assignment.
I have uploaded the notebook `Assignment_03.ipynb`  which is organized as follows:

1. **Imports & Configurations** – Libraries, environment setup, and configurations for GPU/TPU usage.
2. **Data Preprocessing** – Tokenization, padding, vocabulary construction, and train-validation split.
3. **Model Components**:
   - `Encoder` and `Decoder` classes with attention support.
   - `BahdanauAttention` module.
4. **Training Utilities**:
   - Custom training loop.
   - Loss function, accuracy, and masking.
5. **Evaluation Functions**:
   - Greedy decoding.
   - Beam search decoding.
   - BLEU score computation.
6. **W&B Integration** – Hyperparameter sweeps using Weights & Biases.
7. **Experiments** – Training runs with and without attention.
8. **Testing and Evaluation** – Inference and BLEU score on test samples.

##  Attention Mechanism

The notebook uses **Bahdanau Attention**, which is implemented in a custom class. The attention mechanism computes context vectors from encoder outputs and helps the decoder focus on relevant parts of the input sequence.

You can toggle between using **attention** or **no attention** with a configuration flag (`use_attention=True/False`).

##  Evaluation Metrics

- **BLEU Score** – Calculated for sentence-level translations.
- **Accuracy** – Used during training for masked token prediction.

##  Hyperparameter Tuning (W&B Sweep)

Weights & Biases (`wandb`) is used to perform hyperparameter sweeps over:

``` yaml
# sweep without attention
sweep_config = {
  "name": "Sweep 1- Assignment3",
   "method": "bayes",
  "metric": {
        'name': 'validation_accuracy',  
        'goal': 'maximize'              
    },
 
  "parameters": {
        "enc_dec_layers": {
           "values": [1, 2, 3]
        },
        "units": {
            "values": [64, 128, 256]
        },
        "layer_type": {
            "values": ["rnn", "gru", "lstm"]
        },
        "embedding_dim": {
            "values": [64, 128, 256]
        },
        "dropout": {
            "values": [0.2, 0.3]
         },
        "beam_width": {
            "values": [3, 5, 7]
        },
            "teacher_forcing_ratio": {
            "values": [0.3, 0.5, 0.7, 0.9]
        }   
    }
}


# Sweep with attention mechanism 
sweep_config2 = {
  "name": "Attention Sweep - Assignment3",
  "description": "Hyperparameter sweep for Seq2Seq Model with Attention",
  "method": "bayes",
  "metric": {
        "name": "val acc",
        "goal": "maximize"
    },

  "early_terminate": {
        "type": "hyperband",
        "min_iter": 3
    },

  "parameters": {
        "enc_dec_layers": {
           "values": [1, 2, 3]
        },
        "units": {
            "values": [128, 256]
        },
        "dropout": {
            "values": [0, 0.2]
        },
        "attention": {
            "values": [True]
        }
    }
}
```

Each sweep run logs metrics like training loss, validation accuracy, and BLEU score.

##  Best Model

The best model is selected based on validation score and test performance. The optimal configuration typically includes:
```
model = test_on_dataset(language="hi",
                        embedding_dim=256,
                        encoder_layers=1,
                        decoder_layers=1,
                        layer_type="lstm",
                        units=256,
                        dropout=0.3,
                        attention=False)
  ```
Model artifacts and configurations are logged to Weights & Biases for reproducibility.

##  Training the Model

Training is done using a custom loop with `tf.GradientTape`. initiate training with a simple function call:

```bash
train_model(config)

```
## Testing & Evaluation
After training, the model can be tested using beam or greedy decoding:
``` python
evaluate(input_sentence)            # Greedy decoding
beam_evaluate(input_sentence, k=3)  # Beam search with width 3
```
Both functions return the predicted translation and BLEU score.
If attention is enabled, evaluate also visualizes the attention heatmap over source tokens.
