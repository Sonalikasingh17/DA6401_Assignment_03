# DA6401_Assignment_03
# Link for the Wandb Report:

https://wandb.ai/sonalikasingh299-iit-madras/DA6401-Assignment_03/reports/DA6401-Assignment-03--VmlldzoxMjcwNzE2OA


# Sequence-to-Sequence Translation Model with Attention

This project implements a sequence-to-sequence (Seq2Seq) model for language translation using TensorFlow/Keras. It supports different RNN cell types (LSTM, GRU, SimpleRNN), Bahdanau attention, and optional beam search for decoding.

##  Notebook Structure

The notebook `Assignment_03.ipynb` is organized as follows:

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

- RNN cell type (`LSTM`, `GRU`, `SimpleRNN`)
- Embedding size
- Units (hidden layer size)
- Use of attention (`True/False`)
- Beam width

Each sweep run logs metrics like training loss, validation accuracy, and BLEU score.

##  Best Model

The best model is selected based on validation score and test performance. The optimal configuration typically includes:

- `LSTM` cell
- Bahdanau attention
- Beam search decoding (beam width = 3 or 5)

Model artifacts and configurations are logged to Weights & Biases for reproducibility.

##  Training the Model

Training is done using a custom loop with `tf.GradientTape`. You can initiate training with a simple function call:

```python
train_model(config)
