import os
import random
import time
import wandb
import re, string
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from colour import Color
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer


def get_layer(name, units, dropout, return_state=False, return_sequences=False):
    name = name.lower()
    if name == "rnn":
        return layers.SimpleRNN(units=units, dropout=dropout,
                                return_state=return_state,
                                return_sequences=return_sequences)
    if name == "gru":
        return layers.GRU(units=units, dropout=dropout,
                          return_state=return_state,
                          return_sequences=return_sequences)
    if name == "lstm":
        return layers.LSTM(units=units, dropout=dropout,
                           return_state=return_state,
                           return_sequences=return_sequences)
    raise ValueError(f"Unknown layer type: {name}")


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, enc_state, enc_out):

    enc_state = tf.concat(enc_state, 1)
    enc_state = tf.expand_dims(enc_state, 1)

    score = self.V(tf.nn.tanh(self.W1(enc_state) + self.W2(enc_out)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * enc_out
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class Encoder(tf.keras.Model):
    def __init__(self, layer_type, n_layers, units, vocab_size, embedding_dim, dropout):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.units = units
        self.layer_type = layer_type.lower()
        self.dropout = dropout
        self.embedding = layers.Embedding(vocab_size, embedding_dim)

        # Build stacked RNN layers
        self.rnn_layers = []
        if n_layers == 1:
            # Single layer: return full sequence and final state
            self.rnn_layers.append(get_layer(self.layer_type, units, dropout,
                                             return_sequences=True, return_state=True))
        else:
            # Intermediate layers: return sequences, no state
            for _ in range(n_layers - 1):
                self.rnn_layers.append(get_layer(self.layer_type, units, dropout,
                                                 return_sequences=True, return_state=False))
            # Final layer: return sequences and state
            self.rnn_layers.append(get_layer(self.layer_type, units, dropout,
                                             return_sequences=True, return_state=True))

    def call(self, x, hidden):
        """
        x: input tokens (batch, timesteps)
        hidden: initial hidden state(s).
                For LSTM: [h, c], for GRU/RNN: [h].
        Returns: output sequence and final state list.
        """
        # Embed inputs
        x = self.embedding(x)  # (batch, timesteps, embed_dim)
        output = x

        # Prepare initial state for first layer
        if self.layer_type == "lstm":
            initial_state = hidden  # [h, c]
        else:
            # GRU/RNN: hidden is [h]
            initial_state = hidden[0] if isinstance(hidden, list) else hidden

        # Single-layer case
        if self.n_layers == 1:
            if self.layer_type == "lstm":
                output, state_h, state_c = self.rnn_layers[0](output, initial_state=initial_state)
                return output, [state_h, state_c]
            else:
                output, state_h = self.rnn_layers[0](output, initial_state=initial_state)
                return output, [state_h]

        # Multi-layer case
        # First layer with initial state
        output = self.rnn_layers[0](output, initial_state=initial_state)
        # Intermediate layers (no state returned)
        for layer in self.rnn_layers[1:-1]:
            output = layer(output)
        # Final layer returns sequence + state(s)
        if self.layer_type == "lstm":
            output, state_h, state_c = self.rnn_layers[-1](output)
            return output, [state_h, state_c]
        else:
            output, state_h = self.rnn_layers[-1](output)
            return output, [state_h]

    def initialize_hidden_state(self, batch_size):
        """Returns initial zero state."""
        if self.layer_type == "lstm":
            return [tf.zeros((batch_size, self.units)), tf.zeros((batch_size, self.units))]
        else:
            return [tf.zeros((batch_size, self.units))]


class Decoder(tf.keras.Model):
    def __init__(self, layer_type, n_layers, units, vocab_size, embedding_dim, dropout, attention=False):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.units = units
        self.layer_type = layer_type.lower()
        self.dropout = dropout
        self.attention = attention

        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        if self.attention:
            self.attention_layer = BahdanauAttention(units)

        # Build stacked RNN layers for decoder
        self.rnn_layers = []
        if n_layers == 1:
            # Single layer: no time dimension output (one step), return state
            self.rnn_layers.append(get_layer(self.layer_type, units, dropout,
                                             return_sequences=False, return_state=True))
        else:
            # Intermediate layers: return sequences + state
            for _ in range(n_layers - 1):
                self.rnn_layers.append(get_layer(self.layer_type, units, dropout,
                                                 return_sequences=True, return_state=True))
            # Final layer: no time dimension (one step), return state
            self.rnn_layers.append(get_layer(self.layer_type, units, dropout,
                                             return_sequences=False, return_state=True))

        self.dense = layers.Dense(vocab_size, activation='softmax')

    def call(self, x, hidden, enc_out=None):
        """
        x: decoder input tokens (batch, 1)
        hidden: initial hidden state(s) from encoder ([h, c] or [h])
        enc_out: encoder outputs for attention (if any)
        Returns: (predictions, new_state_list, attention_weights)
        """
        x = self.embedding(x)  # (batch, 1, embed_dim)

        # Apply attention if available
        if self.attention and enc_out is not None:
            context_vector, attention_weights = self.attention_layer(hidden, enc_out)
            context_vector = tf.expand_dims(context_vector, 1)  # (batch, 1, units)
            x = tf.concat([context_vector, x], axis=-1)
        else:
            attention_weights = None

        output = x

        # Prepare initial state for first layer
        if self.layer_type == "lstm":
            initial_state = hidden  # [h, c]
        else:
            initial_state = hidden[0] if isinstance(hidden, list) else hidden

        # First layer with initial state
        if self.layer_type == "lstm":
            output, state_h, state_c = self.rnn_layers[0](output, initial_state=initial_state)
        else:
            output, state_h = self.rnn_layers[0](output, initial_state=initial_state)

        # Pass through remaining layers
        for layer in self.rnn_layers[1:]:
            if self.layer_type == "lstm":
                output, state_h, state_c = layer(output)
            else:
                output, state_h = layer(output)

        # Final output (no time dimension)
        output = self.dense(output)  # (batch, vocab_size)

        # Return state list appropriately
        if self.layer_type == "lstm":
            return output, [state_h, state_c], attention_weights
        else:
            return output, [state_h], attention_weights
class Seq2SeqModel():
    def __init__(self, embedding_dim, encoder_layers, decoder_layers, layer_type, units, dropout, attention=False):
        self.embedding_dim = embedding_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.layer_type = layer_type
        self.units = units
        self.dropout = dropout
        self.attention = attention
        self.stats = []
        self.batch_size = 128
        self.use_beam_search = False

    def build(self, loss, optimizer, metric):
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric

    def set_vocabulary(self, input_tokenizer, targ_tokenizer):
        self.input_tokenizer = input_tokenizer
        self.targ_tokenizer = targ_tokenizer
        self.create_model()

    def create_model(self):

        encoder_vocab_size = len(self.input_tokenizer.word_index) + 1
        decoder_vocab_size = len(self.targ_tokenizer.word_index) + 1

        self.encoder = Encoder(self.layer_type, self.encoder_layers, self.units, encoder_vocab_size,
                               self.embedding_dim, self.dropout)

        self.decoder = Decoder(self.layer_type, self.decoder_layers, self.units, decoder_vocab_size,
                               self.embedding_dim,  self.dropout, self.attention)

    @tf.function
 # Within Seq2SeqModel class, updated train_step (no @tf.function decorator)
    # -- Training Loop Fixes in the Seq2SeqModel.fit() method --
    def train_step(self, input_seq, target_seq, enc_states):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_states = self.encoder(input_seq, enc_states)
            dec_input = tf.expand_dims([self.targ_tokenizer.word_index['\t']] * self.batch_size, 1)
            dec_states = enc_states

            for t in range(1, target_seq.shape[1]):
                    # preds, dec_states = self.decoder(dec_input, dec_states, enc_output)
                    preds, dec_states, _ = self.decoder(dec_input, dec_states, enc_output)

                    loss += self.loss(target_seq[:, t], preds)
                    self.metric.update_state(target_seq[:, t], preds)
                    dec_input = tf.expand_dims(target_seq[:, t], 1)

            batch_loss = loss / int(target_seq.shape[1])
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return batch_loss, self.metric.result()

    @tf.function
    def validation_step(self, input, target, enc_state):

        loss = 0

        enc_out, enc_state = self.encoder(input, enc_state)

        dec_state = enc_state
        dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*self.batch_size ,1)

        for t in range(1, target.shape[1]):

            preds, dec_state, _ = self.decoder(dec_input, dec_state, enc_out)
            loss += self.loss(target[:,t], preds)
            self.metric.update_state(target[:,t], preds)

            preds = tf.argmax(preds, 1)
            dec_input = tf.expand_dims(preds, 1)

        batch_loss = loss / target.shape[1]

        return batch_loss, self.metric.result()



    def fit(self, dataset, val_dataset, batch_size=128, epochs=5, use_wandb=False, teacher_forcing_ratio=1.0):
          self.batch_size = batch_size
          self.teacher_forcing_ratio = teacher_forcing_ratio

          steps_per_epoch = len(dataset) // self.batch_size
          steps_per_epoch_val = len(val_dataset) // self.batch_size

          dataset = dataset.batch(self.batch_size, drop_remainder=True)
          val_dataset = val_dataset.batch(self.batch_size, drop_remainder=True)

          # Capture max sequence lengths
          sample_inp, sample_targ = next(iter(dataset))
          self.max_target_len = sample_targ.shape[1]
          self.max_input_len = sample_inp.shape[1]

          for epoch in range(1, epochs+1):
              print(f"EPOCH {epoch}\n")
              total_loss = 0
              total_acc = 0
              self.metric.reset_state()

              starting_time = time.time()

              print("Training...\n")
              for batch, (input_batch, target_batch) in enumerate(dataset.take(steps_per_epoch)):
                  # Re-initialize encoder hidden state for each batch
                  enc_state = self.encoder.initialize_hidden_state(self.batch_size)
                  batch_loss, acc = self.train_step(input_batch, target_batch, enc_state)
                  total_loss += batch_loss
                  total_acc += acc
                  if batch == 0 or ((batch + 1) % 100 == 0):
                      print(f"Batch {batch+1} Loss {batch_loss:.4f}")

              avg_loss = total_loss / steps_per_epoch
              avg_acc = total_acc / steps_per_epoch

              print("\nValidating...")
              total_val_loss = 0
              total_val_acc = 0
              self.metric.reset_state()

              for batch, (input_batch, target_batch) in enumerate(val_dataset.take(steps_per_epoch_val)):
                  enc_state = self.encoder.initialize_hidden_state(self.batch_size)
                  batch_val_loss, val_acc = self.validation_step(input_batch, target_batch, enc_state)
                  total_val_loss += batch_val_loss
                  total_val_acc += val_acc

              avg_val_loss = total_val_loss / steps_per_epoch_val
              avg_val_acc = total_val_acc / steps_per_epoch_val

              print(f"\nTrain Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}%")
              print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%\n")

                  #         print(template.format(avg_loss, avg_acc*100, avg_val_loss, avg_val_acc*100))

              time_taken = time.time() - starting_time
              self.stats.append({"epoch": epoch,
                              "train loss": avg_loss,
                              "val loss": avg_val_loss,
                              "train acc": avg_acc*100,
                              "val acc": avg_val_acc*100,
                              "training time": time_taken})

              if use_wandb:
                  wandb.log(self.stats[-1])

              print(f"\nTime taken for the epoch {time_taken:.4f}")
              print("-"*100)

          print("\nModel trained successfully !!")

    # Similarly, in test/evaluation loops, re-initialize for each batch:

    def evaluate(self, test_dataset, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        steps_per_epoch_test = len(test_dataset) // self.batch_size
        test_dataset = test_dataset.batch(self.batch_size, drop_remainder=True)

        total_test_loss = 0
        total_test_acc = 0
        self.metric.reset_state()

        for batch, (input_batch, target_batch) in enumerate(test_dataset.take(steps_per_epoch_test)):
            enc_state = self.encoder.initialize_hidden_state(self.batch_size)
            batch_loss, acc = self.validation_step(input_batch, target_batch, enc_state)
            total_test_loss += batch_loss
            total_test_acc += acc

        avg_test_loss = total_test_loss / steps_per_epoch_test
        avg_test_acc = total_test_acc / steps_per_epoch_test
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%")
        return avg_test_loss, avg_test_acc



    def translate(self, word, get_heatmap=False):

        word = "\t" + word + "\n"

        inputs = self.input_tokenizer.texts_to_sequences([word])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self.max_input_len,
                                                               padding="post")

        result = ""
        att_wts = []

        enc_state = self.encoder.initialize_hidden_state(1)
        enc_out, enc_state = self.encoder(inputs, enc_state)

        dec_state = enc_state
        dec_input = tf.expand_dims([self.targ_tokenizer.word_index["\t"]]*1, 1)

        for t in range(1, self.max_target_len):

            preds, dec_state, attention_weights = self.decoder(dec_input, dec_state, enc_out)

            if get_heatmap:
                att_wts.append(attention_weights)

            preds = tf.argmax(preds, 1)
            next_char = self.targ_tokenizer.index_word[preds.numpy().item()]
            result += next_char

            dec_input = tf.expand_dims(preds, 1)

            if next_char == "\n":
                return result[:-1], att_wts[:-1]

        return result[:-1], att_wts[:-1]

    def plot_attention_heatmap(self, word, ax, font_path="/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf"):

        translated_word, attn_wts = self.translate(word, get_heatmap=True)
        attn_heatmap = tf.squeeze(tf.concat(attn_wts, 0), -1).numpy()

        input_word_len = len(word)
        output_word_len = len(translated_word)

        ax.imshow(attn_heatmap[:, :input_word_len])

        font_prop = FontProperties(fname=font_path, size=18)

        ax.set_xticks(np.arange(input_word_len))
        ax.set_yticks(np.arange(output_word_len))

        ax.set_xticklabels(list(word))
        ax.set_yticklabels(list(translated_word), fontproperties=font_prop)
