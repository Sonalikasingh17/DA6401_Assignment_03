import os
import random
import time
import wandb
import re, string
import numpy as np
import pandas as pd
from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K



def train_with_wandb(language, test_beam_search=False):

    config_defaults = {"embedding_dim": 64,
                       "enc_dec_layers": 1,
                       "layer_type": "lstm",
                       "units": 128,
                       "dropout": 0,
                       "attention": False,
                       "beam_width": 3,
                       "teacher_forcing_ratio": 1.0
                       }

    wandb.init(config=config_defaults, project="DA6401-Assignment_03", resume=True)
    
    wandb.run.name=f"edim_{wandb.config.embedding_dim}_edl_{wandb.config.enc_dec_layers}_lr_{wandb.config.layer_type}_units_{wandb.config.units}_dp_{wandb.config.dropout}_att_{wandb.config.attention}_bw_{wandb.config.beam_width}_tfc_{wandb.config.teacher_forcing_ratio}"
   
    #Wand run name for hyperparameter sweep of attention mechanism
    ## wandb.run.name = f"layers{num_layers}_units{hiddensize}_drop{dropout}_att{attention}"


    ## 1. SELECT LANGUAGE ##
    TRAIN_TSV, VAL_TSV, TEST_TSV = get_data_files(language)

    ## 2. DATA PREPROCESSING ##
    dataset, input_tokenizer, targ_tokenizer = preprocess_data(TRAIN_TSV)
    val_dataset, _, _ = preprocess_data(VAL_TSV, input_tokenizer, targ_tokenizer)

    ## 3. CREATING THE MODEL ##
    model = Seq2SeqModel(embedding_dim=wandb.config.embedding_dim,
                         encoder_layers=wandb.config.enc_dec_layers,
                         decoder_layers=wandb.config.enc_dec_layers,
                         layer_type=wandb.config.layer_type,
                         units=wandb.config.units,
                         dropout=wandb.config.dropout,
                         attention=wandb.config.attention)

    ## 4. COMPILING THE MODEL
    model.set_vocabulary(input_tokenizer, targ_tokenizer)
    model.build(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(),
                metric = tf.keras.metrics.SparseCategoricalAccuracy())

    ## 5. FITTING AND VALIDATING THE MODEL
    model.fit(dataset, val_dataset, epochs=5, use_wandb=True, teacher_forcing_ratio=wandb.config.teacher_forcing_ratio)

    if test_beam_search:
        ## OPTIONAL :- Evaluate the dataset using beam search and without beam search
        val_dataset, _, _ = preprocess_data(VAL_TSV, model.input_tokenizer, model.targ_tokenizer)
        subset = val_dataset.take(500)

        # a) Without beam search
        _, test_acc_without = model.evaluate(subset, batch_size=100)
        wandb.log({"test acc": test_acc_without})

        # b) With beam search
        beam_search = BeamSearch(model=model, k=wandb.config.beam_width)
        beam_search.evaluate(subset, batch_size=100, use_wandb=True)



#Sweep configuration of Wandb without attention 
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

# sweep_id = wandb.sweep(sweep_config, project="DA6401-Assignment_03")
wandb.agent(sweep_id,function=lambda: train_with_wandb("hi"),project="DA6401-Assignment_03" )


# SWeep with attention
''' 
sweep_config2 = {
  "name": "Attention Sweep - Assignment3",
  "description": "Hyperparameter sweep for Seq2Seq Model with Attention",
  "method": "grid",
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
sweep_id2 = wandb.sweep(sweep_config2, project="DA6401-Assignment_03")
wandb.agent(sweep_id2,function=lambda: train_with_wandb("hi"),project="DA6401-Assignment_03" , count = 5)
'''
# Change run name for this attention sweep 
#wandb.run.name = f"layers{num_layers}_units{hiddensize}_drop{dropout}_att{attention}"
