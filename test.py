# coding=latin-1
import datetime
import re
import os
import csv
import ast
import math
import copy
import glob
import json
import deepl
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report, precision_score, recall_score

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from transformers.models.bert.modeling_bert import *
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils.dummy_tf_objects import TFDPRQuestionEncoder
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, \
    BertModel, Trainer, TrainingArguments, DataCollatorWithPadding

#import models
from models import Task, TaskType, Model, load_model_tokenizer, MultiTaskModelBERT
from dataset import MetaphorDataset, load_dataset
import utils
import dataset
from config import ROOT_PATH, RESULTS_PATH, DATA_PATH, MODEL_PATH, LABEL_COLUMN, TEXT_COLUMN, TASK_IDS
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=(FutureWarning, UserWarning))

from IPython.display import display

def compute_metrics(pred):
    """
    Computes accuracy, macro_f1 score and individual macro f1 per class for a given prediction
    :param pred: The prediction
    :return dict: A dictionary containing accuracy, macro_f1 score and individual macro f1 per class
    """
    labels = pred.label_ids
    preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = preds.argmax(-1)
    if len(preds.shape) <= 2:
        labels = labels[:,:,0]
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    per_class_macro_f1 = f1_score(labels, preds, average=None).tolist()
    precision = precision_score(labels, preds, average=None).tolist()
    recall = recall_score(labels, preds, average=None).tolist()
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1,
        'per_class_macro_f1': per_class_macro_f1,
    }


def train_and_evaluate(model_type: Model.REDE_BERT, epochs=3, batch_size=1, train=True, task_list=None, task_weights=None):
    """
    Training a given model with k-fold cross-validation and various oversampling strategies
    The resulting metrics are saved as CSV to a given path
    :param model_type: The type of model to use
    :param epochs: The amount of epochs to train
    :param folds: The amount of folds to use for k-fold cross validation
    :param path: The path to save the results to
    :return evaluation_results: The results of the evaluation of the trained model
    """
    max_length = 512
    if not task_list:
        task_list = [TaskType.SEQ_CLASSIFICATION, TaskType.EMOTION_REGRESSION, TaskType.SOFT_LABEL_REGRESSION]
    if not task_weights:
        task_weights = [1 for _ in task_list]

    # load dataset + tasks
    tasks, df = load_dataset(task_list)
    # load model
    mtl_bert = MultiTaskModelBERT(model_type, tasks, task_weights=task_weights).to("cuda")
    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)
    # split dataset
    train_samples, test_samples = train_test_split(df, test_size=0.2)

    train_texts = train_samples[TEXT_COLUMN]
    test_texts = test_samples[TEXT_COLUMN]
    train_labels = train_samples[LABEL_COLUMN]
    test_labels = test_samples[LABEL_COLUMN]
    train_tasks = train_samples[TASK_IDS]
    test_tasks = test_samples[TASK_IDS]

    train_encodings = mtl_bert.tokenizer(train_texts.to_list(), truncation=True, padding=True, max_length=max_length)
    test_encodings = mtl_bert.tokenizer(test_texts.to_list(), truncation=True, padding=True, max_length=max_length)

    # convert our tokenized data into a torch Dataset
    train_dataset = MetaphorDataset(train_encodings, train_labels.tolist(), train_tasks.tolist())
    test_dataset = MetaphorDataset(test_encodings, test_labels.tolist(), test_tasks.tolist())

    training_args = TrainingArguments(
        output_dir='./results4/'+model.__str__()+"/"+str([t.value[0] if isinstance(t.value, Tuple) else t.value for t in task_list]),  # output directory
        num_train_epochs=epochs,  # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        #load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        save_total_limit=10,
        save_strategy="epoch",
        logging_strategy="epoch",
        evaluation_strategy="no",  # evaluate each `logging_steps`
        #metric_for_best_model="macro_f1",
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=mtl_bert,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
    )
    if train:
        trainer.train()
        trainer.save_model(output_dir='./results4/'+model.__str__()+"/"+str([t.value[0] if isinstance(t.value, Tuple) else t.value for t in task_list]))

    evaluation = eval(test_labels, test_texts, test_tasks, mtl_bert, trainer, task_list)
    ev = {model_type: evaluation}
    utils.save_results('./results4/'+model.__str__()+"/"+str([t.value[0] if isinstance(t.value, Tuple) else t.value for t in task_list]), ev)
    return evaluation


def eval(test_labels, test_text, test_tasks, mtl_bert, trainer, tasks):
    test_tasks.tolist()
    metrics = []
    for task_id in test_tasks.unique():
        if tasks[task_id] != TaskType.SEQ_CLASSIFICATION:
            continue
        test_l = np.array(test_labels.tolist())[np.array(test_tasks.tolist()) == task_id]
        test_txt = np.array(test_text.tolist())[np.array(test_tasks.tolist())==task_id]
        test_t = [t for t in test_tasks.tolist() if t == task_id]
        test_encodings = mtl_bert.tokenizer(test_txt.tolist(), truncation=True, padding=True, max_length=512)
        test_dataset = MetaphorDataset(test_encodings, test_l.tolist(), test_t)
        metrics.append(trainer.evaluate(eval_dataset=test_dataset))
    return metrics


if __name__ == "__main__":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    #########################
    EPOCHS = 1
    #########################
    transformers.logging.set_verbosity_error()
    current_results = {}

    models = [Model.REDE_BERT] #, Model.INTER_VUA]
    tasks = [TaskType.SEQ_CLASSIFICATION] #, TaskType.SOFT_LABEL_REGRESSION]
    task_weights = None#[1, 0.1]
    for model in models:
        print(model.__str__(), tasks)
        if model == Model.REDE_BERT:
            result = train_and_evaluate(model_type=model, task_list=tasks, task_weights=task_weights, epochs=EPOCHS, batch_size=3, train=False)
        else:
            result = train_and_evaluate(model_type=model, task_list=tasks, task_weights=task_weights, epochs=EPOCHS)
        current_results[model] = result
    """
    models = [Model.INTER_VUA]
    tasks = [TaskType.SEQ_CLASSIFICATION, TaskType.SOFT_LABEL_REGRESSION, TaskType.EMOTION_REGRESSION]
    task_weights = [1, 0.1, 0.1]
    for model in models:
        print(model.__str__(), tasks)
        result = train_and_evaluate(model_type=model, task_list=tasks, task_weights=task_weights, epochs=EPOCHS)
        current_results[model] = result
    """
    utils.save_results("results3", current_results)

