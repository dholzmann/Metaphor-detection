import deepl
import torch
from enum import Enum
from models import Task, TaskType
import pandas as pd
import datasets
import numpy as np
from numpy import exp
from config import ROOT_PATH, RESULTS_PATH, DATA_PATH, MODEL_PATH, LABEL_COLUMN, TEXT_COLUMN, TASK_IDS, IGNORE_VALUE
from IPython.display import display


class MetaphorLabel(Enum):
    Metaphor = 1
    NO_METAPHOR = 0


def load_emotion_bank(label_padding, task_id, dataset=DATA_PATH + "emobank.csv"):
    task_info = Task(
        id=task_id,
        name="emotion regression",
        num_labels=1,
        type=TaskType.EMOTION_REGRESSION,
        text_column="text",
        label_column=["V", "A", "D"]
    )

    raw_dataset = pd.read_csv(dataset)
    data = prep_dataset(label_padding, task_info, raw_dataset)
    return task_info, data


def load_soft_label_dataset(label_padding, task_id, dataset=DATA_PATH + "total_softlabel.csv"):
    task_info = Task(
        id=task_id,
        name="soft label",
        num_labels=1,
        type=TaskType.SOFT_LABEL_REGRESSION,
        text_column="Textstelle",
        label_column=["SoftLabel"]
    )

    raw_dataset = pd.read_csv(dataset)
    data = prep_dataset(label_padding, task_info, raw_dataset)
    return task_info, data


def load_metaphor_classification_dataset(label_padding, task_id, dataset=DATA_PATH + "total_softlabel.csv"):
    task_info = Task(
        id=task_id,
        name="metaphor detection",
        num_labels=1,
        type=TaskType.SEQ_CLASSIFICATION,
        text_column="Textstelle",
        label_column=["Metapher?"]
    )
    raw_dataset = pd.read_csv(dataset)
    data = prep_dataset(label_padding, task_info, raw_dataset)
    return task_info, data


def prep_dataset(label_padding, task_info, raw_data):
    text = raw_data[task_info.text_column]
    task_ids = [task_info.id] * len(text)
    labels = []
    for idx in raw_data.index:
        label_ids = []
        for l_col in task_info.label_column:
            label_ids.append(raw_data[l_col][idx])
        label_ids = label_ids + [IGNORE_VALUE] * (label_padding - len(label_ids))  # label and add padding with value -100
        labels.append(label_ids)

    padded_dataset = pd.DataFrame({TEXT_COLUMN: text,
                                   LABEL_COLUMN: labels,
                                   TASK_IDS: task_ids})
    return padded_dataset


def load_dataset(task_list=None):
    load_task_specific_dataset = {TaskType.SEQ_CLASSIFICATION: load_metaphor_classification_dataset,
                                  TaskType.SOFT_LABEL_REGRESSION: load_soft_label_dataset,
                                  TaskType.EMOTION_REGRESSION: load_emotion_bank}

    if not task_list:
        task_list = [TaskType.SEQ_CLASSIFICATION, TaskType.SOFT_LABEL_REGRESSION]

    dataset = []
    tasks = []

    label_padding = 1
    if TaskType.EMOTION_REGRESSION in load_task_specific_dataset.keys():
        label_padding = 3

    min_dataset_size = np.inf
    for i, task in enumerate(task_list):
        task_info, data = load_task_specific_dataset[task](label_padding, i)
        # dataset.append(pd.DataFrame(data.data))
        dataset.append(data)
        tasks.append(task_info)
        min_dataset_size = min(min_dataset_size, len(data))
    # Merge train datasets
    # dataset = pd.concat(dataset, axis=1)[:min_dataset_size]
    dataset = pd.concat(dataset)
    # dataset.shuffle(seed=123)
    return tasks, dataset

"""
def tokenize_and_pad_text(tokenizer, max_length, label_padding, task_info, raw_data):
    LABEL_COLUMN = "labels"
    TASK_IDS = "task_id"

    text = raw_data[task_info.text_column]
    tokenized_inputs = tokenizer(text.to_list(), padding="max_length", max_length=max_length, truncation=True)
    labels = []
    for idx in raw_data.index:
        label_ids = []
        for l_col in task_info.label_column:
            label_ids.append(raw_data[l_col][idx])
        label_ids = label_ids + [-100] * (label_padding - len(label_ids))  # label and add padding with value -100
        labels.append(label_ids)
    tokenized_inputs[LABEL_COLUMN] = labels
    tokenized_inputs[TASK_IDS] = [task_info.id] * len(tokenized_inputs["labels"])
    return tokenized_inputs
        
def load_dataset(task_list=None):
    if not task_list:
        task_list = [TaskType.SEQ_CLASSIFICATION]
    load_task_specific_dataset = {TaskType.SEQ_CLASSIFICATION: load_metaphor_classification_dataset,
                                  TaskType.SOFT_LABEL_REGRESSION: load_soft_label_dataset,
                                  TaskType.EMOTION_REGRESSION: load_emotion_bank}
    label_padding = 1
    if TaskType.EMOTION_REGRESSION in load_task_specific_dataset.keys():
        label_padding = 3

    dataset = []
    tasks = []
    min_dataset_size = np.inf
    for i, task in enumerate(task_list):
        task_info, data = load_task_specific_dataset[task](i)
        #dataset.append(pd.DataFrame(data.data))
        dataset.append(pd.DataFrame(data))
        tasks.append(task_info)
        min_dataset_size = min(min_dataset_size, len(data))
    # Merge train datasets
    #dataset = pd.concat(dataset, axis=1)[:min_dataset_size]
    #dataset = pd.concat(dataset)
    dataset = datasets.Dataset.from_pandas(pd.concat(dataset))
    #dataset.shuffle(seed=123)
    return tasks, dataset


def load_emotion_bank(task_id, tokenizer, max_length, dataset=DATA_PATH + "emobank.csv"):
    task_info = Task(
        id=task_id,
        name="emotion regression",
        num_labels=3,
        type=TaskType.EMOTION_REGRESSION,
        text_column="text",
        label_column=["V", "A", "D"]
    )
    raw_dataset = pd.read_csv(dataset)
    dataset = raw_dataset[["text", "V", "A", "D"]]

    return task_info, dataset


def load_soft_label_dataset(task_id, tokenizer, max_length, dataset=DATA_PATH + "total_softlabel.csv"):
    task_info = Task(
        id=task_id,
        name="soft label",
        num_labels=1,
        type=TaskType.SOFT_LABEL_REGRESSION,
        text_column="Textstelle",
        label_column=["SoftLabel"]
    )

    raw_dataset = pd.read_csv(dataset)
    dataset = raw_dataset[["Textstelle", "SoftLabel"]]
    dataset
    return task_info, dataset


def load_metaphor_classification_dataset(task_id, tokenizer, max_length, dataset=DATA_PATH + "total_softlabel.csv"):
    task_info = Task(
        id=task_id,
        name="metaphor detection",
        num_labels=2,
        type=TaskType.SEQ_CLASSIFICATION,
        text_column="Textstelle",
        label_column=["Metapher?"]
    )
    raw_dataset = pd.read_csv(dataset)
    dataset = raw_dataset[["Textstelle", "Metapher?"]]
    return task_info, dataset


def load_dataset(tokenizer, task_list=None, max_length=512):
    if not task_list:
        task_list = [TaskType.SEQ_CLASSIFICATION]
    load_task_specific_dataset = {TaskType.SEQ_CLASSIFICATION: load_metaphor_classification_dataset,
                                  TaskType.SOFT_LABEL_REGRESSION: load_soft_label_dataset,
                                  TaskType.EMOTION_REGRESSION: load_emotion_bank}

    dataset = []
    tasks = []
    min_dataset_size = np.inf
    for i, task in enumerate(task_list):
        task_info, data = load_task_specific_dataset[task](i, tokenizer, max_length)
        data = pd.DataFrame(data)
        data.reset_index(drop=True, inplace=True)
        dataset.append(data)
        tasks.append(task_info)
        min_dataset_size = min(min_dataset_size, len(data))
    # Merge train datasets
    dataset = pd.concat(dataset, axis=1)[:min_dataset_size]
    #dataset = datasets.Dataset.from_pandas(dataset)
    #dataset.shuffle(seed=123)
    return tasks, dataset

def load_seq_classification_dataset(tokenizer, data_args, training_args, dataset=DATA_PATH + "total.csv"):
    raw_dataset = pd.read_csv(dataset)
    data = raw_dataset.assign(SoftLabel=lambda x: exp(x.Metapher) / (exp(x.Metapher) + exp(x.Metaphernkandidat) + exp(x.Nein)))
    data = data[["Textstelle", "Metapher?", "SoftLabel", "sentence_id"]]
    data.loc[data['Metapher?'] == 'Nein', 'Metapher?'] = MetaphorLabel.NO_METAPHOR.value
    data.loc[data['Metapher?'] == 'Metaphernkandidat', 'Metapher?'] = MetaphorLabel.METAPHOR_CANDIDATE.value
    data.loc[data['Metapher?'] == 'Unklar', 'Metapher?'] = MetaphorLabel.METAPHOR_CANDIDATE.value
    data.loc[data['Metapher?'] == 'Grenzfall', 'Metapher?'] = MetaphorLabel.METAPHOR_CANDIDATE.value
    data.loc[data['Metapher?'] == 'Metapher', 'Metapher?'] = MetaphorLabel.METAPHOR.value
    #display(data[["Metapher?", "SoftLabel", "sentence_id"]])
    print("yas")

    
    text_column_name = "tokens"
    label_column_name = "ner_tags"

    label_list = raw_datasets["train"].features[label_column_name].feature.names
    num_labels = len(label_list)

    tokenized_datasets = tokenize_token_classification_dataset(
        raw_datasets,
        tokenizer,
        task_id,
        label_list,
        text_column_name,
        label_column_name,
        data_args,
        training_args,
    )

    task_info = Task(
        id=task_id,
        name=dataset_name,
        num_labels=num_labels,
        type=TaskType.TOKEN_CLASSIFICATION
    )

    return (
        tokenized_datasets["train"],
        tokenized_datasets["validation"],
        task_info,
    )
    """


class MetaphorDataset(torch.utils.data.Dataset):
    """
    The dataset class for metaphors
    """

    def __init__(self, encodings, labels, task_ids=None):
        """
        Initializes the dataset
        """
        self.encodings = encodings
        self.labels = labels
        self.task_ids = task_ids

    def __getitem__(self, idx):
        """
        Returns an individual item by id
        :param idx: The id of the item to return
        :return item: The chosen item
        """
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        if self.task_ids:
            item["task_ids"] = torch.tensor([self.task_ids[idx]])
        return item

    def __len__(self):
        """
        Helper to return the size of the dataset
        :return lenght: the size of the dataset
        """
        return len(self.labels)


if __name__ =="__main__":
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
    import models
    from transformers import AutoTokenizer
    task_list = [TaskType.SEQ_CLASSIFICATION, TaskType.EMOTION_REGRESSION, TaskType.SOFT_LABEL_REGRESSION]
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    tasks, df = ld(task_list)
    df = df.sample(frac=1).reset_index(drop=True)
    display(df)
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    i = 0
    for train_index, test_index in kf.split(df[TEXT_COLUMN], df[LABEL_COLUMN]):
        train_samples = df.iloc[train_index.tolist()]
        test_samples = df.iloc[test_index.tolist()]

        train_texts = train_samples[TEXT_COLUMN]
        test_texts = test_samples[TEXT_COLUMN]
        train_labels = train_samples[LABEL_COLUMN]
        test_labels = test_samples[LABEL_COLUMN]
        train_tasks = train_samples[TASK_IDS]
        test_tasks = test_samples[TASK_IDS]

        train_encodings = tokenizer(train_texts.to_list(), truncation=True, padding=True)
        test_encodings = tokenizer(test_texts.to_list(), truncation=True, padding=True)

        # convert our tokenized data into a torch Dataset
        train_dataset = MetaphorDataset(train_encodings, train_labels.tolist(), train_tasks.tolist())
        test_dataset = MetaphorDataset(test_encodings, test_labels.tolist(), test_tasks.tolist())

