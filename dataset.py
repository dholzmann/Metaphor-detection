import torch
from enum import Enum
from models import Task, TaskType
import pandas as pd
import numpy as np
from config import DATA_PATH, LABEL_COLUMN, TEXT_COLUMN, TASK_IDS, IGNORE_VALUE


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

