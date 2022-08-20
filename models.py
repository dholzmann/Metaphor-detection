from enum import Enum
from config import ROOT_PATH
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import *
from transformers import AutoModel, AutoTokenizer
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=(FutureWarning, UserWarning))
"""
The code has been implemented with the help of following sources:
https://github.com/fornaciari/soft-labels.git
https://towardsdatascience.com/how-to-create-and-train-a-multi-task-transformer-model-18c54a146240
https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb
"""

class KLregular(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(torch.sum(P_targ * torch.log2(P_targ/Q_pred), dim=1))
        # return torch.sum(P_targ * torch.log2(P_targ / Q_pred))


class KLinverse(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(torch.sum(Q_pred * torch.log2(Q_pred/P_targ), dim=1))
        # return torch.sum(Q_pred * torch.log2(Q_pred / P_targ))


class CrossEntropySoft(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        Q_pred = F.softmax(Q_pred, dim=1) # per allinearmi a nn.CrossEntropyLoss, che applica softmax a valori qualsiasi
        return torch.mean(-torch.sum(P_targ * torch.log2(Q_pred), dim=1))
        # return -torch.sum(P_targ * torch.log2(Q_pred))


class CrossEntropy(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(-torch.sum(P_targ * torch.log2(Q_pred), dim=1))
        # return -torch.sum(P_targ * torch.log2(Q_pred))


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return torch.sqrt(self.mse(input.float(), target.float()))


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    ALPHA = 0.8
    GAMMA = 5

    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets.to(torch.float32), reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


class Model(Enum):
    REDE_BERT = "redewiedergabe/bert-base-historical-german-rw-cased",      # redewiedergabe
    INTER_VUA = ROOT_PATH + "/intermediate-task-vua/model",                  # MBERT intermediate_task_vua
    INTER_TROFI = ROOT_PATH + "/intermediate-task-trofi/model",              # MBERT intermediate_task_trofi
    MBERT = "bert-base-multilingual-cased",                                 # MBERT
    MTL_REDE_BERT = ROOT_PATH + "/results/m/REDE_BERT/checkpoint-25472", # ROOT_PATH +
    MTL_INTER_VUA1 = ROOT_PATH + "/results/INTER_VUA/mtl-soft_label/checkpoint-7960", # ROOT_PATH +
    MTL_INTER_VUA2 = ROOT_PATH + "/results/INTER_VUA/mtl-soft_label-emotion_regression/checkpoint-12991" # ROOT_PATH +


class TaskType(Enum):
    SEQ_CLASSIFICATION = "seq_classification"
    EMOTION_REGRESSION = "emotion_regression"
    SOFT_LABEL_REGRESSION = "label_regression"


class Task:
    def __init__(self, id:int, name:str, type:TaskType, num_labels:int, text_column:str, label_column):
        self.id = id
        self.name = name
        self.type = type
        self.num_labels = num_labels
        self.text_column = text_column
        self.label_column = label_column


def load_model_tokenizer(model_type: Model = Model.REDE_BERT):
    model_t = model_type.value[0] if isinstance(model_type.value, Tuple) else model_type.value
    model = AutoModel.from_pretrained(model_t, cache_dir=None, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(model_t)
    return model, tokenizer


class MultiTaskModelBERT(nn.Module):
    def __init__(self, model: Model, tasks: List, task_weights: List = None):
        super().__init__()
        self. task_weights = task_weights if task_weights else [1 for _ in tasks]
        self.encoder, self.tokenizer = load_model_tokenizer(model)

        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = MultiTaskModelBERT.load_task_heads(self.encoder.config.hidden_size, task)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder

    @staticmethod
    def load_task_heads(encoder_hidden_size, task: Task):
        if task.type == TaskType.SEQ_CLASSIFICATION:
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == TaskType.SOFT_LABEL_REGRESSION:
            return SoftLabelHead(encoder_hidden_size, task.num_labels)
        elif task.type == TaskType.EMOTION_REGRESSION:
            return EmotionRegressionHead(encoder_hidden_size, task.num_labels)
        else:
            raise NotImplementedError()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, task_ids=None, **kwargs, ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:

            task_id_filter = (task_ids == unique_task_id)[:,0]
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss*self.task_weights[unique_task_id])

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])

        if loss_list:
            loss = torch.stack(loss_list)
            #outputs = (loss.mean(),) + outputs
            outputs = (loss.sum().float(),) + outputs

        return outputs


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, loss_func=FocalLoss(), dropout_p=0.1):
        super().__init__()
        self.loss_function = loss_func
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = pooled_output.to("cuda:0")
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, :, 0]
            loss = self.loss_function(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


class SoftLabelHead(nn.Module):
    def __init__(self, hidden_size, num_labels, num_layers=1, loss_func=KLinverse(), dropout_p=0.1, device='cuda:0'):
        super().__init__()
        self.num_layers = num_layers
        self.loss_function = loss_func
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = pooled_output.to("cuda:0")
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = F.softmax(logits, dim=1)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, :, 0]
            loss = self.loss_function(
                logits.view(-1, self.num_labels), labels.view(-1)
            )
        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()


class EmotionRegressionHead(nn.Module):
    def __init__(self, hidden_size, num_labels, num_layers=1, loss_func=RMSELoss(), dropout_p=0.1, device='cuda:0'):
        super().__init__()
        self.num_layers = num_layers
        self.loss_function = loss_func
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, 3)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1)).float()
        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
