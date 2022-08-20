# Metaphor-detection

Metaphor detection

Supervisor: Dr. rer. pol. Steffen Eger

## Training
Training and evaluating the models is done in test.py. 
For the data set, see https://github.com/3093453/DL4NLP. In order to train the model, the data has to be downloaded into the data directory configured in the config.py file. The intermediate trained MBERT model has been taken from https://github.com/DatNguyen2084/DLDH-Metaphor-detection.
We use the Huggingface Trainer class to train our MTL-BERT model (https://huggingface.co/docs/transformers/main_classes/trainer).
### Saving results
The utils.py file contains functions for saving results.
### Create and load a new Model and Task
Locally trained and saved model have to be saved in the model directory (config.py) and added to the Model enum class in model.py. The enum value has to be the path to the model directory.
The model.py file contains the MTL-BERT model and the task specific layers (task heads). For creating a new task, a new Task description has to be created in the Task enum class. And the load_task_head() function in the main MTL-BERT model has to include the new task. Additionaly, the function for loading the dataset of the task has to be adjusted to include the new data set for the task.

