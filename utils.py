import csv
from pathlib import Path

def df_statistics(df):
    """
    Generates a statistic of a given dataframe
    :param df: The data for which a statistic should be generated
    :return statistic: The statistic of the given dataframe
    """
    return "non-metaphors: {} ({:.0%}), metaphor candidates: {} ({:.0%}), metaphors: {} ({:.0%})".format(df['Metapher?'].value_counts()[0],df['Metapher?'].value_counts()[0]/df['Metapher?'].value_counts().sum(),df['Metapher?'].value_counts()[1],df['Metapher?'].value_counts()[1]/df['Metapher?'].value_counts().sum(),df['Metapher?'].value_counts()[2],df['Metapher?'].value_counts()[2]/df['Metapher?'].value_counts().sum())


def save_results(path, results):
  """
  Saves the results of a training to a given path
  :param path: The path to save to
  :param results: The results to save
  :param annotator: The annotator of this particular result
  """
  print("Saving results to", path)
  Path(path).mkdir(parents=True, exist_ok=True)
  with open(path + '/results.csv', 'w+') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in results.items():
       writer.writerow([key, value])


def save_settings(path, epochs, folds, model_type, tasks):
    """
    Saves the given settings to a text file at a given path
    :param path: the path to output the text file to
    :param epochs: The number of epochs to save
    :param folds: The number of folds to save
    :param model_type: The model type to save
    """

    lines = []
    lines.append('Epochs: ' + str(epochs))
    lines.append('Folds: ' + str(folds))
    lines.append('Model Type: ' + model_type)
    lines.append('Tasks: ' + str(tasks))
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(path + '/settings.txt', 'w+') as f:
        f.write('\n'.join(lines))

    print("Settings saved to", path)