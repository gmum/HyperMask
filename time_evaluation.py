import os
import glob
import time
import tzlocal
import pandas as pd
from pathlib import Path
from datetime import datetime


model_name = 'TinyImageNet_ZenkeNet'
# 'CIFAR_100_ResNet', 'CIFAR_100_ZenkeNet', 'Split_MNIST', 'Permuted_MNIST'
# 'TinyImageNet_ResNet', 'TinyImageNet_ZenkeNet'
if model_name == 'CIFAR_100_ResNet':
    main_path = '/media/kksiazek/Nowy/continual_learning/CIFAR-100/ICLR_models_ResNet/'
    last_task = 9
elif model_name == 'CIFAR_100_ZenkeNet':
    main_path = '/home/kksiazek/Desktop/Results/ZenkeNet/'
    last_task = 9
elif model_name == 'Split_MNIST':
    main_path = '/media/kksiazek/Nowy/continual_learning/SplitMNIST/best_models/ICLR_models/'
    last_task = 4
elif model_name == 'Permuted_MNIST':
    main_path = '/home/kksiazek/Desktop/Results/Permuted_MNIST/'
    last_task = 9
elif model_name == 'TinyImageNet_ZenkeNet':
    main_path = '/home/kksiazek/Desktop/Results/TinyImageNet/ZenkeNet/'
    last_task = 39
elif model_name == 'TinyImageNet_ResNet':
    main_path = '/media/kksiazek/Nowy/continual_learning/TinyImageNet/'
    last_task = 39


local_timezone = tzlocal.get_localzone()
os.chdir(main_path)
results = []
# number of model, creation of "parameters*.csv" file, creation of target weights
for i in range(5):
    os.chdir(f'./{i}/')
    creation_parameters_file = glob.glob('parameters*.csv', recursive=False)[0]
    start_time = Path(creation_parameters_file).stat().st_mtime
    start_time = datetime.fromtimestamp(start_time, local_timezone)

    training_finish = Path(f"target_network_after_{last_task}_task.pt").stat().st_mtime
    finish_time = datetime.fromtimestamp(training_finish, local_timezone)

    duration_sec = (finish_time - start_time).seconds
    full_duration = pd.Timedelta(time.strftime('%H:%M:%S', time.gmtime(duration_sec)))
    results.append([i,
                    pd.Timestamp(start_time).tz_convert(local_timezone.key).tz_localize(None),
                    pd.Timestamp(finish_time).tz_convert(local_timezone.key).tz_localize(None),
                    duration_sec,
                    full_duration])
    os.chdir(main_path)
dataframe = pd.DataFrame(results, columns=['model', 'start_time', 'finish_time',
                                           'duration_seconds', 'duration_hours'])

# Potentially remove some indices
if model_name == 'CIFAR_100_ResNet':
    dataframe.drop([0], axis=0, inplace=True)

print(f'Mean: {dataframe["duration_hours"].mean()}, \n'
      f'std dev: {dataframe["duration_hours"].std()}')
