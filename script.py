import os

models = [
    'mlp',
    'cnn',
    'lstm',
    'gru'
]


learning_rate = 0.01
batch_size = 64
epochs = 100
gpu_id = 0

# Preprocess
command_preprocess = 'python preprocess.py'

# Train
command_train = ['python run.py --model {} --mode train -l {} -b {} -e {} --gpu-id {}'. \
    format(model, learning_rate, batch_size, epochs, gpu_id) for model in models]


# Predict 
command_predict = ['python run.py --model {} --mode predict -b {} --gpu-ids {}'. \
    format(model, batch_size, gpu_id) for model in models]


if command_preprocess is not None:
    os.system(command_preprocess)

for i in range(len(models)):
    os.system(command_train[i])
    os.system(command_predict[i])
