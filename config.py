# Model Configuration
model_name = 'resnet18'
pretrained = True
num_classes = 10

#trainsforms
#mean and std from imagenet, could consider calculating for this dataset
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# Training Configuration
val_size = 0.2
learning_rate = 0.001
num_epochs = 50
batch_size = 64
momentum = 0.9
weight_decay = 0.0005

# Learning Rate Scheduler Configuration
scheduler = 'StepLR'
step_size = 10
gamma = 0.1

# Data Configuration
train_data_path = '/path/to/train/data'
val_data_path = '/path/to/validation/data'

# Other Configurations
save_model_path = '/path/to/save_model.pth'
