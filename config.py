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
learning_rate = 0.0001
num_epochs = 100
batch_size = 16

# Learning Rate Scheduler Configuration
scheduler = 'StepLR'
step_size = 1
gamma = 0.96

#data config
data_path = "C:/Users/Daniel/Pictures/plant_data"


# Other Configurations
save_model_path = 'models'
