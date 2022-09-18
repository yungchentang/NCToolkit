from neural_clamping.nc_wrapper import NCWrapper
from neural_clamping.utils import load_model, load_dataset, model_classes, setup_device


# config
model_name = 'ResNet-110'
image_size = (3, 32, 32)
data = 'CIFAR-100'
lambda_0 = 1
lambda_1 = 0.01
dropout_p = 0.0
focal_loss = True
gamma = 1
init_scale = 0.001
epoch = 100
lr = 0.01
init_temp = 1.0
n_gpu = 1
batch_size = 1000

# gpu devices setup
device, device_ids = setup_device(n_gpu_use=n_gpu)
print(device, device_ids)

# load model
model = load_model(name=model_name, data=data)
num_classes = model_classes(data=data)

# dataset loader
valloader = load_dataset(data=data, split='val', batch_size=batch_size)
testloader = load_dataset(data=data, split='test', batch_size=batch_size)

# build Neural Clamping framework
nc = NCWrapper(model=model,
               lambda_0=lambda_0,
               lambda_1=lambda_1,
               dropout_p=dropout_p,
               init_scale=init_scale,
               init_temp=init_temp,
               image_size=image_size,
               num_classes=num_classes,
               device_ids=device_ids)
# training
result = nc.train_NC(val_loader=valloader, epoch=epoch, lr=lr, focal_loss=focal_loss, gamma=gamma)

# Testing
nc.test_with_NC(test_loader=testloader)
