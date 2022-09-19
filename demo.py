from neural_clamping.nc_wrapper import NCWrapper
from neural_clamping.utils import load_model, load_dataset, model_classes, setup_device, download_gdrive


# config
args = {"model_name": 'ResNet-110', "image_size": (3, 32, 32), "data": 'CIFAR-100', "lambda_0": 1, "lambda_1": 0.01,
        "dropout_p": 0.0, "focal_loss": True, "gamma": 1, "init_scale": 0.001, "epoch": 100, "lr": 0.01,
        "init_temp": 1.0, "n_gpu": 1, "batch_size": 1000}

# gpu devices setup
device, device_ids = setup_device(n_gpu_use=args["n_gpu"])
print(device, device_ids)

# download pretrained model and neural clamping parameters
download_gdrive("1qnjazEVCa-0DHT8C7zKy5VQck8YC_cF0", "resnet110-cifar100.pt")

# load model
model = load_model(name=args["model_name"], data=args["data"], checkpoint_path='resnet110-cifar100.pt')
num_classes = model_classes(data=args["data"])

# dataset loader
valloader = load_dataset(data=args["data"], split='val', batch_size=args["batch_size"])
testloader = load_dataset(data=args["data"], split='test', batch_size=args["batch_size"])

# build Neural Clamping framework
nc = NCWrapper(model=model, num_classes=num_classes, lambda_0=args["lambda_0"], lambda_1=args["lambda_1"],
               dropout_p=args["dropout_p"], init_scale=args["init_scale"], init_temp=args["init_temp"],
               image_size=args["image_size"], device_ids=device_ids)

# training
result = nc.train_NC(val_loader=valloader, epoch=args["epoch"], lr=args["lr"], focal_loss=args["focal_loss"],
                     gamma=args["gamma"])

# Testing
nc.test_with_NC(test_loader=testloader)
