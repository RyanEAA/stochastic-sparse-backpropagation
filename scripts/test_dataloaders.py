import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_dataloaders

datasets_to_test = [
    "Fashion-MNIST",
    "CIFAR-10",
    "CIFAR-100",
    "SVHN",
    "KMNIST",
    "Tiny-ImageNet",
    "UCI-Adult",
    "Covertype",
]

results = {}
for name in datasets_to_test:
    print(f"Testing {name} ...")
    try:
        train_loader, test_loader = get_dataloaders(name, batch_size_train=16, batch_size_test=32, data_dir="./data", num_workers=0)
        batch = next(iter(train_loader))
        x, y = batch
        print(f"  train batch: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}, x.dtype={x.dtype}, y.dtype={y.dtype}")
        batch_test = next(iter(test_loader))
        xt, yt = batch_test
        print(f"  test batch: x.shape={tuple(xt.shape)}, y.shape={tuple(yt.shape)}")
        results[name] = "ok"
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        results[name] = f"error: {type(e).__name__}: {e}"

print('\nSummary:')
for k,v in results.items():
    print(f" - {k}: {v}")
