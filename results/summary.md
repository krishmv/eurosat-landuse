# Results

## Full training set

| Model | Params | Test acc. | Test acc. with D4 averaging | Train time (s) |
|---|---:|---:|---:|---:|
| Small CNN, scratch, RGB | 1.2M | 97.41% | 97.80% | 81 |
| ResNet-18, scratch, RGB | 11.2M | 95.15% | 95.85% | 69 |
| ResNet-18, ImageNet, RGB | 11.2M | 98.57% | 98.98% | 70 |
| ResNet-18, scratch, 13 bands | 11.2M | 97.89% | 98.11% | 134 |
| ResNet-18, ImageNet, 13 bands | 11.2M | 98.76% | 98.89% | 130 |

Best model: **ResNet-18, ImageNet, RGB**.

## Most confused class pairs (best model)

| Class pair | Errors (both directions) |
|---|---:|
| AnnualCrop ↔ PermanentCrop | 8 |
| HerbaceousVegetation ↔ PermanentCrop | 7 |
| HerbaceousVegetation ↔ Pasture | 7 |
| AnnualCrop ↔ Pasture | 7 |
| Highway ↔ River | 6 |

## Label efficiency (test accuracy %, D4 averaging)

| Model | 1% labels | 5% labels | 20% labels | 100% labels |
|---|---:|---:|---:|---:|
| ResNet-18, scratch, RGB | 71.6 ± 0.4 | 82.7 ± 1.1 | 92.9 ± 0.7 | 95.9 |
| ResNet-18, ImageNet, RGB | 90.5 ± 0.6 | 96.3 ± 0.2 | 98.0 ± 0.1 | 99.0 |
| ResNet-18, ImageNet, 13 bands | 90.6 ± 0.8 | 96.4 ± 0.3 | 98.1 ± 0.2 | 98.9 |
