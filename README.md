# Land use from Sentinel-2: how much do pretraining and extra spectral bands help?

Classifying 64×64 Sentinel-2 tiles from the [EuroSAT](https://github.com/phelber/EuroSAT) dataset into 10 land-use classes. A fine-tuned ResNet already reaches high accuracy on this benchmark, so the goal here is not one more accuracy number. These experiments test three questions that matter when the labelled data is satellite imagery:

1. **Pretraining.** ImageNet photos look nothing like satellite tiles. Does starting from ImageNet weights still help, and how much more does it help when only a few hundred labelled tiles are available?
2. **Spectral bands.** Sentinel-2 records 13 bands, from visible light to short-wave infrared, but most models use only RGB. Do the other 10 bands improve accuracy, and for which classes?
3. **Symmetry.** An overhead tile has no "up". Rotating or mirroring it gives the same land use, so the 8 symmetries of the square (the dihedral group D4) can be used both for training augmentation and for averaging predictions at test time. How much does that averaging add?

## Data

EuroSAT MS: 27,000 tiles, 10 classes, 13 bands, 10–60 m resolution, from the [official Zenodo release](https://zenodo.org/records/7711810) (MIT licence). RGB models use bands B04, B03 and B02 of the same files, so every comparison is between models that saw exactly the same images.

Split: stratified 70 / 10 / 20 (train / validation / test), seeded and saved in `splits/`. The validation set picks the checkpoint. The test set is used once per run.

## Setup

| Choice | Value | Why |
|---|---|---|
| Models | Small CNN (1.2M parameters), ResNet-18 (11.2M) | A small from-scratch reference, plus a standard architecture that has ImageNet weights |
| 13-band input | First conv rebuilt: RGB filters copied to B04/B03/B02, other bands start at the mean RGB filter, everything scaled by 3/13 | Keeps the pretrained RGB pathway intact and the first layer's activation scale unchanged (`src/models.py`) |
| Normalisation | Per-band mean and std from the training split only | No information from validation or test data |
| Augmentation | Random element of D4 (rotation by k·90°, optional mirror) | Exact symmetries of the task; no interpolation artefacts |
| Optimiser | AdamW, weight decay 0.05, 100 warm-up steps then cosine decay; lr 1e-3 (scratch), 3e-4 (pretrained) | Lower lr so fine-tuning does not erase the pretrained features |
| Budget | 2,000 steps at batch size 128 for every run | Equal compute for every training-set size, so label-efficiency curves compare data, not training time |
| Evaluation | Test accuracy of the best-validation checkpoint, with and without averaging logits over the 8 D4 views | Answers question 3 at no extra training cost |

## Results

> **TODO:** paste the tables from `results/summary.md` and the figures from `results/figures/` after running the notebook.

### Full training set

<!-- table: main results -->

### Label efficiency

<!-- table + results/figures/label_efficiency.png -->

### Where the extra bands help

<!-- results/figures/band_gain.png -->

### Errors

<!-- results/figures/confusion_best.png, results/figures/mistakes_best.png, most-confused pairs -->

## Findings

> **TODO:** for each of the three questions, write 2–4 sentences on what the numbers show, including anything surprising or anything that did not work.

## Limitations

- **Spatial leakage.** The split is random over tiles, and EuroSAT tiles cut from the same Sentinel-2 scene can be next to each other. Neighbouring tiles in train and test make the task easier than classifying a new region, so absolute accuracies are probably optimistic. Comparisons between models share the same bias. A geographic split that holds out whole regions would measure that directly.
- **Seeds.** Full-data runs use one seed each. The label-efficiency runs use three, and their spread is shown, so differences smaller than that spread should not be read as real.
- **Scope.** One architecture family at the native 64×64 resolution. Upsampling the tiles or using a remote-sensing-pretrained backbone could change the answers.

## Reproduce

Open `notebooks/run_on_colab.ipynb` in Google Colab with a GPU runtime, or locally:

```bash
pip install -r requirements.txt
python -c "from src import data as D; D.download('data')"
python -m src.experiments main     # 5 models, full training set
python -m src.experiments label    # 1%, 5%, 20% of the labels x 3 seeds
python -m src.analysis             # tables -> results/summary.md, figures -> results/figures/
```

A single configuration: `python -m src.train --name my_run --arch resnet18 --pretrained --bands all`.

## Layout

```
src/data.py         download, packing into one memory-mapped array, split, band statistics, D4 maps
src/models.py       small CNN, ResNet-18, 13-band first-layer construction
src/train.py        training loop, evaluation with and without D4 averaging, results JSON
src/experiments.py  the full list of runs
src/analysis.py     tables and figures
splits/             fixed split indices and band statistics
results/            one JSON per run, summary.md, figures/
```

## Reference

P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE J-STARS, 2019.
