# Medical Image Representation Learning

MIRL is an exploratory project on efficient representation learning on medical imagery built by the SMILE lab at UF

## Installation

Use the package manager [conda](https://www.anaconda.com/) to install MIRL.

```bash
conda env create -f environment.yml
```

## Model Architectures & Methodology

- Semi-Supervised Learning
  - Unsupervised training stage
    - [Momentum Contrast](https://arxiv.org/pdf/1911.05722.pdf) (MoCo)
    - [Simple Contrastive Learning of Representations](https://arxiv.org/pdf/2002.05709.pdf) (SimCLR)
    - [Momentum Contrast Version 2](https://arxiv.org/pdf/1911.05722.pdf) (MoCoV2)
  - Finetuning stage (downstream task)
    - Linear Classifier which freezes all backbone encoder weights (enabled with **--frozen**)
    - Transfer Learning which retrains all layers (default)
- Supervised Learning
  - ResNet-50

## Datasets

|    Name    | Image Size | Unlabeled Count | Labeled Count | #Classes | Multilabel |
| :--------: | :--------: | :-------------: | :-----------: | :------: | :--------: |
| RSNA Brain | 224 x 224  |     564,601     |    150,560    |    6     |    Yes     |
|    ACDC    | 400 x 400  |        ?        |       ?       |    ?     |     No     |
|    ISIC    | 224 x 224  |     25,979      |     6,088     |    9     |     No     |
|    Fundus  | 400 x 400  |     26,344      |     7025      |    5     |     No     |
## Training

```bash
cd RSNA_MoCo
```

- Semi-Supervised Learning

  - Unsupervised training stage
    - [Momentum Contrast](https://arxiv.org/pdf/1911.05722.pdf) (MoCo)
      ```bash
      python trainMoCo.py
      ```
    - [Simple Contrastive Learning of Representations](https://arxiv.org/pdf/2002.05709.pdf) (SimCLR)
      ```bash
      python trainSimCLR.py
      ```
    - [Momentum Contrast Version 2](https://arxiv.org/pdf/1911.05722.pdf) (MoCoV2)
      ```bash
      python trainMoCoV2.py
      ```
  - Finetuning stage (Requires e.g. **train10F.txt**)

    - Linear Classifier which freezes all backbone encoder weights

      ```bash
      python MoCo_downstream.py --frozen
      ```

      or

      ```bash
      python MoCov2_downstream.py --frozen
      ```

    - Transfer Learning which retrains all layers (default)

      ```bash
      python MoCo_downstream.py
      ```

      or

      ```bash
      python MoCov2_downstream.py
      ```

- Supervised Learning

  - ResNet-50

    ```bash
    python MoCo_downstream.py --resnet
    ```

    or

    ```bash
    python MoCov2_downstream.py --resnet
    ```

## Results

Results on RSNA data Metric: AUC

| Pecentage of labeled data |  MoCo  | MoCo_efficient | ResNet50 | MoCo_super | MoCoV2_super |
| :-----------------------: | :----: | :------------: | :------: | :--------: | :----------: |
|            100            | 0.9112 |     0.9036     |  0.9250  |   0.9419   |    0.9638    |
|            50             | 0.8967 |     0.8986     |  0.8935  |   0.9298   |    0.9519    |
|            20             | 0.8865 |     0.8958     |  0.8567  |   0.9192   |    0.9344    |
|            10             | 0.8702 |     0.8751     |  0.8476  |   0.9028   |    0.9229    |
|             5             | 0.8599 |     0.8686     |  0.8402  |   0.8837   |    0.8951    |
|             1             | 0.7751 |     0.7907     |  0.7462  |   0.8511   |    0.8177    |

Results on Fundus (Diabetic Retinopathy Detection) data Metric: AUC

| Pecentage of labeled data | Resnet50 | MoCo    | MoCoV2   | MoCo-FTAL |
| :-----------------------: | :----:   | :-----: | :------: | :----:    |
|            100            | 0.57411  | 0.6817  |  0.6407  | 0.7705    |
|            50             | 0.5541   | 0.6651  |  0.6565  | 0.7421    |
|            20             | 0.5463   | 0.6263  |  0.6421  | 0.6956    |
|            10             | 0.54320  | 0.5926  |  0.6355  | 0.6324    |
|             5             | 0.5380   | 0.5593  |  0.6236  | 0.5862    |
|             1             | 0.5369   | 0.5419  |  0.5947  | 0.5427    |

Results on ISIC data Metric: Accuracy

| Pecentage of labeled data |  MoCo  | MoCo_V2 | ResNet50 | SimCLR |
| :-----------------------: | :----: | :-----: | :------: | :----: |
|            100            | 0.6467 | 0.6611  |  0.6407  | 0.6541 |
|            50             | 0.6358 | 0.6565  |  0.6565  | 0.6442 |
|            20             | 0.6200 | 0.6421  |  0.6421  | 0.6310 |
|            10             | 0.6181 | 0.6355  |  0.6355  | 0.6266 |
|             5             | 0.6131 | 0.6236  |  0.6236  | 0.6133 |
|             1             | 0.5817 | 0.5947  |  0.5947  | 0.5906 |

Resutls on Brats18 data Metric: Dice
| Pecentage of labeled data |  deeplabv3  | deeplabv3+MoCo_super | deeplabv3+MoCo | Unet |
| :-----------------------: | :----: | :-----: | :------: | :----: |
|            100            | 0.6988 | 0.7261  |  0.7085  | 0.7387 |
|            50             | 0.6199 | 0.7082  |  0.6952  | 0.7110 |
|            20             | 0.5353 | 0.6255  |  0.6387  | 0.6552 |
|            10             | 0.3983 | 0.5667  |  0.5303  | 0.5545 |
|             5             | 0.3007 | 0.5267  |  0.5394  | 0.5585 |
|             1             | 0.1831 | 0.3579  |  0.3554  | 0.2655 |

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please contact a team member before starting work on a pull request.

## License

[MIT](https://choosealicense.com/licenses/mit/)
