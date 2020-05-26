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
| :--------: | :--------: | :-------------: | :-----------: | -------- | ---------- |
| RSNA Brain | 224 x 224  |     564,601     |    150,560    | 6        | Yes        |
|    ACDC    | 400 x 400  |                 |       ?       | ?        | No         |
|    ISIC    | 224 x 224  |     25,979      |     6088      | 9        | No         |

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
Results on RSNA data

|    Pecentage of labeled data    | MoCo       | MoCo_efficient  | ResNet50      | MoCo_super | 
| :------------------------:      | :--------: | :-------------: | :-----------: | ---------- |
|           100                   | 0.9112     | 0.9036          | 0.9250        |0.9419      |
|            50                   | 0.8967     | 0.8986          | 0.8935        |0.9298      | 
|            20                   | 0.8865     | 0.8958          | 0.8567        |0.9192      |
|            10                   | 0.8702     | 0.8751          | 0.8476        |0.9028      | 
|             5                   | 0.8599     | 0.8686          | 0.8402        |0.8837      | 
|             1                   | 0.7751     | 0.7907          | 0.7462        |0.8511      |

Results on ISIC data

|    Pecentage of labeled data    | MoCo       | MoCo_V2         | ResNet50      | SimCLR     | 
| :------------------------:      | :--------: | :-------------: | :-----------: | ---------- |
|           100                   | 0.6467     | 0.6611          | 0.6407        |0.6541      |
|            50                   | 0.6358     | 0.6565          | 0.6565        |0.6442      | 
|            20                   | 0.6200     | 0.6421          | 0.6421        |0.6310      |
|            10                   | 0.6181     | 0.6355          | 0.6355        |0.6266      | 
|             5                   | 0.6131     | 0.6236          | 0.6236        |0.6133      | 
|             1                   | 0.5817     | 0.5947          | 0.5947        |0.5906      |

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please contact a team member before starting work on a pull request.

## License

[MIT](https://choosealicense.com/licenses/mit/)
