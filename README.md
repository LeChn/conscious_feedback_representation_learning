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
|    ACDC    | 400 x 400  |        ?        |       ?       | ?        | No         |
|    ISIC    |     ?      |        ?        |       ?       | ?        | ?          |

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
      python trainMoCoV2.py
      ```
    - [Momentum Contrast Version 2](https://arxiv.org/pdf/1911.05722.pdf) (MoCoV2)
      ```bash
      python trainSimCLR.py
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

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please contact a team member before starting work on a pull request.

## License

[MIT](https://choosealicense.com/licenses/mit/)
