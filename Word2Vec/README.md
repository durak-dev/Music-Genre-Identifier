# Word2Vec + Neural Network

These notebooks were run on Google Colab.

This is not the final code. Further clean up and improvements will be done.

`word2vec_tuning.ipynb` -> Word2Vec parameter selection

`word2vec_nn.ipynb` -> Macro F1 is the same as the simple logistic regression used in parameter selection (0.413 vs 0.415 on the validation set). This will be improved with changes (Averaging done to the data fed into the Neural Network is the main bottleneck).

## File system

```
Word2Vec/
├── README.md
├── code/
│   ├── word2vec_tuning.ipynb
│   └── word2vec_nn.ipynb
└── output_files/
    ├── word2vec_param_testing.csv
    ├── hyperband_top5.csv
    ├── nn_training_log.csv
    └── submission_w2v_nn.csv
```

### code/

| File | Description |
|---|---|
| `word2vec_tuning.ipynb` | Compares 14 Word2Vec configurations (one parameter changed at a time) with a logistic regression probe. Ends with the chosen configuration. |
| `word2vec_nn.ipynb` | Trains the final Word2Vec, builds one vector per song, tunes a Keras neural network with Hyperband, trains the final network and writes the submission. |

### output_files/ (in the order they are produced)

| File | Created by | Description |
|---|---|---|
| `word2vec_param_testing.csv` | `word2vec_tuning.ipynb` | Results of the 14 Word2Vec configurations (accuracy, macro F1, weighted F1, vocabulary size, seconds). |
| `hyperband_top5.csv` | `word2vec_nn.ipynb` | Best Hyperband trials with validation loss and validation macro F1. The final network configuration is taken from here. |
| `nn_training_log.csv` | `word2vec_nn.ipynb` | Per-epoch loss, accuracy and learning rate of the final training. |
| `submission_w2v_nn.csv` | `word2vec_nn.ipynb` | Test predictions as `id,tag`, in the order of the sample submission. |

Large intermediate files (cleaned data, word vectors, song vectors, the trained network and the Hyperband search folder) are not in the repository (see `.gitignore`). Running the notebooks recreates them.
