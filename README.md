# Dogs-vs-Cats

A fun project to differentiate dogs from cats.
Dataset is from Kaggle: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition.
Download it and create a `data` folder, put `train.zip` and `test.zip` into it.

## Getting Started

#### Train data
`python dogs_vs_cats_training.py`

#### Generate submissions for kaggle
`python create_submissions.py`

#### Notes
- Pretrained VGG16 model: `dogs_vs_cats_training_pretrained.py`
- Pretrained VGG16 model with image augmentation: `dogs_vs_cats_training_pretrained_optimized.py`
- Submission generation script: `create_submissions_pretrained.py`

## Copyright

See [LICENSE](LICENSE) for details.
Copyright (c) 2016 [Dat Tran](http://www.dat-tran.com/).