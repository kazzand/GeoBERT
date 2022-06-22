# GeoBERT
Predictiong geolocation by text of tweets.
## Model overview
I used multilingual bert as text encoder + MLP and train the net as a regression task. I decoded coordinates into the countries and fine-tune bert encoder as metric learning task, to get vectors of texts from same countries much closer. The best average MAE in kilometers obtained is 720 km on validation. I provide you evaluation function so it'll be not that hard to check model performance on your data. In case you have any questions, please do not hesitate to contact me: e-mail - kazzand@yandex.ru

## Directories overview
1. Datasets - directory contains custom torch.Dataset classes
2. Models - directory contains model classes
3. Data - contains tuned model and your train data preprocessed. Use git lfs to download these files or use this [link](https://drive.google.com/file/d/1nYQ4g0jqxihEw9kpXsBysY--QcoVa0sp/view?usp=sharing) to download  model checkpoints and add to data directory directly.
4. Notebooks - contains jupyter notebooks with training and evaluating. Use evaluation.ipynb to evaluate model performance on your data.
