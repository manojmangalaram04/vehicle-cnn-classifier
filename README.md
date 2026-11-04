# VehicleVision - Vehicle Image Classification with a Custom CNN

A **custom Convolutional Neural Network trained from scratch** (no pretrained models) that classifies a
vehicle photo as **Bicycle, Car, Motorbike or Truck**, with a single-page Streamlit app
(drag-and-drop upload -> preview -> Predict -> class, confidence and probability bars).

## Results (measured on the independent test set, never used for training or model selection)

Test images: **197**  |  Test accuracy: **84.77%**  |  Test loss: **0.4629**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bicycle | 0.556 | 0.882 | 0.682 | 17 |
| Car | 0.925 | 0.817 | 0.867 | 60 |
| Motorbike | 0.891 | 0.817 | 0.852 | 60 |
| Truck | 0.871 | 0.900 | 0.885 | 60 |
| **Macro avg** | 0.810 | 0.854 | 0.822 | 197 |
| **Weighted avg** | 0.866 | 0.848 | 0.852 | 197 |

**Accuracy by image source (read this):**

| Source | Test images | Accuracy | Correct per class |
|---|---|---|---|
| existing photos | 68 | 67.6% | bicycle: 15/17, car: 6/16, motorbike: 14/18, truck: 11/17 |
| kaggle | 129 | 93.8% | bicycle: 0/0, car: 43/44, motorbike: 35/42, truck: 43/43 |

The model is clearly stronger on the Kaggle images than on your own photos. Kaggle car/truck images are traffic-camera crops with a shared background, so part of that score may come from the background rather than the vehicle. Expect lower accuracy on new real-world photos, especially cars; more varied photos per class (and many more bicycle photos) are the best way to improve it.

Training: 59 epochs run, stopped by EarlyStopping; best checkpoint = epoch 44 (validation loss 0.5108, validation accuracy 79.1%). Graphs and the confusion matrix are in `outputs/`.

Dataset after cleaning (per split): train: 915; validation: 196; test: 197. Removed before splitting: 8 exact and 3216 near duplicates, 0 corrupted.


## 1. Project structure

```
vehicle_cnn_project/
├── app.py               single-page Streamlit app
├── preprocess.py        extract / inspect / clean / de-duplicate / stratified 70-15-15 split
├── train.py             custom CNN + training (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
├── evaluate.py          test-set-only evaluation, graphs and reports
├── requirements.txt
├── .streamlit/config.toml   dark theme
├── data/                train / validation / test  (bicycle, car, motorbike, truck)
├── models/              vehicle_classifier.keras, class_names.json
├── outputs/             metrics.json, confusion_matrix.png, accuracy_graph.png, loss_graph.png,
│                        classification_report.*, history.csv, dataset_report.json
└── utils/helpers.py     paths, class list, image helpers
```

## 2. Dataset

Final classes: `bicycle`, `car`, `motorbike`, `truck` (one list in `models/class_names.json`, used by training,
evaluation and the app).

| Source | Used for | Notes |
|---|---|---|
| Your own photos ("existing photos") | all four classes | all kept |
| Kaggle vehicle dataset with folders `Car`, `Motorcycle`, `Truck`, `Van` | car, motorbike (`Motorcycle`), truck | `Van` not used. No bicycle class. |

Kaggle also hosts `mohamedmaher5/vehicle-classification` (https://www.kaggle.com/datasets/mohamedmaher5/vehicle-classification),
which lists bicycle, car, motorcycle, auto rickshaw, airplane, ship and train. It is a possible source of more
**bicycle** images. It was not downloaded or used in this project (the build environment could not reach Kaggle),
so its label quality is unchecked. Roboflow Universe also has "vehicle classification" projects, but they were not
verified either.

**Important limitations of the data (please read)**

* Bicycle has only the few images from your own photos, far fewer than the other classes. Training uses
  class-balanced sampling to compensate, but more bicycle photos would help most.
* In the Kaggle data, Car and Truck (and Van) are small crops from a fixed highway camera with a grey-road background,
  while Motorcycle images and your own photos are ordinary photos. A CNN can pick up on the background instead of the
  vehicle. That is why `evaluate.py` also reports accuracy **per source**; compare "existing photos" with "kaggle".
* Near-duplicate images (e.g. consecutive camera frames of one vehicle) are removed before splitting so they cannot
  leak between train/validation/test.

### Rebuild the dataset yourself (Windows, from the ZIP files)

```bat
python preprocess.py --source existing_raw_folder kaggle_folder ^
    --classes car bicycle motorbike truck ^
    --source-labels "existing photos" "kaggle" --max-per-class 400
```

`--zip file.zip` also works. The **first** source is always kept in full when `--max-per-class` is used.
Folder-name aliases (`motor bike`, `motorcycle`, `bike` -> `motorbike`; Bicycle and Motorbike stay separate) are
handled in `preprocess.py`. Corrupted files are skipped, exact (MD5) and near duplicates (perceptual hash) are
removed, then a stratified 70/15/15 split is written to `data/`.

## 3. Install on Windows

1. Install **Python 3.11 or 3.12 (64-bit)** from https://www.python.org/downloads/ and tick *Add python.exe to PATH*.
2. Open **Command Prompt** in the project folder:

```bat
cd vehicle_cnn_project
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run

```bat
:: only needed if you want to rebuild / retrain (the ZIP already contains data, model and results)
python preprocess.py --source existing_raw_folder kaggle_folder --classes car bicycle motorbike truck --source-labels "existing photos" "kaggle" --max-per-class 400
python train.py          :: add --resume to continue an interrupted run
python evaluate.py       :: test-set metrics + graphs into outputs\

:: start the web app
streamlit run app.py
```

Then open http://localhost:8501.

## 5. Model

Input 128x128x3 -> pixel rescaling -> 5x5 stride-2 stem -> 4 blocks of [2 x (Conv + BatchNorm + ReLU) -> MaxPool ->
SpatialDropout] with 48/96/160/256 filters -> Global Average Pooling -> Dropout -> Dense(128) -> BatchNorm ->
Dropout -> Dense(softmax, one neuron per class). Regularisation: augmentation (flip, rotation, zoom, shift,
contrast, brightness; training only), mixup, spatial dropout, dropout, weight decay (AdamW).
Callbacks: EarlyStopping (val_loss), ModelCheckpoint (best val_loss), ReduceLROnPlateau.

## 6. Deploy on Streamlit Community Cloud

1. Create a GitHub repository and push the project:
```bat
git init
git add .
git commit -m "VehicleVision"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
`.gitignore` excludes `data/` (not needed by the app). `models/` and `outputs/` are committed.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**, pick the repository, branch `main`,
main file `app.py`, then **Deploy**. Dependencies come from `requirements.txt` (CPU TensorFlow).
3. The model file is only a few MB, so no Git LFS is needed. For files over 100 MB use Git LFS
(`git lfs track "*.keras"`) or host the file elsewhere and download it at start-up.

I could not run an actual Streamlit Cloud deployment from the build environment; the app was tested locally.
