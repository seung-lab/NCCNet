# NCCNet
Template  matching  by  normalized  cross  correlation (NCC) is widely used for finding image correspondences. We improve the robustness of this algorithm by preprocess-ing images with "siamese" convolutional networks trained to maximize the contrast between NCC values of true and false matches.

## Usage
Start docker
```
nvidia-docker run -it --net=host \
      -v PATH_TO_CRACK_FOLD_DETECTOR_REPO:/PredUnet \
      davidbuniat/cavelab:latest-gpu bash

cd /project
```

To train the model defined in hparams.json run the following code

```
python src/model.py
```
Logs and trained models will appear in `logs/` folder. Please change name in hparams.json on each experiment.


## Data
For training the model you will require to have pairs of image, templates defined by TFRecords files in `/data/bad_trainset_24000_612_324.tfrecords`.

Data collection done using script defined in `src/prepare_data.py`. Even though for now it works with h5 files, will switch soon to CloudVolume.
