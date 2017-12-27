# NCCNet
Template  matching  by  normalized  cross  correlation (NCC) is widely used for finding image correspondences. We improve the robustness of this algorithm by preprocessing images with "siamese" convolutional networks trained to maximize the contrast between NCC values of true and false matches.

## Usage
Start docker environment
```
nvidia-docker run -it --net=host \
      -v PATH_TO_NCCNet_REPO:/NCCNet \
      davidbuniat/cavelab:latest-gpu bash

cd /NCCNet
```

To train the model defined in `hparams.json` run the following code. Parameters in the json file are self-explanatory.

```
python src/train.py
```
Logs and trained models will appear in `logs/` folder. Please change name in `hparams.json` for each experiment to log with different names, otherwise it will throw an error.


## Data
For training the model you will require to have pairs of image-templates defined by TFRecords files in `/data/tf/bad_trainset_24000_612_324.tfrecords`. If you are part of seunglab you can find this file in `seungmount/research/davit/NCCNet/data/tf/bad_trainset_24000_612_324.tfrecords`
