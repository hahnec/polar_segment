# polar_segment

This is a polarimetric image segmentation framework developed for brain tumor identification. This framework uses the `NPP dataset` (formerly `TumorMeasurementsCalib`; see dataloader in git history) containing per-pixel Mueller matrices of ex-vivo brain tissues. Nonetheless, swapping the dataset loading file `horao_dataset.py` with a custom dataset should be possible as long as the labels of the custom dataset are arranged accordingly. The repo uses the following related frameworks:

- [mm_torch](https://github.com/hahnec/mm_torch) for deterministic Mueller matrix processing
- [polar_augment](https://github.com/hahnec/polar_augment) for data augmentation tailored for polarimetry

## Citation

<pre>@article{hahne:2025:polar_segment,
  author={},
  journal={}, 
  title={}, 
  year={},
  volume={},
  number={},
  pages={},
  keywords={},
  doi={}
}</pre>

<pre>@misc{hahne2024isometrictransformationsimageaugmentation,
      title={Physically Consistent Image Augmentation for Deep Learning in Mueller Matrix Polarimetry}, 
      author={Christopher Hahne and Omar Rodriguez-Nunez and Éléa Gros and Théotim Lucas and Ekkehard Hewer and Tatiana Novikova and Theoni Maragkou and Philippe Schucht and Richard McKinley},
      year={2024},
      eprint={2411.07918},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.07918}, 
} </pre>

## Installation

### Dependencies

```bash
$ git clone github.com/hahnec/polar_segment
$ cd polar_segment
$ bash install.sh
```

### Models

1. Download a set of models, e.g. from our publication (please [cite our work](#citation)):

  - *HF link*

2. After successful download, place the models under the `polar_segment/ckpts/` directory.

3. Adjust the configuration by changing the `model_file` variable in `polar_segment/configs/infer.yml` for inference or `polar_segment/configs/train_local.yml` for training.

### Data

The release of the herein used dataset is yet to come. Please send a request to `elearomy.gros@unibe.ch` or `theoni.maragkou@unibe.ch` and ask for permission to download and use this dataset.

## Run

Before running, make sure the previously installed environment is activated via `$ source venv/bin/activate`.

### Inference
```bash
$ python3 infer.py
```

### Training
```bash
$ python3 train.py
```

## Acknowledgment

This work was supported by the Swiss National Science Foundation (SNSF) Sinergia Grant No. CRSII5\_205904, "HORAO - Polarimetric visualization of brain fiber tracts for tumor delineation in neurosurgery." We thank the Translational Research Unit, Institute of Tissue Medicine and Pathology, University of Bern, for their assistance with histology and acknowledge UBELIX, the HPC cluster at the University of Bern (https://www.id.unibe.ch/hpc), for computational resources.
