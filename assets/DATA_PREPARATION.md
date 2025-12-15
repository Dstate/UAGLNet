## Data Preprocessing

We follow [BuildingExtraction](https://github.com/stdcoutzrh/BuildingExtraction) to preprocess all the datasets. Please download the [Inria](https://project.inria.fr/aerialimagelabeling/), [WHU](https://gpcv.whu.edu.cn/data/building_dataset.html), and [Massachusetts](https://www.cs.toronto.edu/~vmnih/data/) datasets, and organize them as follows.

  
### Preparation

Prepare the following folders to organize the datasets:
```none
├──data
├── ├── AerialImageDataset (i.e. INRIA)
│   │   ├── val_images (splited original images, ID 1-5 of each city)
│   │   ├── vak_masks (splited original masks, ID 1-5 of each city)
│   │   ├── train_images (splited original images, the other IDs)
│   │   ├── train_masks (splited original masks,  the other IDs)
├── ├── WHU
│   │   ├── train
│   │   │   ├── images (original images)
│   │   │   ├── masks_origin (original masks)
│   │   ├── val
│   │   │   ├── images (original images)
│   │   │   ├── masks_origin (original masks)
│   │   ├── test
│   │   │   ├── images (original images)
│   │   │   ├── masks_origin (original masks)
├── ├── Massachusetts
│   │   ├── raw
│   │   │   ├── train (original images)
│   │   │   ├── train_labels (original masks, RGB format)
│   │   │   ├── val (original images)
│   │   │   ├── val_labels (original masks, RGB format)
│   │   │   ├── test (original images)
│   │   │   ├── test_labels (original masks, RGB format)
```


### Inria

```bash
# preprocess training set
python tools/inria_patch_split.py \
--input-img-dir "data/AerialImageDataset/train_images" \
--input-mask-dir "data/AerialImageDataset/train_masks" \
--output-img-dir "data/AerialImageDataset/train/images" \
--output-mask-dir "data/AerialImageDataset/train/masks" \
--mode "train"

# preprocess testing set
python tools/inria_patch_split.py \
--input-img-dir "data/AerialImageDataset/val_images" \
--input-mask-dir "data/AerialImageDataset/val_masks" \
--output-img-dir "data/AerialImageDataset/val/images" \
--output-mask-dir "data/AerialImageDataset/val/masks" \
--mode "val"
```

### WHU

Note: Download the WHU aerial imagery dataset.

```bash
# preprocess training set
python tools/whubuilding_mask_convert.py \
--mask-dir "data/WHU/train/masks_origin" \
--output-mask-dir "data/WHU/train/masks" 

# preprocess validation set
python tools/whubuilding_mask_convert.py \
--mask-dir "data/WHU/val/masks_origin" \
--output-mask-dir "data/WHU/val/masks" 

# preprocess testing set
python tools/whubuilding_mask_convert.py \
--mask-dir "data/WHU/test/masks_origin" \
--output-mask-dir "data/WHU/test/masks" 
```

### Massachusetts

Note: Download the Massachusetts buildings dataset.

```bash
# preprocess training set
python tools/mass_patch_split.py \
--input-img-dir "data/Massachusetts/raw/train" \
--input-mask-dir "data/Massachusetts/raw/train_labels" \
--output-img-dir "data/Massachusetts/buildE/train/images" \
--output-mask-dir "data/Massachusetts/buildE/train/masks" \
--mode "train"

# preprocess validation set
python tools/mass_patch_split.py \
--input-img-dir "data/Massachusetts/raw/val" \
--input-mask-dir "data/Massachusetts/raw/val_labels" \
--output-img-dir "data/Massachusetts/buildE/val/images" \
--output-mask-dir "data/Massachusetts/buildE/val/masks" \
--mode "val"

# preprocess testing set
python tools/mass_patch_split.py \
--input-img-dir "data/Massachusetts/raw/test" \
--input-mask-dir "data/Massachusetts/raw/test_labels" \
--output-img-dir "data/Massachusetts/buildE/test/images" \
--output-mask-dir "data/Massachusetts/buildE/test/masks" \
--mode "val"

# crop the images into 512x512
python tools/split_1500_to_512.py \
--path "data/Massachusetts/buildE" \
--save_path "data/Massachusetts/mass_512" 

```

### Final Folder Structure
Finally, you should have the following folders and structure::

```none
├──data
├── ├── AerialImageDataset (i.e. INRIA)
│   │   ├── train
│   │   │   ├── images (processed)
│   │   │   ├── masks (processed)
│   │   ├── val
│   │   │   ├── images (processed)
│   │   │   ├── masks (processed)
│   │   │   ├── masks_gt (processed, for visualization)
├── ├── WHU
│   │   ├── train
│   │   │   ├── images (original images)
│   │   │   ├── masks (processed)
│   │   ├── val
│   │   │   ├── images (original images)
│   │   │   ├── masks (processed)
│   │   ├── test
│   │   │   ├── images (original images)
│   │   │   ├── masks (processed)
├── ├── Massachusetts
│   │   ├── mass_512
│   │   │   ├── train
│   │   │   │   ├── images (processed)
│   │   │   │   ├── masks (processed)
│   │   │   ├── val
│   │   │   │   ├── images (processed)
│   │   │   │   ├── masks (processed)
│   │   │   ├── test
│   │   │   │   ├── images (processed)
│   │   │   │   ├── masks (processed)
```