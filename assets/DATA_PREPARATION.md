### Data Preprocessing

We follow [BuildingExtraction](https://github.com/stdcoutzrh/BuildingExtraction) to preprocess all the datasets. Please download the [Inria](https://project.inria.fr/aerialimagelabeling/), [WHU](https://gpcv.whu.edu.cn/data/building_dataset.html), and [Massachusetts](https://www.cs.toronto.edu/~vmnih/data/) datasets, and organize them as follows.

  
**Preparation**

Prepare the following folders to organize this repo:
```none
├── AerialImageDataset (i.e. INRIA)
│   ├── train
│   │   ├── val_images (splited original images, ID 1-5 of each city)
│   │   ├── vak_masks (splited original masks, ID 1-5 of each city)
│   │   ├── train_images (splited original images, the other IDs)
│   │   ├── train_masks (splited original masks,  the other IDs)
│   │   ├── train
│   │   │   ├── images (processed)
│   │   │   ├── masks (processed)
│   │   ├── val
│   │   │   ├── images (processed)
│   │   │   ├── masks (processed)
│   │   │   ├── masks_gt (processed, for visualization)
├── mass_build
│   ├── png
│   │   ├── train (original images)
│   │   ├── train_labels (original masks, RGB format)
│   │   ├── train_images (processed images)
│   │   ├── train_masks (processed masks, unit8 format)
│   │   ├── val (original images)
│   │   ├── val_labels (original masks, RGB format)
│   │   ├── val_images (processed images)
│   │   ├── val_masks (processed masks, unit8 format)
│   │   ├── test (original images)
│   │   ├── test_labels (original masks, RGB format)
│   │   ├── test_images (processed images)
│   │   ├── test_masks (processed masks, unit8 format)
├── whubuilding
│   ├── train
│   │   ├── images (original images)
│   │   ├── masks_origin (original masks)
│   │   ├── masks (converted masks)
│   ├── val (the same with train)
│   ├── test (the same with test)
│   ├── train_val (Merge train and val)
```


**INRIA**

```bash
# preprocess training set
python tools/inria_patch_split.py \
--input-img-dir "data/AerialImageDataset/train/train_images" \
--input-mask-dir "data/AerialImageDataset/train/train_masks" \
--output-img-dir "data/AerialImageDataset/train/train/images" \
--output-mask-dir "data/AerialImageDataset/train/train/masks" \
--mode "train"

# preprocess testing set
python tools/inria_patch_split.py \
--input-img-dir "data/AerialImageDataset/train/val_images" \
--input-mask-dir "data/AerialImageDataset/train/val_masks" \
--output-img-dir "data/AerialImageDataset/train/val/images" \
--output-mask-dir "data/AerialImageDataset/train/val/masks" \
--mode "val"
```

**WHU**

Download the WHU aerial imagery dataset.

```bash
# preprocess training set
python tools/whubuilding_mask_convert.py \
--mask-dir "data/whubuilding/train/masks_origin" \
--output-mask-dir "data/whubuilding/train/masks" 

# preprocess validation set
python tools/whubuilding_mask_convert.py \
--mask-dir "data/whubuilding/val/masks_origin" \
--output-mask-dir "data/whubuilding/val/masks" 

# preprocess testing set
python tools/whubuilding_mask_convert.py \
--mask-dir "data/whubuilding/test/masks_origin" \
--output-mask-dir "data/whubuilding/test/masks" 
```

**Massachusetts**

Download the Massachusetts buildings dataset.

```bash
# preprocess training set
python tools/mass_patch_split.py \
--input-img-dir "data/mass_build/png/train" \
--input-mask-dir "data/mass_build/png/train_labels" \
--output-img-dir "data/mass_build/png/train_images" \
--output-mask-dir "data/mass_build/png/train_masks" \
--mode "train"

# preprocess validation set
python tools/mass_patch_split.py \
--input-img-dir "data/mass_build/png/val" \
--input-mask-dir "data/mass_build/png/val_labels" \
--output-img-dir "data/mass_build/png/val_images" \
--output-mask-dir "data/mass_build/png/val_masks" \
--mode "val"

# preprocess testing set
python tools/mass_patch_split.py \
--input-img-dir "data/mass_build/png/test" \
--input-mask-dir "data/mass_build/png/test_labels" \
--output-img-dir "data/mass_build/png/test_images" \
--output-mask-dir "data/mass_build/png/test_masks" \
--mode "val"

# crop the images into 512x512

python tools/split_1500_to_512.py \
--path \
--save_path \
--mode \
```

**Final Folder Structure**
At last, the expected folder structure should be as follows:

