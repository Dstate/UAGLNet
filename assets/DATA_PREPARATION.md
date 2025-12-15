### Data Preprocessing

We follow [BuildingExtraction](https://github.com/stdcoutzrh/BuildingExtraction) to preprocess all the datasets. Please download the [Inria](https://project.inria.fr/aerialimagelabeling/), [WHU](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html), and [Massachusetts](https://www.cs.toronto.edu/~vmnih/data/) datasets, and organize them as follows.

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
