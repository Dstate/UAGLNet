# UAGLNet

## Introduction



## Quick Start

### Installation

Clone this repository and create the environment.
```bash
git git@github.com:Dstate/UAGLNet.git
conda create -n uaglnet python=3.8 -y
conda activate uaglnet

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

### Training

Download the pretrained backbone from [link](https://drive.google.com/drive/folders/1_kkl7wh5oE6hOynJmjRonx6CyulcuHlM), and move it to `geoseg/models`
```bash
mv backbone.pth geoseg/models/
```





## Acknowledgement
This work is built upon [BuildingExtraction](https://github.com/stdcoutzrh/BuildingExtraction) and [GeoSeg](https://github.com/WangLibo1995/GeoSeg/tree/main). We sincerely appreciate their contributions which provide a clear pipeline and well-organized code.


<!-- ## Citation
If the paper is helpful with your research, please cite it as:
```
@inproceedings{liu-niu2025lbp,
  title     = {},
  author    = {},
  booktitle = {Proceedings of the International Conference on Machine Learning},
  year      = {}
}
``` -->