# Training
We trained three detection models while training tracking models was not needed. The three detection models we trained are
- **EdgeYOLO** - [GitHub](https://github.com/LSH9832/edgeyolo/blob/main/README_EN.md)
- **YOLOv8** - [GitHub](https://github.com/ultralytics/ultralytics)
- **CEASC** - [GitHub](https://github.com/Cuogeihong/CEASC.git)

---
### EdgeYOLO
Create new python virtual environment using conda
```bash
conda create --name <myenv>
conda activate <myenv>
``` 

#### Step 1: Setup local environment
```bash
cd training/edgeyolo
pip install -r requirements.txt
```

#### Step 2: Download Dataset
 - Follow the steps mentioned in the following repository to download the dataset of need: [vidrone-dataset-download](https://github.com/VisDrone/VisDrone-Dataset?tab=readme-ov-file)
 - To straight away download in coco format use the following link: https://github.com/LSH9832/edgeyolo/releases/download/v1.0.0/visdrone_coco.zip

 Note: For training we used Visdrone-DET for training detection and for tracking evalutation we used Visdrone-VID (VID and MOT are same)

 - Unzip the folder and place the downloaded folder in training folder
 - Update the path in `training/edgeyolo/params/dataset/visdrone.yaml` in the field **dataset_path** with current path of the Visdrone data path

#### Step 3: Update the training parameter
 - Open `training/edgeyolo/params/train/train_visdrone.yaml` and review the hyper-parameters according to requirements (read comments in yaml for more explanation) 

#### Step 4: Training
```bash
python train.py --cfg ./params/train/train_visdrone.yaml
```
---
### CEASC
Setting up CEASC is a tricky process. Since it uses mmdetection which brings in lot of complexities 
having a completely new python virtual env is highly encouraged.

```bash
conda create -n <myenv> python=3.7
conda activate <myenv>
```

#### Step 1: Setup
```bash
cd training/CEASC

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

python3 -m pip install openmim

mim install mmcv-full==1.5.1

python3 -m pip install nltk

python3 -m pip install -r requirements/albu.txt

cd Sparse_conv

python setup.py install

python3 -m pip install mmengine

conda install faiss-cpu -c pytorch

cd ..

pip install -r requirements/build.txt

pip install -v -e .

mkdir output_dyn_ret
```

#### Step 2: Update configs to dataset path
 - CEASC required data to be in coco format. The same data downloaded for EdgeYOLO(coco format) can be used here. 
 - Update the data path in `training/CEASC/configs/UAV/dynamic_gfl_res18_visdrone.py` in field of data_root in line 58

#### Step 3: Run Training 
```bash
mkdir output_ceasc_training

python tools/train.py configs/UAV/dynamic_gfl_res18_visdrone.py --work-dir output_ceasc_training
``` 
Note:
 In case of error about torchvision_0.12.json not found...
 Place file `training/utils/torchvision_0.12.json` in directory: `<venv_path>/lib/python3.7/site-packages/mmcv/model_zoo/`

---
### Yolov8

#### Step 1: Update the data path in `training/utils/visdrone.yaml`

#### Step 2: Install ultralytics using pip: 
```bash
pip install ultralytics
```
#### Step 3: Run training
```bash
cd training/yolov8

python train_yolov8.py
```

Go back to [main documentation](../readme.md)