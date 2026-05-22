# Dataset Preparation

## View-of-Delft (VoD)

Download the official View-of-Delft dataset from:

- https://github.com/tudelft-iv/view-of-delft-dataset

We additionally generate:

- foreground masks
- annotation masks
- radar depth supervision files

### Preparation

```bash
ln -s /your_path/view_of_delft_PUBLIC/ ./data/VoD

# Generate foreground masks and annotations
python tools/gen_panoptic_seg_vod.py

# Convert png masks to npy format
python tools/png2npy_vod.py

# Generate VoD radar metadata (5-frame accumulation)
python tools/create_data_VODradar.py

# Generate VoD lidar metadata
python tools/create_data_VODlidar.py
```

### Folder Structure

The dataset should be organized as follows:

```text
VoD
├── lidar
│   ├── ImageSets
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   └── trainval.txt
│   │
│   ├── training
│   │   ├── annotations
│   │   ├── calib
│   │   ├── image_2
│   │   ├── label_2
│   │   ├── masks
│   │   ├── pose
│   │   ├── velodyne
│   │   └── velodyne_reduced
│   │
│   ├── testing
│   │   ├── calib
│   │   ├── image_2
│   │   └── velodyne
│   │
│   ├── vod_infos_train.pkl
│   ├── vod_infos_val.pkl
│   ├── vod_infos_test.pkl
│   └── vod_infos_trainval.pkl
│
├── radar_5frames
│   ├── ImageSets -> ../lidar/ImageSets
│   │
│   ├── training
│   │   ├── calib
│   │   ├── depth_npy_predict
│   │   ├── image_2
│   │   ├── label_2
│   │   ├── pose
│   │   ├── velodyne
│   │   └── velodyne_reduced
│   │
│   ├── testing
│   │   ├── calib
│   │   ├── image_2
│   │   └── velodyne
│   │
│   ├── vod_infos_train.pkl
│   ├── vod_infos_val.pkl
│   ├── vod_infos_test.pkl
│   └── vod_infos_trainval.pkl
│
└── ...
```

### Additional Generated Files

| Folder | Description |
|---|---|
| `masks/` | foreground masks |
| `annotations/` | annotation masks |
| `depth_npy_predict/` | radar-guided depth supervision |

---

## TJ4DRadSet

Download the official TJ4DRadSet dataset from:

- https://github.com/TJRadarLab/TJ4DRadSet

Since the LiDAR version has not yet been publicly released, we use radar depth maps as depth supervision during training.

We additionally generate:

- foreground masks
- annotation masks
- radar depth supervision files

### Preparation

```bash
ln -s /your_path/TJ4DRadSet_4DRadar/ ./data/TJ4D

# Generate foreground masks and annotations
python tools/gen_panoptic_seg_TJ4D.py

# Convert png masks to npy format
python tools/png2npy_TJ4D.py

# Generate TJ4D radar metadata
python tools/create_data_TJ4Dradar.py
```

### Folder Structure

The dataset should be organized as follows:

```text
TJ4D
├── ImageSets
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── trainval.txt
│   └── readme.txt
│
├── training
│   ├── calib
│   │   ├── 000000.txt
│   │   └── ...
│   │
│   ├── image_2
│   │   ├── 000000.png
│   │   └── ...
│   │
│   ├── label_2
│   │   ├── 000000.txt
│   │   └── ...
│   │
│   ├── velodyne
│   │   ├── 000000.bin
│   │   └── ...
│   │
│   ├── velodyne_reduced
│   │
│   ├── depth_npy_predict
│   │
│   └── masks
│
├── annotations
│
├── masks
│
├── Video_Demo
│   ├── seq04.mp4
│   └── ...
│
├── TJ4D_infos_train.pkl
├── TJ4D_infos_val.pkl
└── TJ4D_infos_trainval.pkl
```

### Additional Generated Files

| Folder | Description |
|---|---|
| `masks/` | foreground masks |
| `annotations/` | annotation masks |
| `depth_npy_predict/` | radar-guided depth supervision |

---

## Notes

- `depth_npy_predict/` is required for radar depth supervision.
- `masks/` and `annotations/` are required for segmentation-guided training.
- All preprocessing scripts are located in the `tools/` directory.
- The generated `.pkl` info files are required before training.