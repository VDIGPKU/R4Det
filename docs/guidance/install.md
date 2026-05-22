# Environment Setup

The codebase requires custom CUDA operators and is tested under the following environment:

| Component      | Version |
| -------------- | ------- |
| OS             | Ubuntu  |
| Python         | 3.8     |
| CUDA           | 11.8    |
| PyTorch        | 2.0.1   |
| torchvision    | 0.15.2  |
| mmcv-full      | 1.7.1   |
| mmdet          | 2.28.2  |
| mmsegmentation | 0.30.0  |

The provided `environment.yaml` contains the complete tested environment for both training and evaluation.

---

## 1. Create Conda Environment

```bash
conda create -n r4det python=3.8 -y
conda activate r4det
```

---

## 2. Install PyTorch

Install PyTorch corresponding to CUDA 11.8:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
--index-url https://download.pytorch.org/whl/cu118
```

---

## 3. Install OpenMMLab Dependencies

```bash
pip install -U openmim

mim install mmcv-full==1.7.1
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
```

---

## 4. Install Neighborhood Attention (NATTEN)

```bash
pip install natten==0.14.6+torch200cu118 \
-f https://shi-labs.com/natten/wheels
```

---

## 5. Install Additional Python Dependencies

```bash
pip install \
opencv-python \
kornia \
k3d \
wandb \
yapf==0.32.0 \
setuptools==60.2.0 \
numba==0.58.1 \
nuscenes-devkit \
shapely \
pyquaternion \
scipy \
tensorboard
```

---

## 6. Install Detectron2

The framework uses a customized Detectron2 branch for the 2D instance segmentation modules.

```bash
cd detr2
python setup.py develop
cd ..
```

---

## 7. Compile the Framework and CUDA Operators

### Install the Main Framework

```bash
python setup.py develop
```

### Compile 3D CUDA Operators

```bash
cd mmdet3d/ops/csrc
python setup.py build_ext --inplace
```

### Compile Deformable Attention Operators

```bash
cd ../deformattn
python setup.py build install
```

Return to the project root directory:

```bash
cd ../../../
```

---

## 8. Verification

After installation, verify that the framework is correctly installed:

```bash
python -c "import mmdet3d; print(mmdet3d.__version__)"
```

If no errors occur, the environment is ready for training and evaluation.
