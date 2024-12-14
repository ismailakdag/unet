
### Introduction
This repository contains scripts for processing images and masks for training a model. It is essential to prepare your images and masks before training.

### Prerequisites
- Python 3.x
- Required libraries (to be specified in a requirements file)

### Image and Mask Preparation
#### Resizing Images and Masks
Before using the images and masks, they must be resized to the desired dimensions. Use the following script:

```bash
python scripts/resize.py
```

#### Converting to .npy Format
After resizing, convert the images and masks to `.npy` format using:

```bash
python scripts/convert_npy.py
```

### Training the Model
Once the images and masks are prepared, you can train the model using:

```bash
python train.py --images_folder <path_to_images> --masks_folder <path_to_masks> --checkpoints_folder <path_to_checkpoints>
```

### Additional Notes
- When resizing images and masks, you can change the target dimensions in the `resize.py` script. Look for the line:

```python
resize_dim = (224, 224)  # Target dimension for resizing
```

and modify the values as needed.

- Additionally, you will need to adjust the arguments in `train.py` to suit your training requirements. Make sure to specify the correct paths for images, masks, and checkpoints when running the training script.

### Turkish Version
#### Giriş
Bu depo, bir modeli eğitmek için görüntüleri ve maskeleri işlemek için betikler içerir. Eğitime başlamadan önce görüntülerinizi ve maskelerinizi hazırlamak önemlidir.

#### Ön Gereksinimler
- Python 3.x
- Gerekli kütüphaneler (bir gereksinimler dosyasında belirtilmelidir)

#### Görüntü ve Maske Hazırlığı
##### Görüntüleri ve Maskeleri Yeniden Boyutlandırma
Görüntülerin ve maskelerin istenen boyutlara yeniden boyutlandırılması gerekir. Aşağıdaki betiği kullanın:

```bash
python scripts/resize.py
```

##### npy Formatına Dönüştürme
Yeniden boyutlandırdıktan sonra, görüntüleri ve maskeleri `.npy` formatına dönüştürmek için:

```bash
python scripts/convert_npy.py
```
