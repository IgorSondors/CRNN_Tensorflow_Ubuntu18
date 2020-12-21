### Виртуальное окружение

#### Pip

- Для использования чистого pip также потребуется установить CUDA+cudnn для запуска кода на GPU, для этого смотрите [официальную документацию](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). Также потребуется установить tensorflow-gpu с помощью строки кода ниже. Обучение можно проводить и на CPU, в этом случае tensorflow-gpu, CUDA и cdnn не требуются.
```
pip install tensorflow==1.15
```
```
pip install tensorflow-gpu==1.15
```
```
pip install easydict
```
```
pip install glog
```
```
pip install opencv-python
```
```
pip install Pillow
```
```
pip install python-Levenshtein
```
```
pip install tqdm
```
```
pip install wordninja
```

#### Pip + conda
```
conda install -c anaconda tensorflow-gpu=1.15
```
```
pip install easydict
```
```
pip install glog
```
```
pip install opencv-python
```
```
pip install Pillow
```
```
pip install python-Levenshtein
```
```
pip install tqdm
```
```
pip install wordninja
```

### Тренировка собственной модели распознавателя

#### Генерация tfrecords

```
cd CRNN_Tensorflow_Ubuntu18
```

```
python tools/write_tfrecords.py
```
В папке records_save корневой директории создаются tfrecords, в папке data создается директория char_dict, содержащая в себе char_dict.json и ord_map.json

#### Запуск обучения

```
cd CRNN_Tensorflow_Ubuntu18
```

```
python tools/train_shadownet.py
```
В корневой директории автоматически создается папка model, в которую добавляются веса модели, полученные в процессе обучения, и папка tboard, в которую добавляются tfevents, соответствующие процессу обучения. Ход тренировки модели можно контролировать с помощью tensorboard

```
tensorboard --logdir=папка_с_tfrecords
```

### Подготовка датасета

### 

### TODO

- [] TF2
- [] Add tensorflow service script

