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

Готовый датасет и веса обученной модели доступны по ссылке, данной в cloud_server. Для создания собственной модели на новом датасете следуйте инструкциям ниже

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
В корневой директории автоматически создается папка model, в которую добавляются веса модели, полученные в процессе обучения, и папка tboard, в которую добавляются tfevents, соответствующие процессу обучения. 

На последней эпохе обучения распознавателя есть смысл задействовать дополнительные метрики для получения более полной информации (на порядок замедляет процесс обучения)

```
python tools/train_shadownet.py --decode_outputs 1
```

Ход тренировки модели можно контролировать с помощью tensorboard

```
tensorboard --logdir=папка_с_tfevents
```

### Подготовка датасета


### TO DO

- [ ] Переписать распознаватель под TF2, сложность в замене [tesorflow.contrib](https://github.com/IgorSondors/CRNN_Tensorflow_Ubuntu18/blob/cbaa4d5c789d3fa6d3f442209fc3b872acd07486/crnn_model/crnn_net.py#L10). Для этого заменить  rnn.stack_bidirectional_dynamic_rnn на [это](https://github.com/tensorflow/tensorflow/issues/33683)
- [ ] Использовать [tensorflow service](https://www.tensorflow.org/tfx/serving/serving_advanced) в качестве продакшн-сервера на замену Flask

