# ШИФТ Интенсив - Детекция CS2

Учебный репозиторий для одной задачи: детекция террористов и контр-террористов на изображениях из Counter-Strike 2.

Проект показывает полный минимальный пайплайн обучения детектора: подготовка датасета, настройка Hydra-конфигов, обучение модели, логирование и оценка чекпоинта.

## Структура проекта

```text
├── configs                         <- Hydra-конфиги
│   ├── callbacks                   <- чекпоинты, early stopping, progress bar
│   ├── data                        <- датамодуль и аугментации
│   ├── debug                       <- отладочные режимы запуска
│   ├── experiment                  <- готовые конфиги экспериментов
│   ├── logger                      <- TensorBoard и CSV логгеры
│   ├── model                       <- модель, backbone, optimizer, scheduler
│   ├── paths                       <- пути к данным, логам и рабочей директории
│   ├── trainer                     <- CPU, GPU и DDP настройки Trainer
│   ├── eval.yaml                   <- основной конфиг оценки
│   └── train.yaml                  <- основной конфиг обучения
│
├── data                            <- изображения CS2 и разметка
│   ├── train
│   │   ├── images                  <- обучающие изображения .jpg
│   │   └── labels                  <- обучающая разметка .txt
│   └── val
│       ├── images                  <- валидационные изображения .jpg
│       └── labels                  <- валидационная разметка .txt
│
├── scripts                         <- вспомогательные скрипты
├── src                             <- код обучения и модели
│   ├── data                        <- датасет и LightningDataModule
│   ├── lit_modules                 <- scheduler'ы
│   ├── models                      <- LightningModule и компоненты детектора
│   ├── utils                       <- утилиты Hydra/Lightning
│   ├── eval.py                     <- запуск оценки чекпоинта
│   └── train.py                    <- запуск обучения
│
├── pyproject.toml                  <- зависимости проекта
└── README.md
```

## Установка

Нужны Python 3.12+ и Poetry.

```bash
conda create -n det-env python=3.12
conda activate det-env

pip install poetry==2.2.1
poetry install --no-root
```

## Данные

Данные должны лежать в `data/`:

```text
data/
├── train
│   ├── images
│   └── labels
└── val
    ├── images
    └── labels
```

Для каждого изображения `*.jpg` нужен файл разметки `*.txt` с таким же именем. Формат разметки - YOLO: одна строка на каждый найденный объект, где `class_id` соответствует одному из классов датасета CS2.

```text
class_id x_center y_center width height
```

Координаты должны быть нормализованы в диапазоне от `0` до `1`.


### Скачивание

Чтобы скачать данные google диска можно использовать команду:

```
gdown 'https://drive.google.com/file/d/1BVGnAGsvmTzE3VXwjRB4lAYveR_ZVz9C/view?usp=sharing' -O /data/
```


## Обучение

Запуск с настройками по умолчанию:

```bash
poetry run python src/train.py
```

Запуск baseline-эксперимента:

```bash
poetry run python src/train.py experiment=baseline
```

Переопределение параметров из консоли:

```bash
poetry run python src/train.py data.batch_size=16 trainer.max_epochs=20 model.optimizer.lr=1e-4
```

Запуск на CPU:

```bash
poetry run python src/train.py trainer=cpu
```

Продолжение обучения из чекпоинта:

```bash
poetry run python src/train.py ckpt_path=/path/to/checkpoint.ckpt
```

Чекпоинты и логи сохраняются в `logs/train/<experiment_name>/runs/<date_time>/`.

## Оценка

```bash
poetry run python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

## Логирование

По умолчанию используется TensorBoard:

```bash
poetry run tensorboard --logdir logs
```

Доступные логгеры:

```bash
poetry run python src/train.py logger=tensorboard
poetry run python src/train.py logger=csv
poetry run python src/train.py logger=many_loggers
```

## Полезные Hydra-команды

Отключить обучение и оставить только тест после инициализации:

```bash
poetry run python src/train.py train=false test=true
```

Отладочный прогон:

```bash
poetry run python src/train.py debug=default
poetry run python src/train.py debug=fdr
poetry run python src/train.py debug=overfit
```

Мультиран:

```bash
poetry run python src/train.py -m data.batch_size=8,16 model.optimizer.lr=1e-3,1e-4
```
