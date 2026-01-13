<div align="center">

# ШИФТ Интенсив - Детекция


Репозиторий для обучения моделей детекции

</div>

## Структура репозитория

## Структура

Вид основной директории и её содержание:

```
├── configs                 <- Hydra конфиги
│   ├── callbacks                <- Callbacks
│   ├── data                     <- Датасеты и даталоадеры
│   ├── debug                    <- Отладка
│   ├── experiment               <- Конфиги для экспериментов
│   ├── extras                   <- Дополнительные фичи
│   ├── hparams_search           <- Поиск гиперпарамов (optuna)
│   ├── hydra                    <- Hydra доп. конфиги
│   ├── local                    <- Конфиги для локального доступа
│   ├── logger                   <- Логирование
│   ├── model                    <- Модели
│   ├── paths                    <- Конфиг с путями
│   ├── trainer                  <- Параметры обучения
│   │
│   ├── eval.yaml             <- Конфиг валидации и теста
│   └── train.yaml            <- Конфиг для обучения
│
├── logs                   <- Логи (появятся при запуске экспериментов)
│
├── notebooks              <- Тетрадки с полезными функциями
│
├── src                    <- Ядро моделинга
│   ├── data                     <- Данные
│   ├── models                   <- Модели
│   ├── utils                    <- Доп. утилиты (логирование, вывод через rich)
│   │
│   ├── eval.py                  <- Валидация
│   └── train.py                 <- Обучение
│
└── README.md
```



## Подготовка окружения

Для установки окружения потребуется conda и установка poetry

```
conda create -n det-env python=3.10.16
conda activate det-env

pip install poetry==2.2.1
poetry install
```

## Запуск скриптов

Теперь для запуска из главной директории:

```bash
python src/train.py
```

Для валидации:

```bash
python src/eval.py --logs/train/<your_time_train>/checkpoints/epoch_XXX.ckpt
```

<br>



## ⚡ Возможности

<details>
<summary><b>Изменение конфига в консоли</b></summary>

```bash
python train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

> **Note**: Также можно добавлять параметры через `+`.

```bash
python train.py +model.new_param="owo"
```

</details>

<details>
<summary><b>Обучение на GPU, CPU и даже DDP</b></summary>

```bash
# train on CPU
python train.py trainer=cpu

# train on 1 GPU
python train.py trainer=gpu

# train on TPU
python train.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer=ddp trainer.devices=4

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

# simulate DDP on CPU processes
python train.py trainer=ddp_sim trainer.devices=2

# accelerate training on mac
python train.py trainer=mps
```


</details>

<details>
<summary><b>Встроенный mixed precision</b></summary>

```bash
# train with pytorch native automatic mixed precision (AMP)
python train.py trainer=gpu +trainer.precision=16
```

</details>


<details>
<summary><b>Поддержка всех популярных логгеров</b></summary>

```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# train model with Weights&Biases (link to wandb dashboard should appear in the terminal)
python train.py logger=wandb
```

> **Note**: Немного информации о трекинге от авторов Lightning [here](#experiment-tracking).

> **Note**: Для wandb - [setup account](https://www.wandb.com/).

> **Note**: [Здесь](https://wandb.ai/hobglob/template-dashboard/) пример логирования через wandb


</details>

<details>
<summary><b>Эксперименты</b></summary>

```bash
python train.py experiment=example
```

> **Note**: Эксперименты в [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Callback по желанию</b></summary>

```bash
python train.py callbacks=default
```

> **Note**: Подробнее о настройке сохранения, ранней остановки и др [здесь](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).

> **Note**: Callbacks configs are placed in [configs/callbacks/](configs/callbacks/).

</details>

<details>
<summary><b>Фишки Lightning</b></summary>

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python train.py +trainer.gradient_clip_val=0.5

# run validation loop 4 times during a training epoch
python train.py +trainer.val_check_interval=0.25

# accumulate gradients
python train.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python train.py +trainer.max_time="00:12:00:00"
```

> **Note**: Немного о полезных фишках: [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

</details>

<details>
<summary><b>Простая отладка</b></summary>

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enforces debug-friendly configuration
python train.py debug=default

# run 1 train, val and test loop, using only 1 batch
python train.py debug=fdr

# print execution time profiling
python train.py debug=profiler

# try overfitting to 1 batch
python train.py debug=overfit

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true

# use only 20% of the data
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **Note**: В [configs/debug/](configs/debug/) лежат настройки отладки

</details>

<details>
<summary><b>Продолжение обучения</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Принимается путь или url.

> **Note**: Логирование начинается заново

</details>

<details>
<summary><b>Валидация чекпоинта</b></summary>

```yaml
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

</details>

<details>
<summary><b>Сетка гиперпараметров</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m data.batch_size=32,64,128 model.lr=0.001,0.0005
```

> **Note**: Hydra обрабатывает конфиги "лениво", поэтому при запуске новой работы лучше до этого конфиги не трогать

</details>

<details>
<summary><b>Сетка гиперпараметров с Optuna</b></summary>

```bash
# this will run hyperparameter search defined in `configs/hparams_search/mnist_optuna.yaml`
# over chosen experiment config
python train.py -m hparams_search=mnist_optuna experiment=example
```

> **Note**: [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) запускается через [свой конфиг](configs/hparams_search/mnist_optuna.yaml).

> **Warning**: При ошибке одной работы последующие тоже завершаются

</details>

<details>
<summary><b>Тэги для экспериментов</b></summary>

Для обозначения запусков:

```bash
python train.py tags=["mnist","experiment_X"]
```

> **Note**: Для форматирования: `python train.py tags=\["mnist","experiment_X"\]`.

Если нет тегов:

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - No tags provided in config. Prompting user to input tags...
Enter a list of comma separated tags (dev):
```

Теги обязательны для мультирана

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: Specify tags before launching a multirun!
```

</details>

<br>
