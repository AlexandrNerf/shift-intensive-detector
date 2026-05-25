from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# setup_root выше эквивалентен следующим действиям:
# - добавить корень проекта в PYTHONPATH
#       (поэтому не нужно заставлять пользователя устанавливать проект как пакет)
#       (это нужно до импорта локальных модулей, например `from src import utils`)
# - задать переменную окружения PROJECT_ROOT
#       (она используется как база для путей в "configs/paths/default.yaml")
#       (так пути одинаковы независимо от директории запуска)
# - загрузить переменные окружения из ".env" в корне проекта
#
# этот блок можно удалить, если:
# 1. установить проект как пакет или перенести entrypoint-файлы в корень проекта
# 2. выставить `root_dir` в "." в "configs/paths/default.yaml"
#
# подробнее: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Оценивает указанный чекпоинт на тестовом наборе датамодуля.

    Функция обернута декоратором @task_wrapper, который управляет поведением
    при ошибках. Это полезно для multirun и сохранения информации о падении.

    :param cfg: конфигурация DictConfig, собранная Hydra.
    :return: кортеж с метриками и словарем всех созданных объектов.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # для предсказаний используйте trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Основная точка входа для оценки.

    :param cfg: конфигурация DictConfig, собранная Hydra.
    """
    # применяем дополнительные утилиты
    # например, запрашиваем теги или печатаем дерево конфига
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
