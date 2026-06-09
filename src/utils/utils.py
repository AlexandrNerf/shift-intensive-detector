import warnings
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Применяет дополнительные утилиты перед запуском задачи.

    Утилиты:
        - игнорирование предупреждений Python
        - установка тегов из командной строки
        - печать конфига через Rich

    :param cfg: объект DictConfig с деревом конфига.
    """
    # выходим, если конфиг `extras` отсутствует
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # отключаем предупреждения Python
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # запрашиваем теги из командной строки, если они не указаны в конфиге
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # красиво выводим дерево конфига через Rich
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Опциональный декоратор, управляющий поведением при ошибках во время выполнения задачи.

    Обертку можно использовать, чтобы:
        - закрывать логгеры даже при исключении внутри задачи;
        - сохранять исключение в `.log` файл;
        - помечать запуск как упавший отдельным файлом в `logs/`;
        - расширять поведение под нужды проекта.

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: функция задачи, которую нужно обернуть.

    :return: обернутая функция задачи.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # выполняем задачу
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # действия при исключении
        except Exception as ex:
            # сохраняем исключение в `.log` файл
            log.exception("")

            # некоторые сочетания гиперпараметров могут быть некорректными
            # или приводить к нехватке памяти; при HPO через Optuna и аналоги
            # иногда полезно не пробрасывать это исключение, чтобы не ронять multirun
            raise ex

        # действия, которые выполняются и после успеха, и после ошибки
        finally:
            # выводим путь к выходной директории в терминал
            log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Безопасно получает значение метрики, залогированной в LightningModule.

    :param metric_dict: словарь со значениями метрик.
    :param metric_name: имя метрики, которую нужно получить.
    :return: значение метрики, если имя было передано.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
