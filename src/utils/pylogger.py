import logging
from typing import Mapping, Optional

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """Консольный Python-логгер, удобный для обучения на нескольких GPU."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Инициализирует консольный логгер для обучения на нескольких GPU.

        Логгер пишет из всех процессов, добавляя rank процесса в начало сообщения.

        :param name: имя логгера. По умолчанию ``__name__``.
        :param rank_zero_only: писать логи только из процесса с rank 0. По умолчанию `False`.
        :param extra: опциональный dict-like объект с контекстной информацией. См. `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: Optional[int] = None, *args, **kwargs) -> None:
        """Передает сообщение нижележащему логгеру, добавляя ранг процесса.

        Если передан `'rank'`, лог будет записан только из этого процесса.

        :param level: уровень логирования. Подробнее см. `logging.__init__.py`.
        :param msg: сообщение для логирования.
        :param rank: rank процесса, из которого нужно писать лог.
        :param args: дополнительные аргументы для нижележащей функции логирования.
        :param kwargs: дополнительные именованные аргументы для нижележащей функции логирования.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
