import logging
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from langfuse import Langfuse
from src.config import LangfuseSettings

logger = logging.getLogger(__name__)


class LangfuseTracer:
    def __init__(self, settings: LangfuseSettings) -> None:
        self.client: Langfuse | None = None

        if not settings.enabled:
            return

        if not settings.public_key or not settings.secret_key:
            logger.warning("Langfuse is enabled but its credentials are missing")
            return

        try:
            self.client = Langfuse(
                public_key=settings.public_key,
                secret_key=settings.secret_key,
                host=settings.host,
                flush_at=settings.flush_at,
                flush_interval=settings.flush_interval,
                debug=settings.debug,
            )
        except Exception:
            logger.exception("Failed to initialize Langfuse tracing")

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def span(
        self,
        name: str,
        input_data: object | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AbstractContextManager[Any]:
        if self.client is None:
            return nullcontext()

        try:
            return self.client.start_as_current_span(
                name=name,
                input=input_data,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Failed to create a Langfuse span")
            return nullcontext()

    def generation(
        self,
        name: str,
        model: str,
        input_data: object | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AbstractContextManager[Any]:
        if self.client is None:
            return nullcontext()

        try:
            return self.client.start_as_current_generation(
                name=name,
                model=model,
                input=input_data,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Failed to create a Langfuse generation")
            return nullcontext()

    @staticmethod
    def update(observation: Any, output_data: object) -> None:
        if observation is not None:
            try:
                observation.update(output=output_data)
            except Exception:
                logger.exception("Failed to update a Langfuse observation")

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                logger.exception("Failed to flush Langfuse events")

    def shutdown(self) -> None:
        if self.client is not None:
            try:
                self.client.shutdown()
            except Exception:
                logger.exception("Failed to shut down Langfuse")
