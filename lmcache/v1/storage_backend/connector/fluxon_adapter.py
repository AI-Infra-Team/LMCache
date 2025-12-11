# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import ConnectorAdapter, ConnectorContext
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

# Standard
from urllib.parse import parse_qs, urlparse

logger = init_logger(__name__)


class FluxonConnectorAdapter(ConnectorAdapter):
    """Adapter for Fluxon connectors."""

    def __init__(self) -> None:
        # fluxon remote url example:
        #   fluxon:///path/to/fluxon_config.yaml
        super().__init__("fluxon://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local import to avoid importing fluxon dependencies unless needed
        from .fluxon_connector import FluxonConnector

        logger.info("Creating Fluxon connector for URL: %s", context.url)

        parsed = urlparse(context.url)

        logger.info("Fluxon url parsed: %s, config file path: %s", parsed, parsed.path)

        return FluxonConnector(
            config_path=parsed.path,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
        )
