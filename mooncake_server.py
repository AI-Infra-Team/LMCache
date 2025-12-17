#!/usr/bin/env python3

import logging
from mooncake.store import MooncakeDistributedStore

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_server():
    """启动 Mooncake Distributed Store Server"""

    store = MooncakeDistributedStore()

    logger.info("Starting Mooncake Distributed Store server...")

    store.setup(
        local_hostname="192.168.0.217",
        # etcd address
        metadata_server="192.168.0.207:32579",
        global_segment_size=32 * 1024 * 1024 * 1024,
        local_buffer_size=4 * 1024 * 1024 * 1024,
        protocol="tcp",
        rdma_devices="",
        master_server_addr="127.0.0.1:50051"
    )

    try:
        # Server 常驻
        logger.info("Server is running. Press Ctrl+C to stop.")
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        store.close()
        logger.info("Server stopped.")


if __name__ == "__main__":
    run_server()