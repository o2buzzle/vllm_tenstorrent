from logging.config import dictConfig

from vllm.logger import DEFAULT_LOGGING_CONFIG


def register():
    """
    Register the vllm_tenstorrent package.
    """
    return "vllm_tenstorrent.platform.TTPlatform"


def _init_logging():
    """Setup logging, extending from the vLLM logging config"""
    config = {**DEFAULT_LOGGING_CONFIG}

    # Copy the vLLM logging configurations
    config["formatters"]["vllm_tenstorrent"] = DEFAULT_LOGGING_CONFIG[
        "formatters"]["vllm"]

    handler_config = DEFAULT_LOGGING_CONFIG["handlers"]["vllm"]
    handler_config["formatter"] = "vllm_tenstorrent"
    config["handlers"]["vllm_tenstorrent"] = handler_config

    logger_config = DEFAULT_LOGGING_CONFIG["loggers"]["vllm"]
    logger_config["handlers"] = ["vllm_tenstorrent"]
    config["loggers"]["vllm_tenstorrent"] = logger_config

    dictConfig(config)


_init_logging()
