from loguru import logger


def silence_plan2eplus_logging():
    logger.disable("plan2eplus")
