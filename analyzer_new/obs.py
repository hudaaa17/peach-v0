import logging

TRACE_EDGES = False
def get_logger(name):
    return logging.getLogger(name)

def log(logger, level, msg, **kw):
    logger.log(getattr(logging, level.upper()), "%s %s", msg, kw)