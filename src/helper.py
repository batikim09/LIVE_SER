
def log(log, logger = None):
    if logger is not None:
        logger.write(log + "\n")
    
    print(log)

