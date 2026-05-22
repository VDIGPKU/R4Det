TRAINING_OLD_VERSION = False
def use_old_version():
    global TRAINING_OLD_VERSION
    TRAINING_OLD_VERSION = True

def get_old_version():
    global TRAINING_OLD_VERSION
    return TRAINING_OLD_VERSION