
from os.path import join, isfile
import sys
import datetime

class Logger():
    def __init__(self, path_save, name_exp=""):
        self.path_save = path_save
        self.time_of_birth = str(datetime.datetime.now()).replace(' ', '_')
        self.path_file = join(path_save, name_exp + '_summary_' + self.time_of_birth + '.txt')

    def super_print(self, statement):
        sys.stdout.write(statement + '\n')
        sys.stdout.flush()
        if self.path_save:
            with open(self.path_file, 'a') as f:
                f.write(statement + '\n')
        return 0

def create_logger(cfg):
    return Logger(cfg.SAVE.LOGPATH, cfg.NAME)
