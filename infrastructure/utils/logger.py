from datetime import datetime
import os
import os.path as osp
import numpy as np
import pandas as pd
from pathlib import Path


class Logger:
    def __init__(self, directory=None, append=False):
        self.file_path = None
        self.keys = []

        self._init_dir(directory, append)

    def _init_dir(self, dirname, append):
        """
            Initializes the directory, if no directory was passed to the
            constructor a timestamped directory is created instead.
        """
        if dirname is None:
            current_timestamp = datetime.now()
            timestamp = current_timestamp.strftime('%Y-%m-%d-%H:%M:%S')
            dirname = "logs-" + timestamp + "/"

        self.filepath = Path(dirname) / "logs.csv"
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.touch()
        if not append:
            self.filepath.write_text("")

    def update_keys(self, new_keys):
        shoudl_update = False
        for key in new_keys:
            if key not in self.keys:
                shoudl_update = True
                self.keys.append(key)
    
        if shoudl_update:
            df = pd.read_csv(self.filepath, sep=';')
            for key in self.keys:
                if key not in df.columns:
                    df[key] = None
            df.to_csv(self.filepath, sep=';', index=False)

    def write(self, data, step):
        """
            Accepts a dictionary of key : scalar pairs as well as the current
            (training/evaluation) step.
        """
        data['step'] = step
        
        if not self.keys:
            self.keys = list(data.keys())
            df = pd.DataFrame([], columns=self.keys)
            df.to_csv(self.filepath, sep=';', index=False)
        
        self.update_keys(data.keys())
        
        row_df = pd.DataFrame([data], columns=self.keys)
        row_df.to_csv(self.filepath, sep=';', index=False, mode='a', header=False)

    def export_df(self):
        """
            Exposes the contents of the csv file as a pandas dataframe.
        """
        return pd.read_csv(self.filename, sep=';')

        

