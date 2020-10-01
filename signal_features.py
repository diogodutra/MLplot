"Library to extract features from time-series signals."

import numpy as np
from utils.ytransforms import SNR


class SignalFeatures(object):
    
    "Extract features from multiple multidimensional time-series sensors signals."
    
    def __init__(self, *, sensors=4, dimensions=3, delimiter='_'):
        
        self.dims = dimensions
        self.delimiter = delimiter
        
        self._intermediary_functions = {
            'signal': lambda x: x,
            'norm': lambda x: np.array([np.linalg.norm(
                x[self._indices(s)], axis=0) for s in range(self.sensors)]),
            'elev': lambda x: np.array([np.arctan2(
                x[self._indices(s)[1]], x[self._indices(s)[0]]) for s in range(self.sensors)]),
            'azim': lambda x: np.array([np.arctan2(
                x[self._indices(s)[2]], x[self._indices(s)[0]]) for s in range(self.sensors)]),
            'pulse': lambda x: x[np.argmax(SNR()(x))],
        }
        
        self._final_functions = {
            'ptp': lambda x: np.ptp(x, axis=1), # Peak-To-Peak
            'std': lambda x: np.std(x, axis=1), # Standard Deviation
        }
        
        
    def _indices(self, sensor):
        "Returns a subset of signal indices corresponding to the input sensor."
        i_ini = self.dims * sensor
        i_end = i_ini + self.dims
        return np.arange(i_ini, i_end)
        
        
    def __call__(self, signals):
        "Extract features from time-series signals."
        
        n_signals = signals.shape[0]
        
        # error handling
        if n_signals % self.dims > 0:
            raise ValueError(f'Input has {n_signals} signals but should have a multiple of dimensions {self.dims}')
        
        
        self.sensors = n_signals // self.dims
    
    
        # intermediary calculations that are used in multiple features
        self._intermediary_results = {}
        for label, fc in self._intermediary_functions.items():
            self._intermediary_results[label] = fc(signals)
        
               
        # extract features
        self._results = {}
        for label_final, fc_final in self._final_functions.items():
            for label_inter, result_inter in self._intermediary_results.items():
                self._results[label_inter + self.delimiter + label_final] = fc_final(result_inter)
                
                
        return self._results
    
    
    def flatten(self):
        "Converts multi-lengthed output to flattened."
        
        if not hasattr(self, '_results'):
            raise ValueError('Missing feature extraction. Run it before this method.')
            
        
        sf_raveled = {}
        for key, values in self._results.items():
            for i, value in enumerate(values):
                sf_raveled[key + '_' + str(i)] = value

        return sf_raveled