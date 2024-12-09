# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle 

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

inputx = (5,121,72,23,112,26.2,0.245,30)
input_ = np.asarray(inputx)
input_ = input_.reshape(1, -1)
prediction = loaded_model.predict(input_)
if prediction[0]==0:
    print("Non-Diabetic")
else:
    print('Diabetic')
