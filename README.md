##  1p_BLA_sync_permutation_social_valence
DOI: https://doi.org/10.1101/2023.10.17.562740
# Genetically- and spatially-defined basolateral amygdala neurons control food consumption and social interaction
Highlights: 
1)	Classification of molecularly-defined glutamatergic neuron types in mouse BLA with distinct spatial expression patterns.
2)	BLALypd1 neurons are positive-valence neurons innately responding to food and promoting normal feeding. 
3)	BLAEtv1 neurons innately respond to aversive and social stimuli.
4)	 BLAEtv1 neurons promote fear learning and social interactions. 

![Screenshot](https://github.com/limserenahansol/1p_BLA_sync_permutation_social_valence/blob/main/graphical%20abstract_hansol.png)

## "Step by step" for codes for BLA neuron calcium imaging analysis

1. Either using our demo files or your calcium imaging and behavior CSV files need to be used:
   Demo lists: 1. Figure4_food, 2. Figure5_social, 3. Figure5_fear_conditioning, 4. suppl.fig10_longitudinal_footshock_social.
>[!NOTE]
>  Demo files are already preprocessed H5 files, so you can skip 2-3steps. 
2.  you need to download "core_codes" **(logger.py, preprocessing.py, util.py)** to synchronize the extracted behavior statistics with calcium traces (The TTL emission-reception delay is negligible (less than 30ms),
   therefore the behavioral statistics time series can be synchronized with calcium traces by the emission/receival time on both devices) and it would generate combined one H5 file (behavior+calcium data)
3.  Run **Synchrnoize_h5generation.py** code to apply core-codes (preprocessing) to your data:
```
import core.preprocessing as prep
```
4. You can run each code (5 codes) described in the paper to generate the results: **1. Fear Conditioning, 2.social, 3. food, 4. permutation, 5, suppl.10 social_footshock.py**
   code number 4 is the percentage comparison with shuffling of data in Figure4-5 as bar graph you can reproduce using the code: Figure4_5_permutation_bargraph.
   
### required 
>[!NOTE]
>-System requirements
>Python (3.10.8): we used  a Python IDE for professional developers by JetBrains, Pycharm.
>Packages:
>suite2p :
>https://github.com/MouseLand/suite2p
```
pip install git+https://github.com/MouseLand/suite2p.git
```
> matplotlib
```
pip install matplotlib
pip install PyQt5
```
>numpy
>[!CAUTION]
our code included : matplotlib Qt5Agg
An Error is happening because Google Colab and Jupyter run on virtual environments which do not support GUI outputs as you cannot open new windows through a browser.
Running it locally on a code editor(Spyder, or even IDLE) ensures that it can open a new window for the GUI to initialize.

>[!TIP]
>- "our analysis pipeline is based on basic python packages:"
```
>  import numpy as np
>import h5py as h5
>import pandas as pd
```

Installation guide
above 
- Demo
-Demo_data.zip


### Acknowledgement
Yue Zhang
