# 1p_BLA_sync_permutation_social_valence
DOI: https://doi.org/10.1101/2023.10.17.562740
https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true

```
"Step by step" for codes for BLA neuron calcium imaging analysis
```
1. Either using our demo files or your calcium imaging and behavior CSV files need to be used: Demo lists: 1. Figure4_food, 2. Figure5_social, 3. Figure5_fear_conditioning, 4. suppl.fig10_longitudinal_footshock_social.
2.  you need to download "core_codes" (logger.py, preprocessing.py, util.py) to synchronize the extracted behavior statistics with calcium traces (The TTL emission-reception delay is negligible (less than 30ms), therefore the behavioral statistics time series can be synchronized with calcium traces by the emission/receival time on both devices) and it would generate combined one H5 file (behavior+calcium data)
3.  Run Synchrnoize_h5generation.py code to apply core-codes (preprocessing) to your data: "import core.preprocessing as prep"
4. You can run each code (5 codes) described in the paper to generate the results: 1. Fear Conditioning, 2.social, 3. food, 4. permutation, 5, suppl.10 social_footshock.
   code number 4 is the percentage comparison with shuffling of data in Figure4-5 as bar graph you can reproduce using the code: Figure4_5_permutation_bargraph.
   
 [!NOTE]      
1. System requirements

Python (3.10.8): we used  a Python IDE for professional developers by JetBrains, Pycharm.

[!TIP]
- "our analysis pipeline is based on basic python packages:" 
import numpy as np
import h5py as h5
import pandas as pd

2. Installation guide
above 
3. Demo
Demo_data.zip
