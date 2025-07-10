# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:14:45 2024

@author: ruiui
"""

import matplotlib.pyplot as plt
import numpy as np

import os
import csv
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
from collections import defaultdict
from statistics import mean, median, stdev
import matplotlib.patches as patches

Vals = [(0.894, 'USD_AA_0'), (0.894, 'USD_AA_1'), (0.894, 'USD_AA_2'), (0.894, 'USD_AA_3'), 
        (0.812, 'USD_AB_0'), (0.812, 'USD_AB_1'), (0.812, 'USD_AB_2'), (0.812, 'USD_AB_3'), 
        (0.731, 'USD_AC_0'), (0.731, 'USD_AC_1'), (0.731, 'USD_AC_2'), (0.731, 'USD_AC_3'), 
        (0.756, 'USD_AE_0'), (0.756, 'USD_AE_1'), (0.756, 'USD_AE_2'), (0.756, 'USD_AE_3'), 
        (0.594, 'USD_AF_0'), (0.594, 'USD_AF_1'), (0.594, 'USD_AF_2'), (0.594, 'USD_AF_3'), 
        (0.562, 'USD_AH_0'), (0.562, 'USD_AH_1'), (0.562, 'USD_AH_2'), (0.562, 'USD_AH_3'), 
        (0.450, 'USD_AI_0'), (0.450, 'USD_AI_1'), (0.450, 'USD_AI_2'), (0.450, 'USD_AI_3'),
        (0.363, 'USD_AJ_0'), (0.363, 'USD_AJ_1'), (0.363, 'USD_AJ_2'), (0.363, 'USD_AJ_3'), 
        (0.788, 'USD_AK_0'), (0.788, 'USD_AK_1'), (0.788, 'USD_AK_2'), (0.788, 'USD_AK_3'), 
        (0.681, 'USD_AL_0'), (0.681, 'USD_AL_1'), (0.681, 'USD_AL_2'), (0.681, 'USD_AL_3'), 
        (0.625, 'USD_AM_0'), (0.625, 'USD_AM_1'), 
        (0.788, 'USD_AN_0'), (0.788, 'USD_AN_1'), 
        (0.788, 'USD_AN_2'), (0.788, 'USD_AN_3'), 
        (0.625, 'USD_AO_0'), (0.625, 'USD_AO_1'), 
        (0.625, 'USD_AO_2'), (0.625, 'USD_AO_3'), 
        (0.462, 'USD_AP_0'), (0.462, 'USD_AP_1'), 
        (0.462, 'USD_AP_2'), (0.462, 'USD_AP_3'), 
        (0.562, 'USD_AR_0'), (0.562, 'USD_AR_1'),
        (0.562, 'USD_AR_2'), (0.562, 'USD_AR_3'), 
        (0.337, 'USD_AS_0'), (0.337, 'USD_AS_1'), (0.337, 'USD_AS_2'), (0.337, 'USD_AS_3'), 
        (0.575, 'USD_AW_0'), 
        (0.575, 'USD_AX_0'), 
        (0.788, 'USD_AZ_0'), (0.788, 'USD_AZ_1'), (0.788, 'USD_AZ_2'), (0.788, 'USD_AZ_3'), 
        (0.625, 'USD_BA_0'), (0.625, 'USD_BA_1'), (0.625, 'USD_BA_2'), (0.625, 'USD_BA_3'), 
        (0.462, 'USD_BB_0'), (0.462, 'USD_BB_1'), (0.462, 'USD_BB_2'), (0.462, 'USD_BB_3'),
        (0.562, 'USD_BD_0'), (0.562, 'USD_BD_1'), (0.562, 'USD_BD_2'), (0.562, 'USD_BD_3'),
        (0.337, 'USD_BE_0'), (0.337, 'USD_BE_1'), (0.337, 'USD_BE_2'), (0.337, 'USD_BE_3'), 
        (0.400, 'USD_BG_0'), (0.400, 'USD_BG_1'), (0.400, 'USD_BG_2'), (0.400, 'USD_BG_3'), 
        (0.788, 'USD_BK_0'), (0.788, 'USD_BK_1'), 
        (0.625, 'USD_BL_0'), (0.625, 'USD_BL_1'),
        (0.462, 'USD_BM_0'), (0.462, 'USD_BM_1'),
        (0.512, 'USD_BO_0'), (0.512, 'USD_BO_1'),
        (0.675, 'USD_BR_0'), (0.675, 'USD_BR_1'), (0.675, 'USD_BR_2'), (0.675, 'USD_BR_3'),
        (0.538, 'USD_BS_0'), (0.538, 'USD_BS_1'), (0.538, 'USD_BS_2'), (0.538, 'USD_BS_3'), 
        (0.344, 'USD_BU_0'), (0.344, 'USD_BU_1'), (0.344, 'USD_BU_2'), (0.344, 'USD_BU_3'),
        (0.425, 'USD_BV_0'), (0.425, 'USD_BV_1'), (0.425, 'USD_BV_2'), (0.425, 'USD_BV_3'), 
        (0.338, 'USD_BW_0'), (0.338, 'USD_BW_1'), (0.338, 'USD_BW_2'), (0.338, 'USD_BW_3'), 
        (0.562, 'USD_BY_0'), (0.562, 'USD_BY_1'), (0.562, 'USD_BY_2'), (0.562, 'USD_BY_3'), 
        (0.538, 'USD_CA_0'), (0.538, 'USD_CA_1'), (0.538, 'USD_CA_2'), (0.538, 'USD_CA_3'), 
        (0.456, 'USD_CB_0'), (0.456, 'USD_CB_1'), (0.456, 'USD_CB_2'), (0.456, 'USD_CB_3'), 
        (0.338, 'USD_CC_0'), (0.338, 'USD_CC_1'),
        
        (0.869, 'US_AA_0'), (0.869, 'US_AA_1'), (0.869, 'US_AA_2'), (0.869, 'US_AA_3'), 
        (0.788, 'US_AB_0'), (0.788, 'US_AB_1'), (0.788, 'US_AB_2'), (0.788, 'US_AB_3'), 
        (0.706, 'US_AC_0'), (0.706, 'US_AC_1'), (0.706, 'US_AC_2'), (0.706, 'US_AC_3'),
        (0.675, 'US_AD_0'), (0.675, 'US_AD_1'), (0.675, 'US_AD_2'), (0.675, 'US_AD_3'),
        (0.706, 'US_AE_0'), (0.706, 'US_AE_1'), (0.706, 'US_AE_2'), (0.706, 'US_AE_3'), 
        (0.544, 'US_AF_0'), (0.544, 'US_AF_1'), (0.544, 'US_AF_2'), (0.544, 'US_AF_3'), 
        (0.481, 'US_AG_0'), (0.481, 'US_AG_1'), (0.481, 'US_AG_2'), (0.481, 'US_AG_3'), 
        (0.487, 'US_AH_0'), (0.487, 'US_AH_1'), (0.487, 'US_AH_2'), (0.487, 'US_AH_3'), 
        (0.425, 'US_AI_0'), (0.425, 'US_AI_1'), (0.425, 'US_AI_2'), (0.425, 'US_AI_3'), 
        (0.312, 'US_AJ_0'), (0.312, 'US_AJ_1'), (0.312, 'US_AJ_2'), (0.312, 'US_AJ_3'),
        (0.712, 'US_AK_0'), (0.712, 'US_AK_1'), (0.712, 'US_AK_2'), (0.712, 'US_AK_3'),
        (0.556, 'US_AL_0'), (0.556, 'US_AL_1'), (0.556, 'US_AL_2'), (0.556, 'US_AL_3'),
        (0.525, 'US_AM_0'), (0.525, 'US_AM_1'),
        (0.738, 'US_AN_0'), (0.738, 'US_AN_1'), (0.738, 'US_AN_2'), (0.738, 'US_AN_3'),
        (0.575, 'US_AO_0'), (0.575, 'US_AO_1'), (0.575, 'US_AO_2'), (0.575, 'US_AO_3'), 
        (0.412, 'US_AP_0'), (0.412, 'US_AP_1'), (0.412, 'US_AP_2'), (0.412, 'US_AP_3'), 
        (0.350, 'US_AQ_0'), (0.350, 'US_AQ_1'), 
        (0.512, 'US_AR_0'), (0.512, 'US_AR_1'), (0.512, 'US_AR_2'), (0.512, 'US_AR_3'), 
        (0.288, 'US_AS_0'), (0.288, 'US_AS_1'), (0.288, 'US_AS_2'), (0.288, 'US_AS_3'), 
        (0.225, 'US_AT_0'), 
        (0.288, 'US_AU_0'), (0.288, 'US_AU_1'), (0.288, 'US_AU_2'), (0.288, 'US_AU_3'),
        (0.162, 'US_AV_0'), (0.162, 'US_AV_1'), (0.162, 'US_AV_2'), (0.162, 'US_AV_3'), 
        (0.475, 'US_AW_0'),
        (0.475, 'US_AX_0'), 
        (0.225, 'US_AY_0'), 
        (0.738, 'US_AZ_0'), (0.738, 'US_AZ_1'), (0.738, 'US_AZ_2'), (0.738, 'US_AZ_3'), 
        (0.575, 'US_BA_0'), (0.575, 'US_BA_1'), (0.575, 'US_BA_2'), (0.575, 'US_BA_3'), 
        (0.412, 'US_BB_0'), (0.412, 'US_BB_1'), (0.412, 'US_BB_2'), (0.412, 'US_BB_3'), 
        (0.350, 'US_BC_0'), (0.350, 'US_BC_1'),
        (0.512, 'US_BD_0'), (0.512, 'US_BD_1'), (0.512, 'US_BD_2'), (0.512, 'US_BD_3'), 
        (0.288, 'US_BE_0'), (0.288, 'US_BE_1'), (0.288, 'US_BE_2'), (0.288, 'US_BE_3'),
        (0.288, 'US_BF_0'), (0.288, 'US_BF_1'), (0.288, 'US_BF_2'), (0.288, 'US_BF_3'),
        (0.350, 'US_BG_0'), (0.350, 'US_BG_1'), (0.350, 'US_BG_2'), (0.350, 'US_BG_3'),
        (0.412, 'US_BH_0'), (0.412, 'US_BH_1'), (0.412, 'US_BH_2'), (0.412, 'US_BH_3'),
        (0.225, 'US_BI_0'), (0.225, 'US_BI_1'), (0.225, 'US_BI_2'), (0.225, 'US_BI_3'), 
        (0.288, 'US_BJ_0'), (0.288, 'US_BJ_1'), (0.288, 'US_BJ_2'), (0.288, 'US_BJ_3'), 
        (0.738, 'US_BK_0'), (0.738, 'US_BK_1'), 
        (0.575, 'US_BL_0'), (0.575, 'US_BL_1'), 
        (0.412, 'US_BM_0'), (0.412, 'US_BM_1'), 
        (0.288, 'US_BN_0'), (0.288, 'US_BN_1'), 
        (0.462, 'US_BO_0'), (0.462, 'US_BO_1'), 
        (0.288, 'US_BP_0'), (0.288, 'US_BP_1'), 
        (0.400, 'US_BQ_0'), (0.400, 'US_BQ_1'), 
        (0.650, 'US_BR_0'), (0.650, 'US_BR_1'), (0.650, 'US_BR_2'), (0.650, 'US_BR_3'),
        (0.512, 'US_BS_0'), (0.512, 'US_BS_1'), (0.512, 'US_BS_2'), (0.512, 'US_BS_3'),
        (0.450, 'US_BT_0'), (0.450, 'US_BT_1'), (0.450, 'US_BT_2'), (0.450, 'US_BT_3'),
        (0.319, 'US_BU_0'), (0.319, 'US_BU_1'), (0.319, 'US_BU_2'), (0.319, 'US_BU_3'),
        (0.375, 'US_BV_0'), (0.375, 'US_BV_1'), (0.375, 'US_BV_2'), (0.375, 'US_BV_3'), 
        (0.312, 'US_BW_0'), (0.312, 'US_BW_1'), (0.312, 'US_BW_2'), (0.312, 'US_BW_3'), 
        (0.250, 'US_BX_0'), (0.250, 'US_BX_1'), (0.250, 'US_BX_2'), (0.250, 'US_BX_3'), 
        (0.538, 'US_BY_0'), (0.538, 'US_BY_1'), (0.538, 'US_BY_2'), (0.538, 'US_BY_3'), 
        (0.288, 'US_BZ_0'), (0.288, 'US_BZ_1'), (0.288, 'US_BZ_2'), (0.288, 'US_BZ_3'),
        (0.438, 'US_CA_0'), (0.438, 'US_CA_1'), (0.438, 'US_CA_2'), (0.438, 'US_CA_3'), 
        (0.381, 'US_CB_0'), (0.381, 'US_CB_1'), (0.381, 'US_CB_2'), (0.381, 'US_CB_3'), 
        (0.288, 'US_CC_0'), (0.288, 'US_CC_1'),
        
        (0.944, 'U_AA_0'), (0.944, 'U_AA_1'), (0.944, 'U_AA_2'), (0.944, 'U_AA_3'), 
        (0.888, 'U_AB_0'), (0.888, 'U_AB_1'), (0.888, 'U_AB_2'), (0.888, 'U_AB_3'), 
        (0.831, 'U_AC_0'), (0.831, 'U_AC_1'), (0.831, 'U_AC_2'), (0.831, 'U_AC_3'), 
        (0.775, 'U_AD_0'), (0.775, 'U_AD_1'), (0.775, 'U_AD_2'), (0.775, 'U_AD_3'), 
        (0.831, 'U_AE_0'), (0.831, 'U_AE_1'), (0.831, 'U_AE_2'), (0.831, 'U_AE_3'), 
        (0.719, 'U_AF_0'), (0.719, 'U_AF_1'), (0.719, 'U_AF_2'), (0.719, 'U_AF_3'), 
        (0.606, 'U_AG_0'), (0.606, 'U_AG_1'), (0.606, 'U_AG_2'), (0.606, 'U_AG_3'), 
        (0.663, 'U_AH_0'), (0.663, 'U_AH_1'), (0.663, 'U_AH_2'), (0.663, 'U_AH_3'), 
        (0.550, 'U_AI_0'), (0.550, 'U_AI_1'), (0.550, 'U_AI_2'), (0.550, 'U_AI_3'),
        (0.438, 'U_AJ_0'), (0.438, 'U_AJ_1'), (0.438, 'U_AJ_2'), (0.438, 'U_AJ_3'),
        (0.888, 'U_AK_0'), (0.888, 'U_AK_1'), (0.888, 'U_AK_2'), (0.888, 'U_AK_3'), 
        (0.831, 'U_AL_0'), (0.831, 'U_AL_1'), (0.831, 'U_AL_2'), (0.831, 'U_AL_3'), 
        (0.775, 'U_AM_0'), (0.775, 'U_AM_1'),
        (0.888, 'U_AN_0'), (0.888, 'U_AN_1'), (0.888, 'U_AN_2'), (0.888, 'U_AN_3'),
        (0.775, 'U_AO_0'), (0.775, 'U_AO_1'), (0.775, 'U_AO_2'), (0.775, 'U_AO_3'),
        (0.663, 'U_AP_0'), (0.663, 'U_AP_1'), (0.663, 'U_AP_2'), (0.663, 'U_AP_3'),
        (0.550, 'U_AQ_0'), (0.550, 'U_AQ_1'), 
        (0.662, 'U_AR_0'), (0.662, 'U_AR_1'), (0.662, 'U_AR_2'), (0.662, 'U_AR_3'), 
        (0.438, 'U_AS_0'), (0.438, 'U_AS_1'), (0.438, 'U_AS_2'), (0.438, 'U_AS_3'), 
        (0.325, 'U_AT_0'), 
        (0.438, 'U_AU_0'), (0.438, 'U_AU_1'), (0.438, 'U_AU_2'), (0.438, 'U_AU_3'), 
        (0.213, 'U_AV_0'), (0.213, 'U_AV_1'), (0.213, 'U_AV_2'), (0.213, 'U_AV_3'), 
        (0.775, 'U_AW_0'), 
        (0.775, 'U_AX_0'), 
        (0.325, 'U_AY_0'), 
        (0.888, 'U_AZ_0'), (0.888, 'U_AZ_1'), (0.888, 'U_AZ_2'), (0.888, 'U_AZ_3'),
        (0.775, 'U_BA_0'), (0.775, 'U_BA_1'), (0.775, 'U_BA_2'), (0.775, 'U_BA_3'),
        (0.663, 'U_BB_0'), (0.663, 'U_BB_1'), (0.663, 'U_BB_2'), (0.663, 'U_BB_3'),
        (0.550, 'U_BC_0'), (0.550, 'U_BC_1'), 
        (0.662, 'U_BD_0'), (0.662, 'U_BD_1'), (0.662, 'U_BD_2'), (0.662, 'U_BD_3'), 
        (0.438, 'U_BE_0'), (0.438, 'U_BE_1'), (0.438, 'U_BE_2'), (0.438, 'U_BE_3'), 
        (0.438, 'U_BF_0'), (0.438, 'U_BF_1'), (0.438, 'U_BF_2'), (0.438, 'U_BF_3'), 
        (0.550, 'U_BG_0'), (0.550, 'U_BG_1'), (0.550, 'U_BG_2'), (0.550, 'U_BG_3'), 
        (0.663, 'U_BH_0'), (0.663, 'U_BH_1'), (0.663, 'U_BH_2'), (0.663, 'U_BH_3'),
        (0.325, 'U_BI_0'), (0.325, 'U_BI_1'), (0.325, 'U_BI_2'), (0.325, 'U_BI_3'),
        (0.438, 'U_BJ_0'), (0.438, 'U_BJ_1'), (0.438, 'U_BJ_2'), (0.438, 'U_BJ_3'),
        (0.888, 'U_BK_0'), (0.888, 'U_BK_1'),
        (0.775, 'U_BL_0'), (0.775, 'U_BL_1'),
        (0.663, 'U_BM_0'), (0.663, 'U_BM_1'), 
        (0.438, 'U_BN_0'), (0.438, 'U_BN_1'), 
        (0.663, 'U_BO_0'), (0.663, 'U_BO_1'), 
        (0.438, 'U_BP_0'), (0.438, 'U_BP_1'), 
        (0.550, 'U_BQ_0'), (0.550, 'U_BQ_1'), 
        (0.775, 'U_BR_0'), (0.775, 'U_BR_1'), (0.775, 'U_BR_2'), (0.775, 'U_BR_3'), 
        (0.663, 'U_BS_0'), (0.663, 'U_BS_1'), (0.663, 'U_BS_2'), (0.663, 'U_BS_3'), 
        (0.550, 'U_BT_0'), (0.550, 'U_BT_1'), (0.550, 'U_BT_2'), (0.550, 'U_BT_3'), 
        (0.494, 'U_BU_0'), (0.494, 'U_BU_1'), (0.494, 'U_BU_2'), (0.494, 'U_BU_3'),
        (0.550, 'U_BV_0'), (0.550, 'U_BV_1'), (0.550, 'U_BV_2'), (0.550, 'U_BV_3'), 
        (0.438, 'U_BW_0'), (0.438, 'U_BW_1'), (0.438, 'U_BW_2'), (0.438, 'U_BW_3'), 
        (0.325, 'U_BX_0'), (0.325, 'U_BX_1'), (0.325, 'U_BX_2'), (0.325, 'U_BX_3'), 
        (0.662, 'U_BY_0'), (0.662, 'U_BY_1'), (0.662, 'U_BY_2'), (0.662, 'U_BY_3'), 
        (0.438, 'U_BZ_0'), (0.438, 'U_BZ_1'), (0.438, 'U_BZ_2'), (0.438, 'U_BZ_3'), 
        (0.663, 'U_CA_0'), (0.663, 'U_CA_1'), (0.663, 'U_CA_2'), (0.663, 'U_CA_3'), 
        (0.606, 'U_CB_0'), (0.606, 'U_CB_1'), (0.606, 'U_CB_2'), (0.606, 'U_CB_3'), 
        (0.438, 'U_CC_0'), (0.438, 'U_CC_1'), 
        
        (0.938, 'Z_AA_0'), (0.938, 'Z_AA_1'), (0.938, 'Z_AA_2'), (0.938, 'Z_AA_3'),
        (0.875, 'Z_AB_0'), (0.875, 'Z_AB_1'), (0.875, 'Z_AB_2'), (0.875, 'Z_AB_3'),
        (0.812, 'Z_AC_0'), (0.812, 'Z_AC_1'), (0.812, 'Z_AC_2'), (0.812, 'Z_AC_3'),
        (0.750, 'Z_AD_0'), (0.750, 'Z_AD_1'), (0.750, 'Z_AD_2'), (0.750, 'Z_AD_3'),
        (0.812, 'Z_AE_0'), (0.812, 'Z_AE_1'), (0.812, 'Z_AE_2'), (0.812, 'Z_AE_3'), 
        (0.688, 'Z_AF_0'), (0.688, 'Z_AF_1'), (0.688, 'Z_AF_2'), (0.688, 'Z_AF_3'), 
        (0.562, 'Z_AG_0'), (0.562, 'Z_AG_1'), (0.562, 'Z_AG_2'), (0.562, 'Z_AG_3'), 
        (0.625, 'Z_AH_0'), (0.625, 'Z_AH_1'), (0.625, 'Z_AH_2'), (0.625, 'Z_AH_3'), 
        (0.500, 'Z_AI_0'), (0.500, 'Z_AI_1'), (0.500, 'Z_AI_2'), (0.500, 'Z_AI_3'),
        (0.375, 'Z_AJ_0'), (0.375, 'Z_AJ_1'), (0.375, 'Z_AJ_2'), (0.375, 'Z_AJ_3'),
        (0.875, 'Z_AK_0'), (0.875, 'Z_AK_1'), (0.875, 'Z_AK_2'), (0.875, 'Z_AK_3'),
        (0.812, 'Z_AL_0'), (0.812, 'Z_AL_1'), (0.812, 'Z_AL_2'), (0.812, 'Z_AL_3'),
        (0.750, 'Z_AM_0'), (0.750, 'Z_AM_1'),
        (0.875, 'Z_AN_0'), (0.875, 'Z_AN_1'), (0.875, 'Z_AN_2'), (0.875, 'Z_AN_3'), 
        (0.750, 'Z_AO_0'), (0.750, 'Z_AO_1'), (0.750, 'Z_AO_2'), (0.750, 'Z_AO_3'), 
        (0.625, 'Z_AP_0'), (0.625, 'Z_AP_1'), (0.625, 'Z_AP_2'), (0.625, 'Z_AP_3'), 
        (0.500, 'Z_AQ_0'), (0.500, 'Z_AQ_1'),
        (0.625, 'Z_AR_0'), (0.625, 'Z_AR_1'), (0.625, 'Z_AR_2'), (0.625, 'Z_AR_3'), 
        (0.375, 'Z_AS_0'), (0.375, 'Z_AS_1'), (0.375, 'Z_AS_2'), (0.375, 'Z_AS_3'),
        (0.250, 'Z_AT_0'),
        (0.375, 'Z_AU_0'), (0.375, 'Z_AU_1'), (0.375, 'Z_AU_2'), (0.375, 'Z_AU_3'), 
        (0.125, 'Z_AV_0'), (0.125, 'Z_AV_1'), (0.125, 'Z_AV_2'), (0.125, 'Z_AV_3'), 
        (0.750, 'Z_AW_0'), 
        (0.750, 'Z_AX_0'), 
        (0.250, 'Z_AY_0'),
        (0.875, 'Z_AZ_0'), (0.875, 'Z_AZ_1'), (0.875, 'Z_AZ_2'), (0.875, 'Z_AZ_3'), 
        (0.750, 'Z_BA_0'), (0.750, 'Z_BA_1'), (0.750, 'Z_BA_2'), (0.750, 'Z_BA_3'),
        (0.625, 'Z_BB_0'), (0.625, 'Z_BB_1'), (0.625, 'Z_BB_2'), (0.625, 'Z_BB_3'),
        (0.500, 'Z_BC_0'), (0.500, 'Z_BC_1'),
        (0.625, 'Z_BD_0'), (0.625, 'Z_BD_1'), (0.625, 'Z_BD_2'), (0.625, 'Z_BD_3'),
        (0.375, 'Z_BE_0'), (0.375, 'Z_BE_1'), (0.375, 'Z_BE_2'), (0.375, 'Z_BE_3'),
        (0.375, 'Z_BF_0'), (0.375, 'Z_BF_1'), (0.375, 'Z_BF_2'), (0.375, 'Z_BF_3'),
        (0.500, 'Z_BG_0'), (0.500, 'Z_BG_1'), (0.500, 'Z_BG_2'), (0.500, 'Z_BG_3'),
        (0.625, 'Z_BH_0'), (0.625, 'Z_BH_1'), (0.625, 'Z_BH_2'), (0.625, 'Z_BH_3'), 
        (0.250, 'Z_BI_0'), (0.250, 'Z_BI_1'), (0.250, 'Z_BI_2'), (0.250, 'Z_BI_3'),
        (0.375, 'Z_BJ_0'), (0.375, 'Z_BJ_1'), (0.375, 'Z_BJ_2'), (0.375, 'Z_BJ_3'), 
        (0.875, 'Z_BK_0'), (0.875, 'Z_BK_1'),
        (0.750, 'Z_BL_0'), (0.750, 'Z_BL_1'), 
        (0.625, 'Z_BM_0'), (0.625, 'Z_BM_1'),
        (0.375, 'Z_BN_0'), (0.375, 'Z_BN_1'),
        (0.625, 'Z_BO_0'), (0.625, 'Z_BO_1'),
        (0.375, 'Z_BP_0'), (0.375, 'Z_BP_1'),
        (0.500, 'Z_BQ_0'), (0.500, 'Z_BQ_1'),
        (0.750, 'Z_BR_0'), (0.750, 'Z_BR_1'), (0.750, 'Z_BR_2'), (0.750, 'Z_BR_3'), 
        (0.625, 'Z_BS_0'), (0.625, 'Z_BS_1'), (0.625, 'Z_BS_2'), (0.625, 'Z_BS_3'), 
        (0.500, 'Z_BT_0'), (0.500, 'Z_BT_1'), (0.500, 'Z_BT_2'), (0.500, 'Z_BT_3'), 
        (0.438, 'Z_BU_0'), (0.438, 'Z_BU_1'), (0.438, 'Z_BU_2'), (0.438, 'Z_BU_3'), 
        (0.500, 'Z_BV_0'), (0.500, 'Z_BV_1'), (0.500, 'Z_BV_2'), (0.500, 'Z_BV_3'), 
        (0.375, 'Z_BW_0'), (0.375, 'Z_BW_1'), (0.375, 'Z_BW_2'), (0.375, 'Z_BW_3'),
        (0.250, 'Z_BX_0'), (0.250, 'Z_BX_1'), (0.250, 'Z_BX_2'), (0.250, 'Z_BX_3'),
        (0.625, 'Z_BY_0'), (0.625, 'Z_BY_1'), (0.625, 'Z_BY_2'), (0.625, 'Z_BY_3'), 
        (0.375, 'Z_BZ_0'), (0.375, 'Z_BZ_1'), (0.375, 'Z_BZ_2'), (0.375, 'Z_BZ_3'), 
        (0.625, 'Z_CA_0'), (0.625, 'Z_CA_1'), (0.625, 'Z_CA_2'), (0.625, 'Z_CA_3'), 
        (0.562, 'Z_CB_0'), (0.562, 'Z_CB_1'), (0.562, 'Z_CB_2'), (0.562, 'Z_CB_3'), 
        (0.375, 'Z_CC_0'), (0.375, 'Z_CC_1')
        ]

W = 12.5*12.5* 16 * 0.1

csv_directory = "C:/Users/ruiui/Desktop/iteration data/z_4x4"
circuit_irr_directory = "C:/Users/ruiui/Desktop/iteration data/_Irr_teste/4x4"

t1 = defaultdict(list)
percentage_diffs_pos = defaultdict(list)
percentage_diffs_val_mpp = defaultdict(list)
percentage_diffs = defaultdict(list)
mpp_sums = defaultdict(float)

# Adjust the values by multiplying by 1000 and converting to integers
adjusted_vals = {name: int(val * 1000) for val, name in Vals}

# Load CSV for each circuit
circuit_dfs = {}
for circuit in ['PER_D', 'PER', 'L_D', 'L', 'R1_D', 'R1', 'R2R3_D', 'R2R3', 'TCT_D', 'TCT', 'SP_D', 'SP']:
    circuit_file_path = os.path.join(circuit_irr_directory, f'z_4x4_irr_{circuit}.csv')
    circuit_dfs[circuit] = pd.read_csv(circuit_file_path)

# Function to get Max Power (P) for a given number and circuit
def get_max_power_for_number(circuit, number):
    df = circuit_dfs.get(circuit)
    if df is not None:
        row = df[df['Irr'] == number]
        if len(row) == 1:
            return row.iloc[0]['MPP']
    return None

# Create a dictionary with the Max Power (P) values for each circuit
max_power_dict = defaultdict(dict)
for name, number in adjusted_vals.items():
    for circuit in circuit_dfs.keys():
        max_power = get_max_power_for_number(circuit, number)
        if max_power is not None:
            max_power_dict[circuit][name] = max_power
        else:
            print(f"Warning: No unique match found for {name} with iteration {number} in circuit {circuit}")

# Read CSV file function
def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path).values.tolist()
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

# Extract parameters from CSV data
def extract_parameters(data):
    try:
        V = [row[0] for row in data]
        I = [row[1] for row in data]
        P = [row[2] for row in data]
        Voc = next(V[i] for i in range(len(I)) if I[i] < 0.0005)
        Isc = max(I)
        MPP_index = P.index(max(P))
        Vmp = V[MPP_index]
        Imp = I[MPP_index]
        MPP = max(P)
        FF = (MPP / (Isc * Voc)) * 100
        Ef = (MPP / W) * 100
        return Voc, Isc, Vmp, Imp, MPP, FF, Ef
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        return None, None, None, None, None, None, None

# Perform calculations for each CSV file grouped by iteration
def calculations(files_by_iteration):
    for iteration, csv_files in files_by_iteration.items():
        for idx, csv_file in enumerate(csv_files):
            data = read_csv_file(csv_file)
            if not data:
                continue
            circuit_name = os.path.basename(csv_file).split('_data_iteration_')[0]
            params = extract_parameters(data)
            if None in params:
                continue
            Voc, Isc, Vmp, Imp, MPP, FF, Ef = params
            reference_mpp = max_power_dict[circuit_name].get(iteration, None)
            if reference_mpp is None:
                print(f"Warning: No reference MPP found for {circuit_name}")
                continue
            
            percentage_diff = ((MPP - reference_mpp) / reference_mpp)
            # percentage_diff = MPP - reference_mpp
            
            percentage_diffs[circuit_name].append(percentage_diff)
            mpp_sums[circuit_name] += percentage_diff
            t1[circuit_name].append((MPP, reference_mpp))
            if 0 < percentage_diff:
                percentage_diffs_pos[circuit_name].append((percentage_diff, iteration, circuit_name))
            if 0 < percentage_diff < 0.05:
                percentage_diff = 0
            percentage_diffs_val_mpp[circuit_name].append((percentage_diff, iteration, circuit_name))

# Create a dictionary to map circuit names to iterations
files_by_iteration = defaultdict(list)

# List CSV files
csv_files = [os.path.join(csv_directory, filename) for filename in os.listdir(csv_directory) if filename.endswith(".csv")]

# Extract circuit names and iterations from file names
for file in csv_files:
    file_name = os.path.basename(file)
    iteration_name = file_name.split('_data_iteration_')[1]
    iteration = iteration_name.split('.')[0]
    files_by_iteration[iteration].append(file)

# Perform calculations
calculations(files_by_iteration)

# Convert percentage_diffs_val_mpp to DataFrame
data_for_df = [
    {
        'Circuit': circuit,
        'Iteration': iteration,
        'PercentageDiff': value
    } for circuit_name, values in percentage_diffs_val_mpp.items() for value, iteration, circuit in values
]

percentage_diffs_df = pd.DataFrame(data_for_df)

# Plotting results
styles = {
    'PER_D': {'color': '#80C080', 'marker': '|', 'phase': -0.3, 'size': 2},
    'PER': {'color': '#008000', 'marker': '+', 'phase': 0.1, 'size': 2},
    
    'L_D': {'color': '#A1875C', 'marker': '+', 'phase': 0.1, 'size': 2},
    'L': {'color': '#654321', 'marker': '+', 'phase': 0.1, 'size': 2},
    
    'R1_D': {'color': '#FFD580', 'marker': '+', 'phase': 0.1, 'size': 2},
    'R1': {'color': '#FFA500', 'marker': '+', 'phase': 0.1, 'size': 2},
    
    'R2R3_D': {'color': '#C080C0', 'marker': '+', 'phase': 0.1, 'size': 2},
    'R2R3': {'color': '#800080', 'marker': '+', 'phase': 0.1, 'size': 2},
    
    'TCT_D': {'color': '#8080FF', 'marker': 'o', 'phase': -0.2, 'size': 2},
    'TCT': {'color': '#0000FF', 'marker': 's', 'phase': -0.1, 'size': 2},
    
    'SP_D': {'color': '#808080', 'marker': '_', 'phase': 0.3, 'size': 2},
    'SP': {'color': '#000000', 'marker': 'D', 'phase': 0.2, 'size': 2}
}

# Convert Vals to a dictionary
Vals_dict = {value: key for key, value in Vals}

# Map the iteration values to the DataFrame
percentage_diffs_df['IterationValue'] = percentage_diffs_df['Iteration'].map(Vals_dict)

# Create custom order list from dictionary keys
iteration_order = list(Vals_dict.keys())

# Add a temporary column for the custom iteration order
percentage_diffs_df['IterationOrder'] = percentage_diffs_df['Iteration'].apply(lambda x: iteration_order.index(x) if x in iteration_order else -1)

# Reset the index
percentage_diffs_df.reset_index(drop=True, inplace=True)

# Drop the temporary column
percentage_diffs_df.drop(columns=['IterationOrder'], inplace=True)

# Extract the iteration type (C_A, C_B, etc.)
percentage_diffs_df['IterationType'] = percentage_diffs_df['Iteration'].str.extract(r'([A-Z]+_[A-Z]+)_')

# Sort the DataFrame first by IterationValue in descending order, then by the custom IterationOrder
percentage_diffs_df = percentage_diffs_df.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

median_values = percentage_diffs_df.groupby(['Circuit', 'IterationType']).agg({
    'PercentageDiff': 'median',
    'IterationValue': 'first'
}).reset_index()

median_values = median_values.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

max_percentages = percentage_diffs_df.groupby(['Circuit', 'IterationType'])['PercentageDiff'].idxmax()

result_df_max = percentage_diffs_df.loc[max_percentages].reset_index(drop=True)
result_df_max = result_df_max.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

min_percentages = percentage_diffs_df.groupby(['Circuit', 'IterationType'])['PercentageDiff'].idxmin()

result_df_min = percentage_diffs_df.loc[min_percentages].reset_index(drop=True)
result_df_min = result_df_min.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

wiwu_U = result_df_min[result_df_min['Iteration'].str.startswith('U')]
wiwu_Z = result_df_min[result_df_min['Iteration'].str.startswith('Z')]

data = defaultdict(list)# Convert Vals to a dictionary
Vals_dict = {value: key for key, value in Vals}

# Map the iteration values to the DataFrame
percentage_diffs_df['IterationValue'] = percentage_diffs_df['Iteration'].map(Vals_dict)

# Create custom order list from dictionary keys
iteration_order = list(Vals_dict.keys())

# Add a temporary column for the custom iteration order
percentage_diffs_df['IterationOrder'] = percentage_diffs_df['Iteration'].apply(lambda x: iteration_order.index(x) if x in iteration_order else -1)

# Reset the index
percentage_diffs_df.reset_index(drop=True, inplace=True)

# Drop the temporary column
percentage_diffs_df.drop(columns=['IterationOrder'], inplace=True)

# Extract the iteration type (C_A, C_B, etc.)
percentage_diffs_df['IterationType'] = percentage_diffs_df['Iteration'].str.extract(r'([A-Z]+_[A-Z]+)_')

# Sort the DataFrame first by IterationValue in descending order, then by the custom IterationOrder
percentage_diffs_df = percentage_diffs_df.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

median_values = percentage_diffs_df.groupby(['Circuit', 'IterationType']).agg({
    'PercentageDiff': 'median',
    'IterationValue': 'first'
}).reset_index()

median_values = median_values.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

max_percentages = percentage_diffs_df.groupby(['Circuit', 'IterationType'])['PercentageDiff'].idxmax()

result_df_max = percentage_diffs_df.loc[max_percentages].reset_index(drop=True)
result_df_max = result_df_max.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

min_percentages = percentage_diffs_df.groupby(['Circuit', 'IterationType'])['PercentageDiff'].idxmin()

result_df_min = percentage_diffs_df.loc[min_percentages].reset_index(drop=True)
result_df_min = result_df_min.sort_values(by=['IterationValue', 'IterationType'], ascending=[False, True])

wiwu_U = result_df_min[result_df_min['Iteration'].str.startswith('U')]
wiwu_Z = result_df_min[result_df_min['Iteration'].str.startswith('Z')]

data = defaultdict(list)

def plot_iterations_cumulative_2(df_max1, df_min1, styles, selected_circuits, it):
    # Filter the DataFrame based on the 'Iteration' column values starting with 'it'
    df_max = df_max1[df_max1['Iteration'].str.startswith(it)]
    df_min = df_min1[df_min1['Iteration'].str.startswith(it)]
    median_values1 =  median_values[median_values['IterationType'].str.startswith(it)]
    
    if df_max.empty or df_min.empty:
        return
    
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot each circuit separately with provided styles
    for circuit in selected_circuits:
        if circuit in styles:  # Ensure the circuit is in styles
            size = styles[circuit]['size']
            circuit_data_max = df_max[df_max['Circuit'] == circuit]
            circuit_data_min = df_min[df_min['Circuit'] == circuit]
            
            if circuit_data_max.empty or circuit_data_min.empty:
                print(f"No data for circuit {circuit}")
                continue
            
            iterations_max = circuit_data_max['Iteration']
            diffs_max = circuit_data_max['PercentageDiff']
            iterations_min = circuit_data_min['Iteration']
            diffs_min = circuit_data_min['PercentageDiff']
            
            # Calculate cumulative differences
            cumulative_diff_max = diffs_max.cumsum().tolist()
            cumulative_diff_min = diffs_min.cumsum().tolist()
            
            circuit_data_median = median_values1[median_values1['Circuit'] == circuit]
            diffs_median = circuit_data_median['PercentageDiff']
            cumulative_diff_median = diffs_median.cumsum().tolist()
                
            # Truncate the longer list to match the length of the shorter one
            min_len = min(len(cumulative_diff_max), len(cumulative_diff_min))
            cumulative_diff_max = cumulative_diff_max[:min_len]
            cumulative_diff_min = cumulative_diff_min[:min_len]
            
            if len(cumulative_diff_max) != len(cumulative_diff_min):
                print(f"Truncated cumulative differences for circuit {circuit}")
            
            # Print cumulative differences
            print(f"Circuit: {circuit}")
            print(f"Cumulative Max: {round(cumulative_diff_max[-1],3)}")
            print(f"Cumulative Min: {round(cumulative_diff_min[-1],3)}")
            print(f"Cumulative Median: {round(cumulative_diff_median[-1],3)}\n")
            
            cummax = cumulative_diff_max[-1] / len(cumulative_diff_max)
            cummin = cumulative_diff_min[-1] / len(cumulative_diff_min)
            cummedian = cumulative_diff_median[-1] / len(cumulative_diff_median)
            
            data[circuit].append((cummax, cummin, cummedian))
            
            # Plot cumulative differences
            line_max, = ax.plot(range(len(cumulative_diff_max)), cumulative_diff_max, 
                                color=styles[circuit]['color'], 
                                label=f"{circuit} max", 
                                linewidth=size)
            
            line_min, = ax.plot(range(len(cumulative_diff_min)), cumulative_diff_min, 
                                color=styles[circuit]['color'],
                                label=f"{circuit} min", 
                                linewidth=size)
           # Fill between max and min lines
            ax.fill_between(range(len(cumulative_diff_min)), cumulative_diff_max, cumulative_diff_min, 
                            color=styles[circuit]['color'], alpha=0.1)
            
            
            line_median, = ax.plot(range(len(cumulative_diff_median)), cumulative_diff_median, 
                                color='black',
                                label=f"{circuit} median", 
                                linewidth=size,
                                linestyle='--')
            
            # Determine the position for text alignment
            x_position = (len(cumulative_diff_median))*1.02
            y_position = cumulative_diff_median[-1]
            
            # Calculate text box dimensions
            bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)
            
            # Add text to the plot aligned with the line
            ax.text(x_position, y_position, f"Circuit: {circuit}",
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=ax.transData,
                    fontsize=12, bbox=bbox_props)
            
    if it == 'Z':
        line_positions = [2.5, 7.5, 11.5, 19.5, 24.5, 27.5, 29.5]
        for pos in line_positions:
            ax.axvline(x=pos, color='gray', linestyle='--', linewidth=1)
    
    # Determine the range of values for x-axis labels
    max_iteration = df_max['IterationValue'].astype(float).max()
    min_iteration = df_max['IterationValue'].astype(float).min()
    
    # Generate 10 evenly spaced labels between min and max iteration values
    x_labels = np.linspace(max_iteration, min_iteration, num=10)
    x_pos = np.linspace(0, len(circuit_data_max)-1, num=10)
    
    # Set x ticks and labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{label:.2f}' for label in x_labels])
    
    ax.set_xlabel('Irradiation')
    ax.set_ylabel('Cumulative % Difference')
    ax.set_title(f'Cumulative Percentage Difference 4x4 for Iterations {it}')
    plt.legend()
    plt.tight_layout()
    plt.show()


selected_circuits = [
'PER_D',
'PER',
'L_D',
'L',
'R1_D',
'R1',
'R2R3_D',
'R2R3',
'TCT_D',
'TCT',
'SP_D',
'SP'
]

# Possible iterations: 
plot_iterations_cumulative_2(result_df_max, result_df_min, styles, selected_circuits,
# 'U_'
# 'US_'
# 'USD_' 
# 'U'
'Z'
)

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Define the output folder
output_folder = "C:/Users/ruiui/Desktop/TABELAS/4"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define the output file path
output_file = os.path.join(output_folder, 'Zd.csv')

# Save DataFrame to CSV
df.to_csv(output_file, index=False)