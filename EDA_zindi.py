# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 18:43:15 2025

@author: karel
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



dataset=pd.read_csv("C:\\Users\\karel\\OneDrive\\Bureau\\ISEP1\\AUTRES ISEP1\\ML\\Projets\\Zindi_Fraude\\training.csv")
df=dataset.copy()
# =============================================================================
# Analyse de forme
# =============================================================================
df.columns
df.shape

# info sur les variables
df.info()

# proportion de valeurs manquantes 
df.isnull().sum()/df.shape[0]

# propriétés statistiques du data
df.describe()

# corrélation
corrélation=df.corr()


plt.figure(figsize = (10,6))
sns.heatmap(corrélation, annot = True ,cmap = "coolwarm", fmt = ".2F", linewidth =0.5 , linecolor = 'white',annot_kws = {'size':10}, cbar_kws={'shrink':0.5}  )
plt.title("Correlation Between Numerical Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

