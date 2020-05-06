import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil, floor

df = pd.DataFrame(columns=['LR', 'MoCo_Linear',
                           'MoCo_Efficient'])

df['LR'] = [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
arr = ['{:.3f}'.format(x) for x in df['LR']]
# df['MoCo_Super'] = [0.8746, 0.9017, 0.9197, 0.93, 0.9461, 0.953]
df['MoCo_Efficient'] = [0, 0.8737, 0.8773, 0.8699, 0.8521, 0.7542, 0, 0]
df['MoCo_Linear'] = [0, 0, 0.4958, 0.7839, 0.7229, 0.7781, 0.5877, 0]
# df['ResNet'] = [0.7752]

sns.set()
ax = sns.lineplot(x='LR', y='value', hue='variable', style="variable",
                  markers=True, dashes=False, data=pd.melt(df, ['LR']))
ax.set(xscale='log')
ax.set_ylim([0.45, 1.00])
plt.xticks(df["LR"], arr)
# # ax.set_xticklabels(df['% Data'])
ax.set(xlabel='Learning Rate', ylabel='Best Average AUC over 100 epochs',
       title='LR tuning for one Percentage of Data')
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[1:], labels=labels[1:], loc='lower right')

plt.show()
