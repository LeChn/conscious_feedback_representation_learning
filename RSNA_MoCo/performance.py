import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['% Data', 'MoCo', 'ResNet',
                           'MoCo_Super', 'MoCo_Efficient'])

df['% Data'] = [1, 5, 10, 20, 50, 100]
df['MoCo_Super'] = [0.8746, 0.9017, 0.9197, 0.93, 0.9461, 0.953]
df['MoCo_Efficient'] = [0.7951, 0.87, 0.891, 0.9003, 0.9093, 0.9131]
df['MoCo'] = [0.7849, 0.8582, 0.8865, 0.8978, 0.907, 0.9108]
df['ResNet'] = [0.7752, 0.8299, 0.8634, 0.8966, 0.9145, 0.9243]

sns.set()
ax = sns.lineplot(x='% Data', y='value', hue='variable', style="variable", markers=True, dashes=False,
                  data=pd.melt(df, ['% Data']))
ax.set(xscale='log')
ax.set_ylim([0.75, 1.00])
plt.xticks(df["% Data"], df["% Data"])
# ax.set_xticklabels(df['% Data'])
ax.set(xlabel='Percentage of Labeled Data', ylabel='AUC',
       title='MoCo_Super VS. MoCo_Efficient VS. MoCo VS. ResNet')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:], loc='lower right')

plt.show()
