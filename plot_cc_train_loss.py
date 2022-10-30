import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

pdf = PdfPages('data/cc_loss.pdf')
sns.set(style="whitegrid")  # 这是seaborn默认的风格


fig = plt.figure(figsize=(20, 5.4))
w = 0.3

method = ['Train Loss', 'Val Loss']

ax1 = fig.add_subplot(131)
train_loss = np.load("data/cc_london_train_loss.npy")
test_loss = np.load("data/cc_london_val_loss.npy")

loss = np.concatenate([train_loss[None, :], test_loss[None, :]])

data = pd.DataFrame(columns=['Epochs', 'Loss', 'Loss Type'])
for i in range(20):
    for j in range(2):
        data = data.append({"Epochs": str(i+1), 'Loss': loss[j, i], 'Loss Type': method[j]},
                           ignore_index=True)
paper_rc = {'lines.linewidth': 3, 'lines.markersize': 5}
sns.set_context("paper", rc=paper_rc)
ax = sns.lineplot(data=data, x='Epochs', y='Loss', hue='Loss Type', style='Loss Type', dashes=False)
plt.setp(ax.xaxis.get_ticklabels(), fontsize=14)
plt.setp(ax.yaxis.get_ticklabels(), fontsize=14)
plt.setp(ax.xaxis.get_label(), fontsize=17)
plt.setp(ax.yaxis.get_label(), fontsize=17)
plt.setp(ax.get_legend().get_texts(), fontsize='13')  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='14')  # for legend title
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=w, hspace=0.15)
plt.title("(a) London", y=-0.3, fontsize=19)

ax1 = fig.add_subplot(132)
train_loss = np.load("data/cc_madrid_train_loss.npy")
test_loss = np.load("data/cc_madrid_val_loss.npy")

loss = np.concatenate([train_loss[None, :], test_loss[None, :]])

data = pd.DataFrame(columns=['Epochs', 'Loss', 'Loss Type'])
for i in range(20):
    for j in range(2):
        data = data.append({"Epochs": str(i+1), 'Loss': loss[j, i], 'Loss Type': method[j]},
                           ignore_index=True)
paper_rc = {'lines.linewidth': 3, 'lines.markersize': 5}
sns.set_context("paper", rc=paper_rc)
ax = sns.lineplot(data=data, x='Epochs', y='Loss', hue='Loss Type', style='Loss Type', dashes=False)
plt.setp(ax.xaxis.get_ticklabels(), fontsize=14)
plt.setp(ax.yaxis.get_ticklabels(), fontsize=14)
plt.setp(ax.xaxis.get_label(), fontsize=17)
plt.setp(ax.yaxis.get_label(), fontsize=17)
plt.setp(ax.get_legend().get_texts(), fontsize='13')  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='14')  # for legend title
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=w, hspace=0.15)
plt.title("(b) Madrid", y=-0.3, fontsize=19)

ax1 = fig.add_subplot(133)
train_loss = np.load("data/cc_melbourne_train_loss.npy")
test_loss = np.load("data/cc_melbourne_val_loss.npy")

loss = np.concatenate([train_loss[None, :], test_loss[None, :]])

data = pd.DataFrame(columns=['Epochs', 'Loss', 'Loss Type'])
for i in range(20):
    for j in range(2):
        data = data.append({"Epochs": str(i+1), 'Loss': loss[j, i], 'Loss Type': method[j]},
                           ignore_index=True)
paper_rc = {'lines.linewidth': 3, 'lines.markersize': 5}
sns.set_context("paper", rc=paper_rc)
ax = sns.lineplot(data=data, x='Epochs', y='Loss', hue='Loss Type', style='Loss Type', dashes=False)
plt.setp(ax.xaxis.get_ticklabels(), fontsize=14)
plt.setp(ax.yaxis.get_ticklabels(), fontsize=14)
plt.setp(ax.xaxis.get_label(), fontsize=17)
plt.setp(ax.yaxis.get_label(), fontsize=17)
plt.setp(ax.get_legend().get_texts(), fontsize='13')  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='14')  # for legend title
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=w, hspace=0.15)
plt.title("(c) Melbourne", y=-0.3, fontsize=19)

plt.tight_layout()
# plt.show()
# plt.savefig('temp.pdf', dpi=200, bbox_inches='tight')
pdf.savefig(dpi=plt.gcf().dpi, bbox_inches='tight')
# plt.show()
pdf.close()
plt.close()
