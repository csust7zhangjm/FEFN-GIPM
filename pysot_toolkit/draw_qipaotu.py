import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_pdf import PdfPages
pdf = PdfPages('sr_ao.pdf')
plt.rc('font',family='Times New Roman')

fig, ax = plt.subplots()  # type:figure.Figure, axes.Axes
ax.set_title('The AO and SR$_{0.5}$ on GOT-10K', fontsize=15)
ax.set_xlabel('SR$_{0.5}$(%)', fontsize=14)
ax.set_ylabel('AO', fontsize=14)


trackers = ['SiamRPN++', 'ATOM', 'SiamCAR', 'SiamFC++', 'D3S', 'Ocean', 'PrDiMP', 'KYS', 'SiamRCNN', 'TransT', 'STARK', 'Ours']
SR_05 = np.array([61.6, 63.4, 67.0, 69.5, 67.6, 72.1, 73.8, 75.1, 72.8, 76.8, 77.7, 81.4])
sr_norm = np.array([61.6, 63.4, 67.0, 69.5, 67.6, 72.1, 73.8, 75.1, 72.8, 76.8, 77.7, 81.4]) / 100
performance = np.array([0.517, 0.556, 0.569, 0.595, 0.597, 0.611, 0.634, 0.636, 0.649, 0.671, 0.680, 0.710])

circle_color = ['cornflowerblue', 'deepskyblue',  'turquoise', 'gold', 'yellowgreen', 'orange', 'lightpink', 'deeppink', 'lightsalmon', 'lightcoral', 'lime', 'r']
# Marker size in units of points^2
volume = (400 * sr_norm/8 * performance/0.6) ** 2

ax.scatter(SR_05, performance, c=circle_color, s=volume, alpha=0.4)
ax.scatter(SR_05, performance, c=circle_color, s=20, marker='o')
# text
ax.text(SR_05[0] - 1.5, performance[0] + 0.014, trackers[0], fontsize=8, color='k')
ax.text(SR_05[1] - 0.8, performance[1] + 0.016, trackers[1], fontsize=8, color='k')
ax.text(SR_05[2] - 1, performance[2] - 0.02, trackers[2], fontsize=8, color='k')
ax.text(SR_05[3] - 1.3, performance[3] - 0.024, trackers[3], fontsize=8, color='k')
ax.text(SR_05[4] - 0.5, performance[4] + 0.018, trackers[4], fontsize=8, color='k')
ax.text(SR_05[5] - 0.7, performance[5] - 0.022, trackers[5], fontsize=8, color='k')
ax.text(SR_05[6] - 0.8, performance[6] - 0.025, trackers[6], fontsize=8, color='k')
ax.text(SR_05[7] - 0.4, performance[7] + 0.02, trackers[7], fontsize=8, color='k')
ax.text(SR_05[8] - 1.3, performance[8] + 0.02, trackers[8], fontsize=8, color='k')
ax.text(SR_05[9] - 0.8, performance[9] - 0.025, trackers[9], fontsize=8, color='k')
ax.text(SR_05[10] - 0.8, performance[10] + 0.021, trackers[10], fontsize=8, color='k')
ax.text(SR_05[11] - 0.9, performance[11] + 0.023, trackers[11], fontsize=12, color='k')

ax.grid(which='major', axis='both', linestyle='-.') # color='r', linestyle='-', linewidth=2
ax.set_xlim(60, 84, 5)
ax.set_ylim(0.5, 0.75, 0.02)
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)

fig.savefig('GOT_10k_qp.pdf')


pdf.savefig()
pdf.close()
plt.show()
