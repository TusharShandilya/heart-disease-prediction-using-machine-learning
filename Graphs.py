from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

fig = figure(figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')

fig.canvas.set_window_title('Accuracy Comparison Chart')

# Legend
red_patch = mpatches.Patch(color='#cc0000', label='Selected Algorithm')
orange_patch = mpatches.Patch(color='#FF7F0E', label='Old Algorithm')
blue_patch = mpatches.Patch(color='#1F77B4', label='New Algorithm')
plt.legend(handles=[red_patch, orange_patch, blue_patch])

N = 7
old_accuracy = (84, 84.71, 85.47, 77.26, 84.81, 81.94, 84.34)
improved_accuracy = (84.52, 85, 85.47, 81.50, 84.81, 81.94, 84.34)

ind = np.arange(N)
width = 0.35
a = plt.bar(ind, old_accuracy, width, label='Old Accuracy')
b = plt.bar(ind + width, improved_accuracy, width, label='Current Accuracy')

b.get_children()[2].set_color('#cc0000')
a.get_children()[2].set_alpha(0.3)

rects = a.patches

for rect, label in zip(rects, old_accuracy):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + .5, label,
             ha='center', va='bottom', alpha=0.6)

rects = b.patches

for rect, label in zip(rects, improved_accuracy):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + .5, label,
             ha='center', va='bottom')

plt.ylabel('Accuracy %')
plt.title('Accuracy Chart')

plt.xticks(ind + width / 2, ('KNN', 'SVM', 'Logistic Regr.', 'Decision Tree', 'Random Forest', 'Naives Bayes', 'ANN'),
           rotation=90)

plt.ylim(0, 100)

plt.show()
