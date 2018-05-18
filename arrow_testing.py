import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure();
arrow = patches.FancyArrowPatch((1,0),(0,1),mutation_scale=50,arrowstyle='-|>');
plt.gca().add_patch(arrow);
plt.xlim(0,5)
plt.ylim(0,5)
plt.show()
