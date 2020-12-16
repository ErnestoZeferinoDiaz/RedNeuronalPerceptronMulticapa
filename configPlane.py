import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

axs2D=[]
axs3D=[]

plt.style.use('dark_background')
fig = plt.figure() 

axs2D.append( fig.add_subplot(1,2,1) )

axs2D = np.array([axs2D]).flatten()
for ax in axs2D:
  ax.grid(color='gray', linestyle='-', which='both', alpha=0.2)
  ax.set_aspect('equal', adjustable='box')  
  
  ax.spines['right'].set_color('none')
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_position(('data',0))
  ax.spines['left'].set_position(('data',0))

  ax.set_xlabel('X - axis')
  ax.set_ylabel('Y - axis')

  ax.xaxis.set_ticks_position('bottom')
  ax.yaxis.set_ticks_position('left')




axs3D.append( fig.add_subplot(1,2,2,projection="3d") )

axs3D = np.array([axs3D]).flatten()
for ax in axs3D:
  ax.w_xaxis.set_pane_color((1,1,1,0.0))
  ax.w_yaxis.set_pane_color((1,1,1,0.0))
  ax.w_zaxis.set_pane_color((1,1,1,0.0))

  ax.w_xaxis.gridlines.set_alpha(0.2)
  ax.w_yaxis.gridlines.set_alpha(0.2)
  ax.w_zaxis.gridlines.set_alpha(0.2)

  ax.set_xlabel('X - axis')
  ax.set_ylabel('Y - axis')
  ax.set_zlabel('Z - axis')

  ax.set_aspect('auto','box')

