import matplotlib.pyplot as plt
import matplotlib as mpl

# Define color palette
COLORS_SONG = {
    '#3b3e37',           
    '#535e4e'           
    '#6d593d',
    '#9c8559',
    '#d2c591',
}

# # Matplotlib style configuration
# def set_style():
#     """Apply consistent styling to plots."""
#     mpl.rcParams['figure.facecolor'] = 'white'
#     mpl.rcParams['axes.facecolor'] = COLORS['background']
#     mpl.rcParams['axes.edgecolor'] = 'black'
#     mpl.rcParams['axes.linewidth'] = 1.2
#     mpl.rcParams['grid.color'] = 'white'
#     mpl.rcParams['grid.linewidth'] = 1.0
#     mpl.rcParams['font.size'] = 11
#     mpl.rcParams['axes.labelsize'] = 12
#     mpl.rcParams['axes.titlesize'] = 14
#     mpl.rcParams['xtick.labelsize'] = 10
#     mpl.rcParams['ytick.labelsize'] = 10
#     mpl.rcParams['legend.fontsize'] = 11
#     mpl.rcParams['lines.linewidth'] = 2

# Usage example:
# from styling import set_style, COLORS
# set_style()
# plt.plot(x, y, color=COLORS['vacuum'], label='Vacuum')