import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D   


def visualize_location(df, bin_size, flare_class='X'):
    cmap=plt.cm.Blues

    plt.rcParams.update({'font.size': 15})

    
    df = df.copy()
    df[["lon", "lat"]] = df["fl_location"].str.strip(r"[()]").str.split(",", expand=True).astype(str)
    df['lon'] = pd.to_numeric(df['lon']).round(decimals=2).astype(str).replace(r'\.0$', '', regex=True)
    df[["lon", "lat"]] = df[['lon', 'lat']].astype(float)

    #To include the limb locations in the bins with in limb-location. Just handling border cases.
    #This actually makes, grid of [-75 to 70] [85 to 90] inclusive of both the ends hence 6 by 5 grid in the border
    df.loc[df['lon'] == -70, 'lon'] -= 1
    df.loc[df['lon'] == 90, 'lon'] -= 1

    print(df.lon.max(), df.lon.min())
    print(df.lat.max(), df.lat.min())

    # Create a new column 'result' based on the condition
    df['result'] = np.where(df['flare_prob'] >= 0.5, 'TP', 'FN')

    # Define the grid and group the DataFrame by the grid and 'result'
    grid = bin_size
    df_grouped = df.groupby([np.floor(df['lat']/grid)*grid, np.floor(df['lon']/grid)*grid, 'result']).size().unstack(fill_value=0)

    # Compute TP/(TP+FN)
    df_grouped['TP/(TP+FN)'] = df_grouped['TP'] / (df_grouped['TP'] + df_grouped['FN'])
    vmin = df_grouped['TP/(TP+FN)'].min()
    vmax = df_grouped['TP/(TP+FN)'].max()
    # print(df_grouped.to_markdown())

    # Create the heatmap
    heatmap, xedges, yedges = np.histogram2d(df_grouped.index.get_level_values(1), 
                                            df_grouped.index.get_level_values(0), 
                                            bins=[np.arange(-90, 95, grid), np.arange(-35, 40, grid)], 
                                            weights=df_grouped['TP/(TP+FN)'])

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # fig = figure(figsize=(16, 4), dpi=300)
    fig, ax = plt.subplots(figsize=(10.1, 4.3))

    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    # Add stars at the center of any grid where TP is 0 but FN is greater than 0
    for i, x in enumerate(np.arange(-90, 95, grid)):
        for j, y in enumerate(np.arange(-35, 40, grid)):
            if (y, x) in df_grouped.index:
                if df_grouped.loc[(y, x), 'TP'] == 0 and df_grouped.loc[(y, x), 'FN'] != 0:
                    star_artist = ax.scatter(x+(bin_size/2), y+(bin_size/2), marker='*', s=50, color='red', edgecolors='red')
                        
    # cbar = plt.colorbar(im, label='Recall', shrink=0.28, pad=0.02, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar.ax.tick_params(labelsize=6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Recall')

    ax.axvline(x=-70, color='red', linestyle='-')
    ax.axvline(x=70, color='red',  linestyle='-')

    ax.set_xticks(np.arange(-90, 95, grid))
    ax.set_ylabel('Heliographic Latitude', fontsize=15)
    ax.set_yticks(np.arange(-35, 40, grid))
    ax.grid(True)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    ax.grid(which='minor', color='#EEEEEE', linestyle='-', linewidth=0.01)            
    # Add the legend for the stars
    if flare_class=='X':
        legend_elements = [Line2D([], [], marker='*', color='red', label='No Correct Predictions', markersize=8, linestyle='None')]
        l = ax.legend(handles=legend_elements, loc=(0.4, 1.004))
        l.get_frame().set_alpha(0.7)
    # ax.set_title('Recall within a 10-degree grid', fontsize=15)

    if flare_class!= 'Combined':
        ax.tick_params(left=True,
                bottom=False,
                labelleft=True,
                labelbottom=False)
        ax.grid(True)
    else:
        ax.set_xlabel('Heliographic Longitude (Central Meridian Distance)', fontsize=15)
        ax.set_xticks(np.arange(-90, 95, grid))
        ax.set_xticklabels(np.arange(-90, 95, grid), fontsize=15, rotation=45)

    fig.tight_layout(pad=0)
    fig.savefig(f'plots/{flare_class}flare_bin_{bin_size}.svg', dpi=300, transparent=True)
    # plt.show()

bin = 5
i=bin
df1 = pd.read_csv(r'x_class.csv')
df2 = pd.read_csv(r'm_class.csv')

visualize_location(df1, i,  'X')
visualize_location(df2, i,  'M')
visualize_location(pd.concat([df1, df2]), i, 'Combined')
