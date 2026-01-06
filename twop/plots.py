import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import Normalize

def axs_provided(axs=None, **fig_kwargs):
    if axs is None:
        fig, axs = plt.subplots(**fig_kwargs)
        return axs, False
    return axs, True

def plot_raw_traces(s2p_out, cell_range=None, axs=None, title=None, **fig_kwargs):
    """
    Plot ΔF/F traces for each cell in individual subplots (one per row).

    Parameters
    ----------
    s2p_out : object
        Suite2p output object containing F (ΔF/F) traces as array [n_cells, n_frames].
    cell_range : tuple (start, stop), optional
        Range of cells to plot. Defaults to all cells.
    axs : matplotlib Axes or array-like, optional
        Predefined Axes to plot into. If None, a new figure/subplots are created.
    **fig_kwargs : dict
        Extra kwargs passed to plt.subplots (e.g. figsize=(10, 8)).
    """
    # Determine range
    if cell_range is None:
        cell_range = (0, s2p_out.F.shape[0])
    start, stop = cell_range

    n_cells = stop - start
    n_frames = s2p_out.F.shape[1]
    frame_list = np.arange(n_frames)

    # Create subplots (one row per cell)
    if axs is None:
        fig, axs = plt.subplots(
            n_cells, 1, sharex=True, figsize=fig_kwargs.get("figsize", (10, 1.5 * n_cells))
        )
        axs_provided = False
    else:
        fig = None
        axs_provided = True

    # Ensure axs is iterable
    if n_cells == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        cell_idx = start + i
        ax.plot(frame_list, s2p_out.F[cell_idx], lw=0.8, color='black')
        ax.set_ylabel(f"Cell {cell_idx}", rotation=0, labelpad=25, va='center')
        ax.axis('off')  # Turn off axis lines/ticks
    axs[-1].set_xlabel("Frame")

    plt.tight_layout(h_pad=0)
    # Add figure-level title (not tied to any subplot)
    if title is not None:
        if not axs_provided:
            fig.suptitle(title, fontsize=14, y=1.02)
        else:
            plt.suptitle(title, fontsize=14, y=1.02)
    if not axs_provided:
        return fig, axs


def plot_dff(s2p_out, cell_range=None, axs=None, title=None, **fig_kwargs):
    """
    Plot dff traces for each cell in individual subplots (one per row).

    Parameters
    ----------
    s2p_out : object
        Suite2p output object containing F (ΔF/F) traces as array [n_cells, n_frames].
    cell_range : tuple (start, stop), optional
        Range of cells to plot. Defaults to all cells.
    axs : matplotlib Axes or array-like, optional
        Predefined Axes to plot into. If None, a new figure/subplots are created.
    **fig_kwargs : dict
        Extra kwargs passed to plt.subplots (e.g. figsize=(10, 8)).
    """
    dff = s2p_out.calc_deltaF()[0]
    # Determine range
    if cell_range is None:
        cell_range = (0, dff.shape[0])
    start, stop = cell_range

    n_cells = stop - start
    n_frames = dff.shape[1]
    frame_list = np.arange(n_frames)

    # Create subplots (one row per cell)
    if axs is None:
        fig, axs = plt.subplots(
            n_cells, 1, sharex=True, figsize=fig_kwargs.get("figsize", (10, 1.5 * n_cells))
        )
        axs_provided = False
    else:
        fig = None
        axs_provided = True

    # Ensure axs is iterable
    if n_cells == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        cell_idx = start + i
        ax.plot(frame_list, dff[cell_idx], lw=0.8, color='black')
        ax.set_ylabel(f"Cell {cell_idx}", rotation=0, labelpad=25, va='center')
        ax.axis('off')  # Turn off axis lines/ticks
    axs[-1].set_xlabel("Frame")

    plt.tight_layout(h_pad=0)
    # Add figure-level title (not tied to any subplot)
    if title is not None:
        if not axs_provided:
            fig.suptitle(title, fontsize=14, y=1.02)
        else:
            plt.suptitle(title, fontsize=14, y=1.02)
    if not axs_provided:
        return fig, axs
    

def plot_spikes(s2p_out, cell_range=None, axs=None, title=None, **fig_kwargs):
    """
    Plot spike traces for each cell in individual subplots (one per row).

    Parameters
    ----------
    s2p_out : object
        Suite2p output object containing F (ΔF/F) traces as array [n_cells, n_frames].
    cell_range : tuple (start, stop), optional
        Range of cells to plot. Defaults to all cells.
    axs : matplotlib Axes or array-like, optional
        Predefined Axes to plot into. If None, a new figure/subplots are created.
    **fig_kwargs : dict
        Extra kwargs passed to plt.subplots (e.g. figsize=(10, 8)).
    """
    # Determine range
    if cell_range is None:
        cell_range = (0, s2p_out.F.shape[0])
    start, stop = cell_range

    n_cells = stop - start
    n_frames = s2p_out.F.shape[1]
    frame_list = np.arange(n_frames)

    # Create subplots (one row per cell)
    if axs is None:
        fig, axs = plt.subplots(
            n_cells, 1, sharex=True, figsize=fig_kwargs.get("figsize", (10, 1.5 * n_cells))
        )
        axs_provided = False
    else:
        fig = None
        axs_provided = True

    # Ensure axs is iterable
    if n_cells == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        cell_idx = start + i
        spikes = s2p_out.get_cell_spikes()
        ax.plot(frame_list, spikes[cell_idx], lw=0.8, color='black')
        ax.set_ylabel(f"Cell {cell_idx}", rotation=0, labelpad=25, va='center')
        ax.axis('off')  # Turn off axis lines/ticks
    axs[-1].set_xlabel("Frame")

    plt.tight_layout(h_pad=0)
    # Add figure-level title (not tied to any subplot)
    if title is not None:
        if not axs_provided:
            fig.suptitle(title, fontsize=14, y=1.02)
        else:
            plt.suptitle(title, fontsize=14, y=1.02)
    if not axs_provided:
        return fig, axs
    
def spikeevent_plot(s2p_out, cell_range=None, axs=None, vertical_spacing=1, **fig_kwargs):
    """
    Raster-style spike events for cells in [cell_range[0], cell_range[1]).
    If axs is None, a new axis is created via axs_provided.
    """
    axis, _ = axs_provided(axs, **fig_kwargs)

    spikes = s2p_out.get_cell_spikes()
    # subset to requested cells
    if cell_range:
        spikes = spikes[cell_range[0]:cell_range[1], :]

    for cell_idx in range(spikes.shape[0]):
        spike_times = np.where(spikes[cell_idx] != 0)[0]
        axis.vlines(
            spike_times,
            ymin=cell_idx * vertical_spacing,
            ymax=(cell_idx + 1) * vertical_spacing,
            color="black",
            linewidth=1)

    # labels/formatting
    axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.set_xlabel("Frame #")
    axis.set_ylabel("Cell # (relative in range)")
    return axis


def smoothed_event_plot(s2p_out, cell_range=None, axs=None, smoothing_sigma=0.5, title=None, **fig_kwargs):
    """
    Heatmap of (optionally smoothed) spike events for cells in [cell_range[0], cell_range[1]).
    If axs is None, a new axis is created via axs_provided.
    """
    axis, _ = axs_provided(axs, **fig_kwargs)

    # Extract the spikes in the specified range of cells
    spks = s2p_out.spks[s2p_out.cell_indices, :]
    if cell_range is None:
        cell_range = [0, s2p_out.num_cells]
    spikes = spks[cell_range[0]:cell_range[1], :]

    # Apply smoothing (Gaussian filter) to the spike matrix
    smoothed_matrix = gaussian_filter(spikes, sigma=smoothing_sigma)

    # # Normalize data (fit on original, transform original → then plot smoothed as-is, or
    # # normalize smoothed; here we keep your original intent and normalize the raw spikes)
    # scaler = MinMaxScaler()
    # _ = scaler.fit(spikes)
    # spikes_norm = scaler.transform(spikes)  # kept in case you want to plot normalized later

    # Create the heatmap plot
    cax = axis.imshow(
        smoothed_matrix,
        aspect="auto",
        cmap="hot",
        origin="lower",
        interpolation="none",
    )

    # Add colorbar
    plt.colorbar(cax, ax=axis, label="Spike Intensity")

    # Set labels and formatting
    axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.set_xlabel("Frame (#)")
    axis.set_ylabel("Cell (#, relative in range)")
    if title is not None:
        axis.set_title(title)
    return axis


def spike_raster(s2p_out, cell_range=None, axs=None, title=None, **fig_kwargs):
    """
    Raster-style spike events for cells in [cell_range[0], cell_range[1]).
    If axs is None, a new axis is created via axs_provided.
    """
    axis, _ = axs_provided(axs, **fig_kwargs)

    spikes = s2p_out.get_cell_spikes()
    # subset to requested cells
    if cell_range:
        spikes = spikes[cell_range[0]:cell_range[1], :]

    # fig, axs = plt.subplots(1, len(spikes), figsize=(50,4))

    # plt.suptitle("Spiking: Front_left")
    # #plt.supylabel('Spike amplitude')
    # for ax_idx in range(len(spikes)):
    #     plot_spikes = spikes[ax_idx]
    norm = Normalize(vmin=np.percentile(spikes, 1), vmax=np.percentile(spikes, 99))
    axis.imshow(spikes, aspect='auto', cmap='binary', origin='lower', norm=norm)
    if title is not None:
        axis.set_title(title)

    # Remove x-axis ticks and labels
    axis.xaxis.set_major_locator(ticker.NullLocator())
    axis.xaxis.set_major_formatter(ticker.NullFormatter())

    # Remove y-axis ticks and labels
    axis.yaxis.set_major_locator(ticker.NullLocator())
    axis.yaxis.set_major_formatter(ticker.NullFormatter())
    plt.show()
    return axis
   