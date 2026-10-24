import stainnormprogram.norm.background as background
from stainnormprogram.slide import SlideImage, SlideImageCollection
from stainnormprogram.assessment.color import ColorSample, ColorDistribution, ColorComparison

import os
import math
import io
import base64
import pickle
import random
from typing import Dict, List, Tuple, Any, Union

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'iframe'

import colour
import numpy as np
import matplotlib.pyplot as plt

plot_labels = {
    "color_dist_h": ["Distance", "HSV Hue"],
    "color_dist_s": ["Distance", "HSV Saturation"],
    "color_dist_v": ["Distance", "HSV Value"],
    "color_dist_rgb": ["Distance", "RGB"],
    "color_dist_oklab": ["Distance", "OKLAB"],

    "palette_compare": ["Difference", "Comparison to Reference Palette"],
    "palette_Hdiv": ["Diversity", "Palette Hematoxylin Diversity"],
    "palette_Ediv": ["Diversity", "Palette Eosin Diversity"],
    "palette_HEsep": ["Separability", "Palette Hematoxylin & Eosin Separability"],

    "gamut_oog_count": ["Pixels", "Number of Pixels Outside of Reference Gamut"],
    "gamut_ig_count": ["Pixels", "Number of Pixels Inside Reference Gamut"],

    "background": ["Distance", "Comparison to Reference Background"],

    "contrast": ["+/- Change", "Contrast Change from Original"],

    "structure": ["Absolute Change", "Structure Change from Original"],
}

class HistPlot:

    def __init__(self, data: Union[ColorSample, ColorDistribution, List[Union[ColorSample, ColorDistribution]]], trace_labels=None):
        self.data = data
        self.trace_labels = trace_labels
        
        if self.trace_labels is None:
            if not isinstance(data, list):
                self.trace_labels = data.identifier
            else:
                identifiers = [d.identifier for d in data]
                if len(set(identifiers)) == len(identifiers):
                    self.trace_labels = identifiers


    def hist_plot_3d(self, xedges, yedges, zedges, hist, axes=["x","y","z"], title="Histogram"):
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        zcenters = (zedges[:-1] + zedges[1:]) / 2

        # Compute a meshgrid to get all of the coordinates
        xc, yc, zc = np.meshgrid(xcenters, ycenters, zcenters, indexing='ij')
        x_flat = xc.flatten()
        y_flat = yc.flatten()
        z_flat = zc.flatten()

        # Mask out all zero count bins
        hist_flat = hist.flatten()
        mask = hist_flat > 0
        x_flat = x_flat[mask]
        y_flat = y_flat[mask]
        z_flat = z_flat[mask]
        hist_flat = hist_flat[mask]

        # Plot histogram
        fig = go.Figure(data=go.Scatter3d(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            mode='markers',
            marker=dict(
                sizemode='diameter',
                sizeref=10,
                color=hist_flat,
                colorscale='Viridis',
            ),
            customdata=hist_flat,
            hovertemplate=(
                "<b>Bar Properties</b><br>" +
                f"{axes[0]}: %{x:.3f}<br>" +
                f"{axes[1]}: %{y:.3f}<br>" +
                f"{axes[2]}: %{z:.3f}<br>" +
                "Height: %{customdata:}"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(height=800, width=800,
            scene=dict(
                xaxis=dict(title=dict(text=axes[0])),
                yaxis=dict(title=dict(text=axes[1])),
                zaxis=dict(title=dict(text=axes[2]))
            ),
            title=dict(text=f'{title}'),
            font=dict(family="Arial", size=14, weight="bold"),
            title_font_size=25,
        )
        
        return fig

    def oklab_hist_plot(self, density=False):

        if not isinstance(self.data, (ColorSample, ColorDistribution)):
            raise Exception

        # Get histogram
        if isinstance(self.data, ColorSample):
            hist = self.data.get_oklab_hist(density=density)
        else:
            hist = self.data.get_oklab_hist()

        fig = self.hist_plot_3d(self.data.oklab_l_hist_edges, self.data.oklab_a_hist_edges, self.data.oklab_b_hist_edges, hist, axes=["l", "a", "b"], title="OKLAB")
        return fig
        

    def rgb_hist_plot(self, density=False):

        if not isinstance(self.data, (ColorSample, ColorDistribution)):
            raise Exception

        # Get histogram
        if isinstance(self.data, ColorSample):
            hist = self.data.get_rgb_hist(density=density)
        else:
            hist = self.data.get_rgb_hist()

        fig = self.hist_plot_3d(self.data.rgb_r_hist_edges, self.data.rgb_g_hist_edges, self.data.rgb_b_hist_edges, hist, axes=["R", "G", "B"], title="RGB")  
        return fig

    def hist_plot_2d(self, edges, heights, identifiers=None, axis_title="Values", title="Histogram", density=False):
        bin_centers = (edges[:-1] + edges[1:]) / 2
        bin_widths = np.diff(edges)

        if isinstance(heights, np.ndarray):
            heights = [heights]
        if not isinstance(identifiers, list):
            identifiers = [identifiers]
            if len(identifiers) != len(heights):
                identifiers = ["Image" + str(i) for i in range(1,len(heights)+1)]

        fig = go.Figure()

        for i, height in enumerate(heights):
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=height,
                    width=bin_widths,
                    offset=0,
                    marker_line_color="white",
                    marker_line_width=1,
                    opacity=0.6,
                    name=identifiers[i],
                )
            )

        yaxis_title = "Density" if density else "Pixels"
        fig.update_layout(
            title=title,
            xaxis_title=axis_title,
            yaxis_title=yaxis_title,
            bargap=0,
            barmode="overlay",
            font=dict(family="Arial", size=14, weight="bold"),
            title_font_size=25,
        )

        return fig
            
    def h_hist_plot(self, density=False):
        
        if not isinstance(self.data, (ColorSample, ColorDistribution)):
            raise Exception

        if isinstance(self.data, ColorSample):
            heights = self.data.get_h_hist(density=density)
        else:
            heights = self.data.get_h_hist()
            density = True

        fig = self.hist_plot_2d(self.data.hsv_h_hist_edges, heights, identifiers=self.data.identifier, axis_title="Hue", title="HSV Hue", density=density)
        return fig
        
    def s_hist_plot(self, density=False):

        if not isinstance(self.data, (ColorSample, ColorDistribution)):
            raise Exception

        if isinstance(self.data, ColorSample):
            heights = self.data.get_s_hist(density=density)
        else:
            heights = self.data.get_s_hist()
            density = True

        fig = self.hist_plot_2d(self.data.hsv_s_hist_edges, heights, identifiers=self.data.identifier, axis_title="Saturation", title="HSV Saturation", density=density)
        return fig
         
    def v_hist_plot(self, density=False):

        if not isinstance(self.data, (ColorSample, ColorDistribution)):
            raise Exception

        if isinstance(self.data, ColorSample):
            heights = self.data.get_v_hist(density=density)
        else:
            heights = self.data.get_v_hist()
            density = True

        fig = self.hist_plot_2d(self.data.hsv_v_hist_edges, heights, identifiers=self.data.identifier, axis_title="Value", title="HSV Value", density=density)
        return fig

    def h_hist_compare(self, density=False):
        
        if not isinstance(self.data, list):
            raise Exception

        for d in self.data:
            if isinstance(d, ColorDistribution):
                density = True

        heights = []
        for d in self.data:
            if isinstance(d, ColorSample):
                heights.append(d.get_h_hist(density=density))
            else:
                heights.append(d.get_h_hist())

        fig = self.hist_plot_2d(self.data[0].hsv_h_hist_edges, heights, identifiers=self.trace_labels, axis_title="Hue", title="HSV Hue", density=density)
        return fig
        
    def s_hist_compare(self, density=False):
        
        if not isinstance(self.data, list):
            raise Exception

        for d in self.data:
            if isinstance(d, ColorDistribution):
                density = True

        heights = []
        for d in self.data:
            if isinstance(d, ColorSample):
                heights.append(d.get_s_hist(density=density))
            else:
                heights.append(d.get_s_hist())

        fig = self.hist_plot_2d(self.data[0].hsv_s_hist_edges, heights, identifiers=self.trace_labels, axis_title="Saturation", title="HSV Saturation", density=density)
        return fig

    def v_hist_compare(self, density=False):
        
        if not isinstance(self.data, list):
            raise Exception

        for d in self.data:
            if isinstance(d, ColorDistribution):
                density = True

        heights = []
        for d in self.data:
            if isinstance(d, ColorSample):
                heights.append(d.get_v_hist(density=density))
            else:
                heights.append(d.get_v_hist())

        fig = self.hist_plot_2d(self.data[0].hsv_v_hist_edges, heights, identifiers=self.trace_labels, axis_title="Value", title="HSV Value", density=density)
        return fig

    def uv_hist(self, color=None, density=False):
        """
        Plots the CIE 1976 UCS chromaticity diagram on a u'v' coordinate grid, then adds points.
        Modified from
            Source - https://stackoverflow.com/a/76033328
            Posted by Kel Solaar
            Retrieved 2026-07-02, License - CC BY-SA 4.0
        """

        data = self.data
        if not isinstance(data, list):
            data = [data]

        if color is not None and not isinstance(color, list):
            color = [color]
            if len(color) != len(data):
                raise Exception
        else:
            color = ["black"] * len(data)

        for d in self.data:
            if isinstance(d, ColorDistribution):
                density = True
        
        u_prime_centers = (data[0].uv_u_hist_edges[:-1] + data[0].uv_u_hist_edges[1:]) / 2
        v_prime_centers = (data[0].uv_v_hist_edges[:-1] + data[0].uv_v_hist_edges[1:]) / 2

        colour.plotting.colour_style()
        # Resolution of the background image divided by 10, i.e. 1000px here.
        plt.rcParams.update({"figure.figsize": (10.00, 10.00)})
        
        # Drawing the background image with Matplotlib.
        figure, axes = colour.plotting.diagrams.plot_chromaticity_diagram_colours(
            diagram_colours="RGB", axes_visible=False, show=False, method='CIE 1976 UCS')
        figure.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0) 
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        data_png = "data:image/png;base64," + base64.b64encode(buffer.getbuffer()).decode("utf8")
        plt.close()

        identifiers = self.trace_labels
        if not isinstance(identifiers, list):
            identifiers = [identifiers]
            if self.trace_labels is None:
                identifiers = ["Image" + str(i) for i in range(1,len(data)+1)]
        
        # Displaying it with Plotly.
        figure = go.Figure()
        figure.update_layout(
            width=1000,
            height=1000,
            xaxis=dict(
                range=[0, 1]
            ),
            yaxis=dict(
                range=[0, 1]
            ),
        )
        figure.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
          )
        figure.add_layout_image(
            dict(
                source=data_png,
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="stretch",
                layer="below"
            )
        )

        for i, d in enumerate(data):
            x_coords = []
            y_coords = []
            vals = []

            for u in range(d.uv_bins[0]):
                for v in range(d.uv_bins[1]):
                    if d.uv_hist[u,v] > 0:
                        x_coords.append(u_prime_centers[u])
                        y_coords.append(v_prime_centers[v])
                        vals.append(d.uv_hist[u,v])

            opacities = np.array(vals)
            opacities = opacities / opacities.max()
            opacities = np.power(opacities, 0.5)
            opacities = opacities.clip(0,1)

            figure.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    marker=dict(
                        size=15,
                        color=color[i],
                        opacity=opacities,
                    ),
                    name=identifiers[i],
                    text=identifiers[i],
                    customdata=vals,
                    hovertemplate=(
                        "<b>Bar Properties</b><br>" +
                        "u': %{x:.3f}<br>" +
                        "v': %{y:.3f}<br>" +
                        "Height: %{customdata:}"
                        "<extra></extra>"
                    )
                )
            )

        figure.update_layout(
            width=1000,
            height=1000,
            xaxis=dict(range=[0, 0.7]),
            yaxis=dict(range=[0, 0.7]),
            title=f"CIE 1976 u'v' Histogram",
            xaxis_title="u'",
            yaxis_title="v'",
            font=dict(family="Arial", size=14, weight="bold"),
            title_font_size=25,
        )

        print("Click on the labels in the legend to toggle the plots on/off. Repeatedly toggle to see the difference between the two plots.")
        
        return figure

class ScatterPlot:
    """A scatter plot where the x-axis is images.
    """

    def __init__(self, data: Dict[str, List[Tuple[str, str, Union[float, Tuple[float, float]]]]], yaxis_title: str = "Value", title: str = "Scatter Plot"):
        """Constructs a ScatterPlot.

        Args:
            data: Data to plot. A dict with keys for each trace, and each trace being a list of (identifier, group, value) tuples. Value can be a float or a tuple of two floats (mean, std).
            yaxis_title: Y-axis title.
            title: Title of the plot.
        """
        self.data = data
        self.yaxis_title = yaxis_title
        self.title = title
        self.fig = None

    def make_plot(self, threshold: Union[float, None] = None, sort: bool = True, sort_by: Union[str, None] = None, clip_ids: bool = False, hide_ids: bool = False):

        fig = go.Figure()

        for trace, points in self.data.items():
            ids = []
            vals = []
            groups = []
            errors = []
            for pt in points:
                ids.append(pt[0])
                groups.append(pt[1])
                val = pt[2]
                if isinstance(val, tuple):
                    vals.append(val[0])
                    errors.append(val[1])
                else:
                    vals.append(val)
                
            # Clip identifiers if they are too long
            if clip_ids:
                for i in range(len(ids)):
                    if len(ids[i]) > 25:
                        ids[i] = ids[i][0:25]

            if sort and trace == sort_by:
                if len(errors) > 0:
                    sorted_tuples = sorted(zip(vals, errors, ids))
                    vals, errors, ids = zip(*sorted_tuples)
                    ids = list(ids)
                    vals = list(vals)
                    errors = list(errors)
                else:
                    sorted_tuples = sorted(zip(vals, ids))
                    vals, ids = zip(*sorted_tuples)
                    vals = list(vals)
                    ids = list(ids)

                fig.update_xaxes(
                    categoryorder="array",
                    categoryarray=ids,
                )

            fig.add_trace(
                go.Scatter(
                    x=ids,
                    y=vals,
                    mode="markers",
                    marker=dict(size=12),
                    customdata=groups,
                    hovertemplate="<b>ID:</b> %{x}<br><b>Group:</b> %{customdata}<br><b>Value:</b> %{y}<extra></extra>",
                    name=trace,
                )
            )

            if len(errors) > 0:
                fig.update_traces(
                    error_y=dict(
                        type="data",
                        array=errors,
                        visible=True,
                    ),
                    selector=dict(name=trace),
                )

        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                line_width=3,
                annotation_text=f"Threshold: {threshold}",
                annotation_position="top left",
            )

        fig.update_layout(
            title=self.title,
            xaxis_title="Images",
            yaxis_title=self.yaxis_title,
            template="plotly_white",
            xaxis=dict(tickangle=-45),
            font=dict(family="Arial", size=14, weight="bold"),
            title_font_size=25,
        )

        if hide_ids:
            fig.update_layout(xaxis=dict(showticklabels=False))

        self.fig = fig

    def get_fig(self):
        return self.fig
