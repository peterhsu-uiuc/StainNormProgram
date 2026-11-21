from stainnormprogram.slide import SlideImage
from stainnormprogram.norm.utils import nnls
from stainnormprogram.norm.adjustments import Adjustment

import json
import os

import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.optimize import least_squares

class MacenkoPointCorrection:

    def __init__(self):
        """Perform correction by selecting corresponding points between a properly normalized image and a failure case.
        """

        self.f = None

        # New matrix and maxC after correction
        self.matrix_new = None
        self.maxC_new = None

        # Save locations
        self.ref_y = None
        self.ref_x = None
        self.query_y = None
        self.query_x = None
        self.tile_size = None

        self.before_ref_slide = None
        self.after_ref_slide = None 
        self.ref_matrix = None
        self.ref_maxC = None
        self.ref_I0 = None
        self.ref_pre_adjust = None
        self.before_query_slide = None
        self.after_query_slide = None
        self.query_matrix = None
        self.query_maxC = None
        self.query_I0 = None
        self.query_pre_adjust = None
        self.query_post_adjust = None
        self.target_matrix = None
        self.target_maxC = None
        self.target_I0 = None

        print("""
        DIRECTIONS:

        First, call all relevant functions to set variables up properly.
            * set_reference
            * set_query
            * set_styles

        When ready to select points, call make_selection.

        Pick a point on the reference, then a point on the query, and repeat until you have >= 10 pairs (20 is recommended).
        For each pair, you should pick corresponding points (e.g., darkest part of a nucleus in the reference with darkest part of a nucleus in the query).
        Among all of the pairs, vary the feature selected (e.g. for one pair you can do darkest parts of nuclei, for another pair you can do lightest parts of stroma).

        Then, compute the correction using the compute function.

        Finally, once computation is done, you can view the results using display_results and save using save_corrected_style.
        """)

    def __repr__(self):
        return f"MacenkoPointCorrection(\n \
            before_ref_slide={self.before_ref_slide}\n \
            after_ref_slide={self.after_ref_slide}\n \
            ref_matrix={self.ref_matrix}\n \
            ref_maxC={self.ref_maxC}\n \
            ref_I0={self.ref_I0}\n\n \
            ref_pre_adjust={self.ref_pre_adjust}\n\n\
            before_query_slide={self.before_query_slide}\n \
            after_query_slide={self.after_query_slide}\n \
            query_matrix={self.query_matrix}\n \
            query_maxC={self.query_maxC}\n \
            query_I0={self.query_I0}\n \
            query_pre_adjust={self.query_pre_adjust}\n \
            query_post_adjust={self.query_post_adjust}\n\n \
            target_matrix={self.target_matrix}\n \
            target_maxC={self.target_maxC}\n \
            target_I0={self.target_I0}\n)"

    def save_corrected_style(self, output_path: str):
        """Save the corrected source stain style.

        Args:
            output_path: File path to output the corrected staining style to. Should end with .json.
        """

        if os.path.exists(output_path):
            raise Exception("File already exists. Choose a different output_path.")

        if self.matrix_new is None or self.maxC_new is None:
            raise Exception("Run correction first.")

        output_dict = {
            "matrix": self.matrix_new.tolist(),
            "maxC": self.maxC_new.tolist(),
            "I0": self.query_I0.tolist(),
            "correction": "MacenkoPointCorrection",
        }
 
        with open(output_path, "w") as file:
            json.dump(output_dict, file, indent=4)

    def set_reference(self, before_ref_slide: str | SlideImage, after_ref_slide: str | SlideImage, pre_adjustment: Adjustment | None = None):
        """Set the reference slide (a slide that was normalized well).

        Args:
            before_ref_slide: Either a SlideImage or str path to the image before normalization (the original).
            after_ref_slide: Either a SlideImage or str path to the image after normalization.
            pre_adjustment: Pre-adjustment to apply. This should be the same pre-adjustment used when you stain normalized the reference.
        """
        if isinstance(before_ref_slide, SlideImage):
            self.before_ref_slide = before_ref_slide
        else:
            self.before_ref_slide = SlideImage(before_ref_slide)
        if isinstance(after_ref_slide, SlideImage):
            self.after_ref_slide = after_ref_slide
        else:
            self.after_ref_slide = SlideImage(after_ref_slide)

        self.before_ref_slide.set_adjustment(pre_adjustment)
        self.after_ref_slide.set_adjustment(None)

        self.ref_pre_adjust = pre_adjustment
        
    def set_query(self, before_query_slide: str | SlideImage, after_query_slide: str | SlideImage, pre_adjustment: Adjustment | None = None, post_adjustment: Adjustment | None = None):
        """Set the query slide (a slide that was not normalized well).

        When setting ``pre_adjustment`` and ``post_adjustment``, note that you should use these exact Adjustments when performing stain normalization after correction to ensure
        the results you see during correction match those when normalizing the entire image. Also note that ``post_adjustment`` should be the same as the post-adjustment that was used
        on the normalized reference.

        Args:
            before_query_slide: Either a SlideImage or str path to the image before normalization (the original).
            after_query_slide: Either a SlideImage or str path to the image after normalization.
            pre_adjustment: Pre-adjustment to apply.
            post_adjustment: Post-adjustment to apply.
        """
        if isinstance(before_query_slide, SlideImage):
            self.before_query_slide = before_query_slide
        else:
            self.before_query_slide = SlideImage(before_query_slide)
        if isinstance(after_query_slide, SlideImage):
            self.after_query_slide = after_query_slide
        else:
            self.after_query_slide = SlideImage(after_query_slide)

        self.before_query_slide.set_adjustment(pre_adjustment)
        self.after_query_slide.set_adjustment(None)

        self.query_pre_adjust = pre_adjustment
        self.query_post_adjust = post_adjustment

    def load_style(self, path: str, separate: bool = False) -> dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper function for loading stain style JSONs.

        Args:
            path: Path to the JSON file.
            separate: If True, will return a tuple of matrix, maxC, and I0 instead of a dictionary.

        Returns:
            Either a tuple or dictionary of the matrix, maxC, and I0.
        """
        with open(path, "r") as file:
            style = json.load(file)
        style["matrix"] = np.array(style["matrix"])
        style["maxC"] = np.array(style["maxC"])
        style["I0"] = np.array(style["I0"])
        if not separate:
            return style
        return style["matrix"], style["maxC"], style["I0"]
        
    def set_styles(self, ref_style, query_style, target_style):
        """Set stain styles.

        Args:
            ref_style: Path to the reference image stain style JSON.
            query_style: Path to the query image stain style JSON.
            target_style: Path to the target stain style JSON.
        """
        if isinstance(ref_style, str) and ref_style.endswith(".json"):
            self.ref_matrix, self.ref_maxC, self.ref_I0 = self.load_style(ref_style, separate=True)
        if isinstance(query_style, str) and query_style.endswith(".json"):
            self.query_matrix, self.query_maxC, self.query_I0 = self.load_style(query_style, separate=True)
        if isinstance(target_style, str) and target_style.endswith(".json"):
            self.target_matrix, self.target_maxC, self.target_I0 = self.load_style(target_style, separate=True)

    def make_selection(self, ref_y: int, ref_x: int, query_y: int, query_x: int, tile_size: int) -> plotly.graph_objects.Figure:
        """Make point selections.

        Args:
            ref_y: Upper left corner y-coordinate for the region of interest in the reference to show.
            ref_x: Upper left corner x-coordinate for the region of interest in the reference to show.
            query_y: Upper left corner y-coordinate for the region of interest in the query to show.
            query_x: Upper left corner x-coordinate for the region of interest in the query to show.
            tile_size: Pixel size of the regions of interest.

        Returns:
            A Plotly figure for making the point selection.
        """
        self.ref_y = ref_y
        self.ref_x = ref_x
        self.query_y = query_y
        self.query_x = query_x
        self.tile_size = tile_size

        if self.f is None:
            return self.setup_fig(ref_y, ref_x, query_y, query_x, tile_size)

        else:
            self.clicked_ref = self.f.data[0].customdata[1]
            self.clicked_query = self.f.data[1].customdata[1]
            return self.setup_fig(ref_y, ref_x, query_y, query_x, tile_size, clicked_ref=self.clicked_ref, clicked_query=self.clicked_query)

    def setup_fig(self, ref_y: int, ref_x: int, query_y: int, query_x: int, tile_size: int, clicked_ref: list[tuple[int, int]] = [], clicked_query: list[tuple[int, int]] = []) -> plotly.graph_objects.Figure:
        """Setup the Plotly point selection figure.

        Args:
            ref_y: Upper left corner y-coordinate for the region of interest in the reference to show.
            ref_x: Upper left corner x-coordinate for the region of interest in the reference to show.
            query_y: Upper left corner y-coordinate for the region of interest in the query to show.
            query_x: Upper left corner x-coordinate for the region of interest in the query to show.
            tile_size: Pixel size of the regions of interest.
            clicked_ref: Clicked reference points (if resuming).
            clicked_query: Clicked query points (if resuming).

        Returns:
            A Plotly figure for making the point selection.
        """

        # Load in ref tile
        ref_tile = self.after_ref_slide.get_tile(ref_y, ref_x, tile_size, tile_size)

        # Load in query tile
        query_tile = self.after_query_slide.get_tile(query_y, query_x, tile_size, tile_size)

        # Show both in an interactive plot
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Reference", "Query"))

        fig.add_trace(
            go.Image(z=ref_tile.rgb(as_uint8=True), x0=ref_x, y0=ref_y, customdata=["Reference", clicked_ref]), row=1, col=1
        )
        
        fig.add_trace(
            go.Image(z=query_tile.rgb(as_uint8=True), x0=query_x, y0=query_y, customdata=["Query", clicked_query]), row=1, col=2
        )
        
        fig.update_layout(title=f"Select correspondences in the Reference and Query images",
                         height=600,
                         width=1200,
                         )
        
        f = go.FigureWidget(fig)
        
        def callback(trace, points, selector):

            if points.point_inds:
                
                x = points.xs[0]
                y = points.ys[0]

                if trace.customdata[0] == "Reference":
                    trace.customdata[1].append((x,y))
                    f.layout.title.subtitle.text = f"Selected (x={x}, y={y}) on Reference"
                elif trace.customdata[0] == "Query":
                    trace.customdata[1].append((x,y))
                    f.layout.title.subtitle.text = f"Selected (x={x}, y={y}) on Query"
        
        f.data[0].on_click(callback)
        f.data[1].on_click(callback)

        self.f = f
        
        return self.f

    def clear_selection(self):
        """Clear the selection. To select again, call make_selection.
        """
        self.f = None

    def compute(self, c_ref: list[tuple[int]] | None = None, c_query: list[tuple[int]] | None = None, lam: float = 0.0):
        """Compute the correction from the selected points.

        Args:
            c_ref: Selected coordinate points on the reference image. If not given, will take from the Plotly figure automatically.
            c_query: Selected coordinate points on the query image. If not given, will take from the Plotly figure automatically.
            lam: Lambda coefficient to control the influence of the past stain style calculation [0.0,1.0]. 0.0 is no influence from the past.
        """
        # Given: a list of corresponding points (as pairs) in overall slide coordinates
        if c_ref is None:
            self.clicked_ref = self.f.data[0].customdata[1]
        else:
            self.clicked_ref = c_ref
        if c_query is None:
            self.clicked_query = self.f.data[1].customdata[1]
        else:
            self.clicked_query = c_query

        print("Reference points:", self.clicked_ref)
        print("Query points:", self.clicked_query)

        if len(self.clicked_ref) != len(self.clicked_query):
            raise Exception("You did not click the same number of points in the reference and query.")
        if len(self.clicked_ref) < 10:
            raise Exception("Pick more points. You have less than 10.")
        print("Number of correspondences:", len(self.clicked_ref))

        # Extract the Y values from those points before normalization
        # Get Y values using I0 and pre_adjust
        Y_ref = []
        Y_query = []
        for ref_point in self.clicked_ref:
            Y = self.before_ref_slide.get_tile(ref_point[1], ref_point[0], 1, 1).od(self.ref_I0).squeeze()
            Y_ref.append(Y)
        for query_point in self.clicked_query:
            Y = self.before_query_slide.get_tile(query_point[1], query_point[0], 1, 1).od(self.query_I0).squeeze()
            Y_query.append(Y)

        Y_ref = np.array(Y_ref).transpose()
        Y_query = np.array(Y_query).transpose()
        
        # Using the ref source style (vectors and maxC), calculate the C_desired values
        C_ref = nnls(self.ref_matrix, Y_ref) # (2,N)
        scaling_factor = np.divide(self.target_maxC, self.ref_maxC)[:, np.newaxis] # (2,1)
        C_desired = C_ref * scaling_factor # (2,N)

        # Divide C_desired by maxC of the target
        C_A = C_desired / self.target_maxC[:, np.newaxis] # (2,N)
        C_A = C_A.transpose() # (N,2)
        
        # Calculate initial guesses
        # Use the query source style (vectors and maxC)
        X1_0 = self.query_matrix[0,0] * self.query_maxC[0]
        X2_0 = self.query_matrix[1,0] * self.query_maxC[0]
        X3_0 = self.query_matrix[2,0] * self.query_maxC[0]
        Y1_0 = self.query_matrix[0,1] * self.query_maxC[1]
        Y2_0 = self.query_matrix[1,1] * self.query_maxC[1]
        Y3_0 = self.query_matrix[2,1] * self.query_maxC[1]

        eq1_0 = np.array([X1_0, Y1_0]) # (2,)
        eq2_0 = np.array([X2_0, Y2_0])
        eq3_0 = np.array([X3_0, Y3_0])

        Y_query = Y_query.transpose() # (N,3)
        y1 = Y_query[:,0] # (N,)
        y2 = Y_query[:,1]
        y3 = Y_query[:,2]

        # Perform least squares estimate with initial estimate
        def lstsq_fn(x, A, y, x0, lam):
            residual = A @ x - y
            regularized = np.sqrt(lam) * (x - x0)
            return np.concatenate([residual, regularized])

        eq1 = least_squares(lstsq_fn, eq1_0, args=(C_A, y1, eq1_0, lam))
        eq2 = least_squares(lstsq_fn, eq2_0, args=(C_A, y2, eq2_0, lam))
        eq3 = least_squares(lstsq_fn, eq3_0, args=(C_A, y3, eq3_0, lam))

        X1 = eq1.x[0]
        X2 = eq2.x[0]
        X3 = eq3.x[0]
        Y1 = eq1.x[1]
        Y2 = eq2.x[1]
        Y3 = eq3.x[1]

        H_scaled = np.array([X1, X2, X3])
        E_scaled = np.array([Y1, Y2, Y3])

        # Compute the magnitudes of the estimates
        maxC_H_new = np.linalg.norm(H_scaled, ord=2)
        maxC_E_new = np.linalg.norm(E_scaled, ord=2)

        H = H_scaled / maxC_H_new
        E = E_scaled / maxC_E_new

        # Store new stain matrix and maxC values
        self.matrix_new = np.array([H, E]).transpose() # (3,2)
        self.maxC_new = np.array([maxC_H_new, maxC_E_new]) # (2,)

        print("New query source matrix:")
        print(self.matrix_new)
        print("New query source maxC:")
        print(self.maxC_new)
        print("Cost values:", eq1.cost, eq2.cost, eq3.cost)

    def display_results(self, ref_y: int = None, ref_x: int = None, query_y: int = None, query_x: int = None, tile_size: int = None) -> plotly.graph_objects.Figure:
        """Display results of the correction.

        Args:
            ref_y: Upper left corner y-coordinate for the region of interest in the reference to show.
            ref_x: Upper left corner x-coordinate for the region of interest in the reference to show.
            query_y: Upper left corner y-coordinate for the region of interest in the query to show.
            query_x: Upper left corner x-coordinate for the region of interest in the query to show.
            tile_size: Pixel size of the regions of interest.

        Returns:
            A Plotly figure showing the results of correction.
        """
        if ref_y is None:
            ref_y = self.ref_y
        if ref_x is None:
            ref_x = self.ref_x
        if query_y is None:
            query_y = self.query_y
        if query_x is None:
            query_x = self.query_x
        if tile_size is None:
            tile_size = self.tile_size
        if self.matrix_new is None or self.maxC_new is None:
            raise Exception("Call compute first.")
            
        # Find original Y values of the tile
        query_tile_od = self.before_query_slide.get_tile(query_y, query_x, tile_size, tile_size).od(self.query_I0)
        h, w, _ = query_tile_od.shape
        
        # Perform stain normalization using those values
        query_tile_od = query_tile_od.reshape(-1,3).transpose() # (3,N)

        C = nnls(self.matrix_new, query_tile_od)

        scaling_factor = np.divide(self.target_maxC, self.maxC_new)[:, np.newaxis]
        C2 = C * scaling_factor

        Inorm = np.exp(-self.target_matrix.dot(C2))
        if self.query_post_adjust is not None:
            Inorm = self.query_post_adjust.apply(Inorm)
        Inorm = Inorm * self.target_I0[:, np.newaxis]
        Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)
        Inorm = np.reshape(Inorm.transpose(), (h, w, 3))


        ref_tile = self.after_ref_slide.get_tile(ref_y, ref_x, tile_size, tile_size).rgb(as_uint8=True)
        query_tile_norm = self.after_query_slide.get_tile(query_y, query_x, tile_size, tile_size).rgb(as_uint8=True)
        query_tile_orig = self.before_query_slide.get_tile(query_y, query_x, tile_size, tile_size).rgb(as_uint8=True)
        
        
        fig = make_subplots(rows=1, cols=4, subplot_titles=("Reference Norm", "Query Corrected Norm", "Query Initial Norm", "Query Before Norm"))

        fig.add_trace(
            go.Image(z=ref_tile, x0=ref_x, y0=ref_y), row=1, col=1
        )

        fig.add_trace(
            go.Image(z=Inorm, x0=query_x, y0=query_y), row=1, col=2
        )
        
        fig.add_trace(
            go.Image(z=query_tile_norm, x0=query_x, y0=query_y), row=1, col=3
        )

        fig.add_trace(
            go.Image(z=query_tile_orig, x0=query_x, y0=query_y), row=1, col=4
        )

        fig.update_layout(title=f"Correction results",
                         height=600,
                         width=1200,
                         )

        return fig
        