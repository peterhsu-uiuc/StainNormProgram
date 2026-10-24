import stainnormprogram.norm.background as background

import numpy as np
from scipy.ndimage import gaussian_filter

class ColorSample:
    """ColorSample creates color histograms of the slide.
    """

    def __init__(self,
                 slide,
                 rgb_bins=(51,51,51),
                 h_bins=200,
                 s_bins=100,
                 v_bins=100,
                 oklab_bins=(200,160,160),
                 oklab_a_range=(-0.4,0.4),
                 oklab_b_range=(-0.4,0.4),
                 uv_bins=(100,100),
                 uv_u_range=(0.0,0.6),
                 uv_v_range=(0.0,0.6),
                 background_method="hsv_threshold",
                 sampling_method="random",
                 max_pixels=10_000_000,
                 tile_size=256,
                 identifier=None,
        ):
        
        self.slide = slide
        self.rgb_bins = rgb_bins
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.oklab_bins = oklab_bins
        self.oklab_a_range = oklab_a_range
        self.oklab_b_range = oklab_b_range
        self.uv_bins = uv_bins
        self.uv_u_range = uv_u_range
        self.uv_v_range = uv_v_range
        self.background_method = background_method
        self.sampling_method = sampling_method
        self.max_pixels = max_pixels
        self.identifier = identifier if identifier is not None else self.slide.identifier
        
        # Setup histograms
        self.rgb_hist = np.zeros(rgb_bins, dtype=np.uint64)
        self.h_hist = np.zeros((h_bins,), dtype=np.uint64)
        self.s_hist = np.zeros((s_bins,), dtype=np.uint64)
        self.v_hist = np.zeros((v_bins,), dtype=np.uint64)
        self.oklab_hist = np.zeros(oklab_bins, dtype=np.uint64)
        self.uv_hist = np.zeros(uv_bins, dtype=np.uint64)

        # Edges
        self.hsv_h_hist_edges = np.linspace(0, 1, num=self.h_bins+1)
        self.hsv_s_hist_edges = np.linspace(0, 1, num=self.s_bins+1)
        self.hsv_v_hist_edges = np.linspace(0, 1, num=self.v_bins+1)
        self.rgb_r_hist_edges = np.linspace(0, 255, num=self.rgb_bins[0]+1)
        self.rgb_g_hist_edges = np.linspace(0, 255, num=self.rgb_bins[1]+1)
        self.rgb_b_hist_edges = np.linspace(0, 255, num=self.rgb_bins[2]+1)
        self.oklab_l_hist_edges = np.linspace(0, 1, num=self.oklab_bins[0]+1)
        self.oklab_a_hist_edges = np.linspace(self.oklab_a_range[0], self.oklab_a_range[1], num=self.oklab_bins[1]+1)
        self.oklab_b_hist_edges = np.linspace(self.oklab_b_range[0], self.oklab_b_range[1], num=self.oklab_bins[2]+1)
        self.uv_u_hist_edges = np.linspace(self.uv_u_range[0], self.uv_u_range[1], num=self.uv_bins[0]+1)
        self.uv_v_hist_edges = np.linspace(self.uv_v_range[0], self.uv_v_range[1], num=self.uv_bins[1]+1)

        # Counters
        self.num_tiles = 0
        self.num_pixels = 0

        # Tile size
        self.slide.set_tile_size(tile_size)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["slide"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.slide = None

    def get_h_hist(self, density=False):

        if density:
            out = self.h_hist / np.sum(self.h_hist)
            return out

        return self.h_hist

    def get_s_hist(self, density=False):

        if density:
            out = self.s_hist / np.sum(self.s_hist)
            return out

        return self.s_hist

    def get_v_hist(self, density=False):

        if density:
            out = self.v_hist / np.sum(self.v_hist)
            return out

        return self.v_hist

    def get_rgb_hist(self, density=False):

        if density:
            out = self.rgb_hist / np.sum(self.rgb_hist)
            return out

        return self.rgb_hist

    def get_oklab_hist(self, density=False):

        if density:
            out = self.oklab_hist / np.sum(self.oklab_hist)
            return out

        return self.oklab_hist

    def get_uv_hist(self, density=False):

        if density:
            out = self.uv_hist / np.sum(self.uv_hist)
            return out

        return self.uv_hist

    def compute_tile(self, tile):

        if self.background_method == "hsv_threshold":
            fg_mask = background.hsv_threshold_foreground(tile).flatten()
        else:
            raise Exception(f"Background method {self.background_method} not supported.")

        n = np.sum(fg_mask)
        if n == 0:
            return

        if self.num_tiles % 10 == 0:
            print(f"Processed {self.num_tiles} tiles so far...")

        rgb_tile = tile.rgb(as_uint8=True).reshape((-1,3))
        hsv_tile = tile.hsv().reshape((-1,3))
        oklab_tile = tile.oklab().reshape((-1,3))
        uv_u_tile, uv_v_tile = tile.u_prime_v_prime()
        uv_u_tile = uv_u_tile.flatten()
        uv_v_tile = uv_v_tile.flatten()

        rgb = rgb_tile[fg_mask, :]
        hsv_h = hsv_tile[:,0][fg_mask]
        hsv_s = hsv_tile[:,1][fg_mask]
        hsv_v = hsv_tile[:,2][fg_mask]
        oklab = oklab_tile[fg_mask, :]
        uv_u = uv_u_tile[fg_mask]
        uv_v = uv_v_tile[fg_mask]
        uv_not_nan = ~(np.isnan(uv_u) | np.isnan(uv_v))
        uv_u = uv_u[uv_not_nan]
        uv_v = uv_v[uv_not_nan]
        
        self.h_hist += np.histogram(hsv_h, bins=self.hsv_h_hist_edges)[0].astype(np.uint64)
        self.s_hist += np.histogram(hsv_s, bins=self.hsv_s_hist_edges)[0].astype(np.uint64)
        self.v_hist += np.histogram(hsv_v, bins=self.hsv_v_hist_edges)[0].astype(np.uint64)

        r_idx = (rgb[:,0] / 255 * (self.rgb_bins[0] - 1)).astype(int)
        g_idx = (rgb[:,1] / 255 * (self.rgb_bins[1] - 1)).astype(int)
        b_idx = (rgb[:,2] / 255 * (self.rgb_bins[2] - 1)).astype(int)

        np.add.at(self.rgb_hist, (r_idx, g_idx, b_idx), 1)

        l_idx = (oklab[:,0] * (self.oklab_bins[0] - 1)).astype(int) 
        a_idx = ((oklab[:,1] - self.oklab_a_range[0]) / (self.oklab_a_range[1] - self.oklab_a_range[0]) * (self.oklab_bins[1] - 1)).astype(int)
        b_idx = ((oklab[:,2] - self.oklab_b_range[0]) / (self.oklab_b_range[1] - self.oklab_b_range[0]) * (self.oklab_bins[2] - 1)).astype(int)
        
        np.add.at(self.oklab_hist, (l_idx, a_idx, b_idx), 1)

        u_idx = ((uv_u - self.uv_u_range[0]) / (self.uv_u_range[1] - self.uv_u_range[0]) * (self.uv_bins[0] - 1)).astype(int)
        v_idx = ((uv_v - self.uv_v_range[0]) / (self.uv_v_range[1] - self.uv_v_range[0]) * (self.uv_bins[1] - 1)).astype(int)

        np.add.at(self.uv_hist, (u_idx, v_idx), 1)

        self.num_tiles = self.num_tiles + 1
        self.num_pixels = self.num_pixels + n
    
    def compute(self):

        print(f"Computing ColorSample for {self.identifier}...")

        self.slide.set_mode(self.sampling_method)
        for tile, _ in self.slide:
            self.compute_tile(tile)
            if self.num_pixels > self.max_pixels:
                break
            
        print("Finished computing ColorSample")
    

class ColorDistribution:

    def __init__(self, samples, identifier=None, equally_weighted=False):
        
        self.samples = samples
        if identifier is not None:
            self.identifier = identifier
        elif isinstance(samples, ColorSample):
            self.identifier = samples.identifier
        elif (isinstance(samples, list) and len(samples) == 1):
            self.identifier = samples[0].identifier
        else:
            raise Exception("Specify an identifier for this ColorDistribution.")
        self.equally_weighted = equally_weighted

        self.set_bins_ranges(samples)

        # Setup histograms
        hist_dtype = np.float64 if self.equally_weighted else np.uint64
        self.rgb_hist = np.zeros(self.rgb_bins, dtype=hist_dtype)
        self.h_hist = np.zeros((self.h_bins,), dtype=hist_dtype)
        self.s_hist = np.zeros((self.s_bins,), dtype=hist_dtype)
        self.v_hist = np.zeros((self.v_bins,), dtype=hist_dtype)
        self.oklab_hist = np.zeros(self.oklab_bins, dtype=hist_dtype)
        self.uv_hist = np.zeros(self.uv_bins, dtype=hist_dtype)

        # Edges
        self.hsv_h_hist_edges = np.linspace(0, 1, num=self.h_bins+1)
        self.hsv_s_hist_edges = np.linspace(0, 1, num=self.s_bins+1)
        self.hsv_v_hist_edges = np.linspace(0, 1, num=self.v_bins+1)
        self.rgb_r_hist_edges = np.linspace(0, 255, num=self.rgb_bins[0]+1)
        self.rgb_g_hist_edges = np.linspace(0, 255, num=self.rgb_bins[1]+1)
        self.rgb_b_hist_edges = np.linspace(0, 255, num=self.rgb_bins[2]+1)
        self.oklab_l_hist_edges = np.linspace(0, 1, num=self.oklab_bins[0]+1)
        self.oklab_a_hist_edges = np.linspace(self.oklab_a_range[0], self.oklab_a_range[1], num=self.oklab_bins[1]+1)
        self.oklab_b_hist_edges = np.linspace(self.oklab_b_range[0], self.oklab_b_range[1], num=self.oklab_bins[2]+1)
        self.uv_u_hist_edges = np.linspace(self.uv_u_range[0], self.uv_u_range[1], num=self.uv_bins[0]+1)
        self.uv_v_hist_edges = np.linspace(self.uv_v_range[0], self.uv_v_range[1], num=self.uv_bins[1]+1)

    def set_bins_ranges(self, samples):
        
        if isinstance(samples, list):
            init_sample = samples[0]
        if isinstance(samples, ColorSample):
            init_sample = samples

        self.rgb_bins = init_sample.rgb_bins
        self.h_bins = init_sample.h_bins
        self.s_bins = init_sample.s_bins
        self.v_bins = init_sample.v_bins
        self.oklab_bins = init_sample.oklab_bins
        self.oklab_a_range = init_sample.oklab_a_range
        self.oklab_b_range = init_sample.oklab_b_range
        self.uv_bins = init_sample.uv_bins
        self.uv_u_range = init_sample.uv_u_range
        self.uv_v_range = init_sample.uv_v_range

        if isinstance(samples, list):
            for sample in samples:
                if (sample.rgb_bins != self.rgb_bins
                    or sample.h_bins != self.h_bins
                    or sample.s_bins != self.s_bins
                    or sample.v_bins != self.v_bins
                    or sample.oklab_bins != self.oklab_bins
                    or sample.oklab_a_range != self.oklab_a_range
                    or sample.oklab_b_range != self.oklab_b_range
                    or sample.uv_u_range != self.uv_u_range
                    or sample.uv_v_range != self.uv_v_range
                   ):
                    
                    raise Exception("The bin counts and ranges must match for all ColorSamples.")

    def get_h_hist(self):
        return self.h_hist

    def get_s_hist(self):
        return self.s_hist

    def get_v_hist(self):
        return self.v_hist

    def get_rgb_hist(self):
        return self.rgb_hist

    def get_oklab_hist(self):
        return self.oklab_hist

    def get_uv_hist(self):
        return self.uv_hist

    def compute(self):

        samples = self.samples
        if isinstance(self.samples, ColorSample):
            samples = [self.samples]
        
        for sample in samples:

            self.h_hist += sample.get_h_hist(density=self.equally_weighted)
            self.s_hist += sample.get_s_hist(density=self.equally_weighted)
            self.v_hist += sample.get_v_hist(density=self.equally_weighted)
            self.rgb_hist += sample.get_rgb_hist(density=self.equally_weighted)
            self.oklab_hist += sample.get_oklab_hist(density=self.equally_weighted)
            self.uv_hist += sample.get_uv_hist(density=self.equally_weighted)
                
        self.h_hist = self.h_hist / np.sum(self.h_hist)
        self.s_hist = self.s_hist / np.sum(self.s_hist)
        self.v_hist = self.v_hist / np.sum(self.v_hist)
        self.rgb_hist = self.rgb_hist / np.sum(self.rgb_hist)
        self.oklab_hist = self.oklab_hist / np.sum(self.oklab_hist)
        self.uv_hist = self.uv_hist / np.sum(self.uv_hist)

        # Smooth
        self.h_hist = gaussian_filter(self.h_hist, sigma=2.0)
        self.h_hist = self.h_hist / np.sum(self.h_hist)

        self.s_hist = gaussian_filter(self.s_hist, sigma=2.0)
        self.s_hist = self.s_hist / np.sum(self.s_hist)

        self.v_hist = gaussian_filter(self.v_hist, sigma=2.0)
        self.v_hist = self.v_hist / np.sum(self.v_hist)

        self.rgb_hist = gaussian_filter(self.rgb_hist, sigma=0.5)
        self.rgb_hist = self.rgb_hist / np.sum(self.rgb_hist)

        self.oklab_hist = gaussian_filter(self.oklab_hist, sigma=0.5)
        self.oklab_hist = self.oklab_hist / np.sum(self.oklab_hist)

        self.uv_hist = gaussian_filter(self.uv_hist, sigma=0.5)
        self.uv_hist = self.uv_hist / np.sum(self.uv_hist)



class ColorComparison:

    def __init__(self, reference, query):
        self.reference = reference
        self.query = query

        self.h_dist = None
        self.s_dist = None
        self.v_dist = None
        self.rgb_dist = None
        self.oklab_dist = None

    def get_val(self, val_type):
        if val_type == "h":
            return self.h_dist
        elif val_type == "s":
            return self.s_dist
        elif val_type == "v":
            return self.v_dist
        elif val_type == "rgb":
            return self.rgb_dist
        elif val_type == "oklab":
            return self.oklab_dist
        else:
            raise Exception(f"{val_type} is not a valid value type.")

    @property
    def query_identifier(self):
        return self.query.identifier

    @property
    def reference_identifier(self):
        return self.reference.identifier

    def dist_metric(self, reference_hist, query_hist):
        return self.bhattacharyya(reference_hist, query_hist)

    def bhattacharyya(self, reference_hist, query_hist):
        BC = np.sum(np.sqrt(reference_hist * query_hist))
        Db = -1.0 * np.log(BC)
        return Db

    def compute(self):
        
        print(f"Comparing {self.query.identifier} to {self.reference.identifier}...")
        
        self.h_dist = self.dist_metric(self.reference.get_h_hist(), self.query.get_h_hist(density=True))

        self.s_dist = self.dist_metric(self.reference.get_s_hist(), self.query.get_s_hist(density=True))

        self.v_dist = self.dist_metric(self.reference.get_v_hist(), self.query.get_v_hist(density=True))

        self.rgb_dist = self.dist_metric(self.reference.get_rgb_hist(), self.query.get_rgb_hist(density=True))

        self.oklab_dist = self.dist_metric(self.reference.get_oklab_hist(), self.query.get_oklab_hist(density=True))


