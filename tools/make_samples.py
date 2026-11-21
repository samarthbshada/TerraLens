"""Create synthetic demo image pairs in ./samples (NOT real satellite data)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synthetic import make_pair, write_png, write_rgbn_tiff

OUT = os.path.join(os.path.dirname(HERE), "samples")
os.makedirs(OUT, exist_ok=True)
t1, t2, truth = make_pair(512, 7, with_nir=True)
write_png(os.path.join(OUT, "demo_T1_rgb.png"), t1)
write_png(os.path.join(OUT, "demo_T2_rgb.png"), t2)
write_rgbn_tiff(os.path.join(OUT, "demo_T1_rgbn.tif"), t1, 10.0)
write_rgbn_tiff(os.path.join(OUT, "demo_T2_rgbn.tif"), t2, 10.0)
print("samples written to", OUT)
