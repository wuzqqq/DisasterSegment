from PIL import Image
import numpy as np

path = r'D:\DisasterSegment\test\targets\guatemala-volcano_00000003_post_disaster_target.png'

img = Image.fromarray(np.array(Image.open(path)) * 30)
img.show()