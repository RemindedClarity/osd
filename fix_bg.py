from PIL import Image
import numpy as np

img = Image.open('osd_original.png').convert('RGBA')
data = np.array(img)

r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]

# Any pixel where R, G, and B are all below 30 = black = make transparent
mask = (r < 30) & (g < 30) & (b < 30)
data[:,:,3][mask] = 0

Image.fromarray(data).save('osd_character.png')
print('Done - osd_character.png saved with transparent background')
