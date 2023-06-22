# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:45:00 2020

@author: Buria Alabau Bosque
"""

from PIL import Image

# Open Paddington
img = Image.open("./Images/imagen.jpg")

# Resize smoothly down to 16x16 pixels
imgSmall = img.resize((20,20),resample=Image.BILINEAR)

# Scale back up using NEAREST to original size
result = imgSmall.resize(img.size,Image.NEAREST)

# Save
result.save('Madeline3.png')