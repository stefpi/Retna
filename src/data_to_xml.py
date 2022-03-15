from xml.dom import minidom 
import base64
import os 
import numpy as np
import cv2

def main():
    eyes_path = 'src\Data_Eyes'
    files = os.listdir(eyes_path)
    save_path_file = "src\data.xml"

    root = minidom.Document()
    xml = root.createElement('Data')
    root.appendChild(xml)

    for file in files:
        image_np = cv2.imread(eyes_path + "\\" + file) # Get the np array of the image

        file = file.split('.')[0] # Deal with file type suffix
        coords = file.split('-') # Get list of coordinates

        productChild = root.createElement('Image')
        productChild.setAttribute('x', coords[0])
        productChild.setAttribute('y', coords[1])

        # Generate the bitmaps of the image
        bmp_b = root.createElement('BMP_B')
        bmp_g = root.createElement('BMP_G')
        bmp_r = root.createElement('BMP_R')

        # In order to go from the bitmap's array back to the og np array
        # Use the command new_im = numpy.concatenate((b, g, r), axis = 2)
        b, g, r = np.split(image_np, 3, axis=2)

        # Done to avoid an error
        b = np.ascontiguousarray(b)
        g = np.ascontiguousarray(g)
        r = np.ascontiguousarray(r)

        # Decoding the b64 encoding should be pretty basic
        bmp_b.setAttribute("Base64", base64.b64encode(b).decode('utf-8'))
        bmp_g.setAttribute("Base64", base64.b64encode(g).decode('utf-8'))
        bmp_r.setAttribute("Base64", base64.b64encode(r).decode('utf-8'))

        productChild.appendChild(bmp_b)
        productChild.appendChild(bmp_g)
        productChild.appendChild(bmp_r)

        xml.appendChild(productChild)

    xml_str = root.toprettyxml(indent='\t')

    with open(save_path_file, "w") as f:
        f.write(xml_str)

if __name__ == "__main__":
    main()
