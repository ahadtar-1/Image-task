"""
The module comprises of the APIs.
"""

from flask import Flask, request, jsonify, send_file
import cv2 as cv
import os
from image_tools import convert_coloredimage, convert_greyscaleimage, resize_image, blur_image, change_eyecolor

uploadsDir = "/home/ekkelai/Desktop/EkkelAI-TrainingTask2/Tempdata"

app = Flask(__name__)


@app.route('/convertcolored', methods = ['GET', 'POST'])
def convert_colored()-> str:
    """
    The api function that generates a grayscale image from a colored image

    Parameters
    ----------
    None

    Returns
    -------
    str
    
    """

    if request.method == 'POST':
        f = request.files['file']
        fileName = f.filename
        f.save(os.path.join(uploadsDir,fileName))
        path = convert_coloredimage(os.path.join(uploadsDir,fileName))
        if path:
            return send_file(path)
        else:
            return "No image"


@app.route('/eyecolorchange', methods = ['GET', 'POST'])
def eyecolor_change()-> str:
    """
    The api function that generates an image with eye color changed.

    Parameters
    ----------
    None

    Returns
    -------
    str
    
    """

    if request.method == 'POST':
        f = request.files['file']
        fileName = f.filename
        f.save(os.path.join(uploadsDir,fileName))
        path = change_eyecolor(os.path.join(uploadsDir,fileName))
        if path:
            return send_file(path)
        else:
            return "No image"


@app.route('/convertgreyscale', methods = ['GET', 'POST'])
def convert_greyscale()-> str:
    """
    The api function that generates a colored image from a greyscale image

    Parameters
    ----------
    None

    Returns
    -------
    str
    
    """

    if request.method == 'POST':
        f = request.files['file']
        fileName = f.filename
        f.save(os.path.join(uploadsDir,fileName))
        path = convert_greyscaleimage(os.path.join(uploadsDir,fileName))
        if path:
            return send_file(path)
        else:
            return "No image"


@app.route('/imageresize', methods = ['GET', 'POST'])
def image_resize()-> str:
    """
    The api function that generates a resized image

    Parameters
    ----------
    None

    Returns
    -------
    str
    
    """

    if request.method == 'POST':
        f = request.files['file']
        fileName = f.filename
        f.save(os.path.join(uploadsDir,fileName))
        path = resize_image(os.path.join(uploadsDir,fileName))
        if path:
            resizedImage = cv.imread(path)
            cv.imshow("coloredimage", resizedImage)
            cv.waitKey(15000)
            return "Image Resized"
        else:
            return "No image"


@app.route('/blurimage', methods = ['GET', 'POST'])
def image_blur()-> str:
    """
    The api function that generates a blurred image

    Parameters
    ----------
    None

    Returns
    -------
    str
    
    """

    if request.method == 'POST':
        f = request.files['file']
        fileName = f.filename
        f.save(os.path.join(uploadsDir,fileName))
        path = blur_image(os.path.join(uploadsDir,fileName))
        if path:
            return send_file(path)
        else:
            return "No image"
