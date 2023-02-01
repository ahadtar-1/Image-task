"""
The module comprises of the functions to perform different operations on images. 
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import os
from flask import send_file, Response

uploadsDir = "/home/ekkelai/Desktop/EkkelAI-TrainingTask2/Tempdata"


def convert_coloredimage(filePath: str)-> None:
    """
    The function that converts a colored image to a  grey scale image.

    Parameters
    ----------
    str

    Returns
    -------
    str
    
    """

    myImage = cv.imread(filePath)
    newImage = cv.cvtColor(myImage, cv.COLOR_BGR2GRAY)
    os.chdir(uploadsDir)
    fileName = 'new.jpg'
    cv.imwrite(fileName, newImage)
    path = os.path.join(uploadsDir, fileName)
    return path


def convert_greyscaleimage(pathtoFile: str)-> None:
    """
    The function that converts a greyscale image to a colored image.

    Parameters
    ----------
    str

    Returns
    -------
    str
    
    """

    myImage = cv.imread(pathtoFile, cv.IMREAD_GRAYSCALE)
    newImage = cv.cvtColor(myImage, cv.COLOR_GRAY2RGB) * 255
    #finalImage = cv.merge([myImage, myImage, myImage])
    os.chdir(uploadsDir)
    fileName = 'colorednew.jpg'
    cv.imwrite(fileName, newImage)
    path = os.path.join(uploadsDir, fileName)
    return path


def resize_image(pathtoFile: str, newWidth: int)-> str:
    """
    The function that resizes an image.

    Parameters
    ----------
    str
    int

    Returns
    -------
    str
    
    """

    myImage = cv.imread(pathtoFile)
    os.chdir(uploadsDir)
    aspectRatio = float(myImage.shape[1] / myImage.shape[0])
    newHeight = int(newWidth / aspectRatio)
    newImage = cv.resize(myImage, (newWidth, newHeight), interpolation = cv.INTER_LINEAR)
    os.chdir(uploadsDir)
    fileName = 'resizedimg.jpg'
    cv.imwrite(fileName, newImage)
    path = os.path.join(uploadsDir, fileName)
    return path


def blur_image(pathtoFile: str)-> str:
    """
    The function that blurs an image.

    Parameters
    ----------
    str

    Returns
    -------
    str
    
    """

    myImage = cv.imread(pathtoFile)
    os.chdir(uploadsDir)
    blurredImage = cv.blur(myImage, (25,25))
    fileName = 'blurredimg.jpg'
    cv.imwrite(fileName, blurredImage)
    path = os.path.join(uploadsDir, fileName)
    return path


def enhance_image(pathtoFile: str)-> str:
    """
    The function that brightens and darkens image.

    Parameters
    ----------
    list

    Returns
    -------
    None
    
    """

    matrix = np.ones(myImage.shape, dtype = "uint8") * 50
    imgrgbBrighter = cv.add(myImage, matrix)
    imgrgbDarker = cv.subtract(myImage, matrix)
    cv.imshow("Bright Image", imgrgbBrighter)
    cv.imshow("Dark Image", imgrgbDarker)
    cv.waitKey(9000)


def count_faces(myImage: list)-> None:
    """
    The function that counts the number of faces in an image.

    Parameters
    ----------
    list

    Returns
    -------
    None

    """

    faceCascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    grayImage = cv.cvtColor(myImage, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImage, scaleFactor = 1.1, minNeighbors = 9, minSize = (30, 30))
    numFaces = len(faces)
    for (x, y, w, h) in faces:
        cv.rectangle(grayImage, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow("Faces found", grayImage)
    cv.waitKey(9000)
    print("Number of faces: ", numFaces)


def change_eyecolor(pathtoFile: str)-> None:
    """
    The function that changes the eye color of a person in an image.

    Parameters
    ----------
    str

    Returns
    -------
    None

    """

    myImage = cv.imread(pathtoFile)
    overlay = myImage.copy()
    mpfaceMesh = mp.solutions.face_mesh
    leftEye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398] # left eyes indices
    RightEye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246] # right eyes indices
    leftIris = [474,475, 476, 477] # left iris indices
    rightIris = [469, 470, 471, 472] # right iris indices
    with mpfaceMesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as faceMesh:
        rgbImage = cv.cvtColor(myImage, cv.COLOR_BGR2RGB)
        imgHeight, imgWidth = myImage.shape[:2]
        results = faceMesh.process(rgbImage)
        mask = np.zeros((imgHeight, imgWidth), dtype=np.uint8)

        if results.multi_face_landmarks:           
            meshPoints=np.array([np.multiply([p.x, p.y], [imgWidth, imgHeight]).astype(int) for p in results.multi_face_landmarks[0].landmark])           
            (lCx, lCy), lRadius = cv.minEnclosingCircle(meshPoints[leftIris])
            (rCx, rCy), rRadius = cv.minEnclosingCircle(meshPoints[rightIris])
            centerLeft = np.array([lCx, lCy], dtype=np.int32)
            centerRight = np.array([rCx, rCy], dtype=np.int32)
            cv.circle(myImage, centerLeft, int(lRadius), (255, 0, 0), -1, cv.LINE_AA)
            cv.circle(myImage, centerRight, int(rRadius), (255, 0, 0), -1, cv.LINE_AA)
            newImage = cv.addWeighted(overlay, 0.9, myImage, 1 - 0.89, 0)
            os.chdir(uploadsDir)
            fileName = "newimg.jpg"
            cv.imwrite(fileName, newImage)
            path = os.path.join(uploadsDir, fileName)
            return path


def count_objects(myImage: list)-> None:
    """
    The function that counts the number of objects in an image.

    Parameters
    ----------
    list

    Returns
    -------
    None

    """

    #gray = cv.cvtColor(myImage, cv.COLOR_BGR2GRAY)
    #blurred = cv.GaussianBlur(gray, (5, 5), 0)
    #params = cv.SimpleBlobDetector_Params()
    #detector = cv.SimpleBlobDetector_create(params)
    #keypoints = detector.detect(blurred)
    #img_with_blobs = cv.drawKeypoints(myImage, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #num_blobs = len(keypoints)
    #cv.imshow("Blobs", img_with_blobs)
    #print("Number of blobs:", num_blobs)
    #cv.waitKey(9000)
    #cv.destroyAllWindows()


def pose_estimation(myImage: list)-> None:
    """
    The api function that detects the pose of a person in an image.

    Parameters
    ----------
    list

    Returns
    -------
    None

    """

    #person_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_person.xml")
    #gray = cv.cvtColor(myImage, cv.COLOR_BGR2GRAY)
    #persons = person_cascade.detectMultiScale(gray, 1.3, 5)

    #for (x, y, w, h) in persons:
    #    person = myImage[y:y + h, x:x + w]
    
    #    net = cv.dnn.readNetFromCaffe("pose_deploy.prototxt", "pose_iter_440000.caffemodel")
    #    inWidth = 368
    #    inHeight = 368
    #    inpBlob = cv.dnn.blobFromImage(person, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    #    net.setInput(inpBlob)
    #    output = net.forward()
    
    #    hips = output[0, 56, 0]
    #    knees = output[0, 57, 0]
    #    ankles = output[0, 58, 0]
    
    #    if hips[1] > ankles[1] and knees[1] > ankles[1]:
    #        print("Person is sitting")
    #    else:
    #        print("Person is standing")
