import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
import os

st.title(':blue[Check-In System :school:]')

# directory/folder to face store
dir_path = r'./known_faces'

# Create arrays of known face encodings and their names
known_face_encodings = []

known_face_names = []

# Iterate directory & create face encodings with names
for file_path in os.listdir(dir_path):
    # check if current file_path is a file
    if os.path.isfile(os.path.join(dir_path, file_path)):
        if file_path.endswith( ('.jpeg','.png','.jpg')):
            img = face_recognition.load_image_file(dir_path + "/" +file_path)
            face_enc = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(face_enc)
            known_face_names.append(file_path.split(".")[0])

# print ("%d face encoded" % len(known_face_encodings))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

picture = st.camera_input("Take a picture")

if st.button('CheckIn'):
    if picture:

        # To read image file buffer as a PIL Image:
        img = Image.open(picture)
        # To convert PIL Image to numpy array:
        img = np.array(img)

        rgb_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_img2)
        # print("Found {} faces in image.".format(len(face_locations)))
        face_encodings = face_recognition.face_encodings(rgb_img2)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append("%s[%.2f]" % (name, face_distances[best_match_index]))

        if name != "Unknown":
            x = name.split("_")
            st.write(x[0] + " " + x[1] + " - Checked in :white_check_mark:")
        else:
            st.write(name + " person - Not Checked in :no_entry_sign:")
    else:
        st.error("Take a picture first!!")