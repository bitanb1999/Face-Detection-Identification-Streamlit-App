import cv2
import face_recognition
import streamlit as st
import tempfile
import os
import shutil
from zipfile import ZipFile
import base64
import numpy as np
from PIL import Image


def get_binary_file_downloader_html(bin_file, label='Download', button_text='Download'):
    """
    Generates HTML code for a download link of a binary file.

    Parameters:
    bin_file (str): Path to the binary file to be downloaded.
    label (str, optional): Label to display before the download link (default is 'Download').
    button_text (str, optional): Text to display on the download button (default is 'Download').

    Returns:
    str: HTML code for a download link that allows users to download the specified binary file.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{button_text}</a>'
    return href



def main():
    st.title('Face Detection and Identification App')
    st.sidebar.title('Face Detection and Identification App')
    st.sidebar.subheader('Parameters')

    # Create a space for displaying the original video
    video_container = st.empty()

    detection_confidence = st.sidebar.slider('Face Similarity Threshold (lower = strict comparison)', min_value=0.1, max_value=0.9, value=0.3)
    model_selection = st.sidebar.selectbox('Model Selection (cnn = more accurate but slow)', options=["hog", "cnn"])

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4"]) #other file formats if needed can be included
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        st.sidebar.warning("Please upload a video in mp4 format.")
        return  # Stop further execution if no video is selected

    try:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
        if not vid.isOpened():
            st.error("Error opening video file.")
            return
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

    st.markdown('## Detected Face')
    stframe = st.empty()

    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)

    # Create a directory to store extracted face images
    faces_directory = tempfile.mkdtemp()

    known_face_encodings = []
    face_names = []
    # Display the original video
    #video_container.video(vid.read())
    ii=0  #to store face number
    while vid.isOpened():
        ret, image = vid.read()
        if not ret:
            break
        
        video_container.image(image, caption= "Original Video", channels="BGR", use_column_width=True)
        if model_selection == 'hog':
            face_locations = face_recognition.face_locations(image, model='hog')
        elif model_selection == 'cnn':   #usually more accurate but time consuming
            face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0,model='cnn')

        if len(face_locations) > 0:
            for idx, (top, right, bottom, left) in enumerate(face_locations):
                face_image = np.ascontiguousarray(image[top:bottom, left:right])
                face_encodings = face_recognition.face_encodings(face_image)
                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    # Save the detected face image
                    face_image_path = os.path.join(faces_directory, f"face_{ii}.jpg")
                    cv2.imwrite(face_image_path, cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

                    known_face_encodings.append(face_encoding)
                    face_names.append(f"Face {ii}")

                    stframe.image(face_image, caption=f"Face {ii}", use_column_width=True)
                    ii+=1

    vid.release()
    video_container.empty()
    stframe.empty()


    # List to store face image filenames
    face_filenames = [filename for filename in os.listdir(faces_directory) if filename.startswith('face_')]

    # Create a dictionary to store face encodings
    face_encodings = {}

    for face_filename in face_filenames:
        # Load each face image
        face_image = face_recognition.load_image_file(os.path.join(faces_directory, face_filename))
        
        # Compute face encodings
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding) > 0:
            face_encodings[face_filename] = face_encoding[0]

    try:
        # Create a temporary directory to store grouped faces
        with tempfile.TemporaryDirectory() as grouped_faces_directory:
            # Create a dictionary to store face groups
            face_groups = {}
            for face_filename, _ in face_encodings.items():
                face_groups[face_filename] = face_filename
            # Calculate distances between all face pairs
            ITERATED=[]
            for face_filename1, face_encoding1 in face_encodings.items():
                for face_filename2, face_encoding2 in face_encodings.items():
                    if face_filename1 != face_filename2 and face_filename2 not in ITERATED :
                        distance = face_recognition.face_distance([face_encoding1], face_encoding2)
                        if distance <= detection_confidence: # Threshold for considering faces as similar
                            
                            # Add face to a group
                            group_id1 = face_groups[face_filename1]
                            group_id2 = face_groups[face_filename2]
                            if group_id1 != group_id2:
                                # Merge groups
                                group_ids = [group_id1, group_id2]
                                merged_group_id = min(group_ids)
                                for face_filename, group_id in face_groups.items():
                                    if group_id in group_ids:
                                        face_groups[face_filename] = merged_group_id
                ITERATED.append(face_filename1)
        

            # Move grouped faces to their respective directories
            for face_filename, group_id in face_groups.items():
                group_directory = os.path.join(grouped_faces_directory, f'Group_{group_id[:-4]}')
                os.makedirs(group_directory, exist_ok=True)

                source_path = os.path.join(faces_directory, face_filename)
                target_path = os.path.join(group_directory, face_filename)
                shutil.move(source_path, target_path)

            grouped_images_dict = {}
            with ZipFile('grouped_faces.zip', 'w') as zipf:
                for root, dirs, _ in os.walk(grouped_faces_directory):
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        group_images =[]
                        for file_name in os.listdir(dir_path):
                            file_path = os.path.join(dir_path, file_name)
                            group_images.append(file_path)
                            zipf.write(file_path, os.path.relpath(file_path, grouped_faces_directory))
                        grouped_images_dict[dir_name] = group_images

            # Display the total number of groups
            st.write(f"Total Number of Groups: {len(grouped_images_dict)}")

            # Display images within the same group together
            for group_name, images_in_group in grouped_images_dict.items():
                st.write(f"{group_name} - Number of Images: {len(images_in_group)}")
                
                # Create five streamlit columns for displaying images side by side
                columns = st.columns(5)  # Create five side-by-side columns
                
                for index, image_path in enumerate(images_in_group):
                    # Display images using an absolute path
                    image = Image.open(image_path)
                    
                    # Place images in respective columns, cycling through columns
                    columns[index % 5].image(image, caption=f"Image: {os.path.basename(image_path)}", use_column_width=True)
    except Exception as e:
        st.error(f"An error occurred during identity recognition: {str(e)}")

    # Display the download link for the zip file
    st.sidebar.markdown(get_binary_file_downloader_html('grouped_faces.zip', 'Download Grouped Extracted Faces', button_text="Download Grouped Extracted Faces"), unsafe_allow_html=True)

    # Clean up
    # Check if the grouped_faces_directory exists
    if os.path.exists(grouped_faces_directory):
        # Remove the grouped_faces_directory and its contents
        shutil.rmtree(grouped_faces_directory)


if __name__ == '__main__':
    main()
