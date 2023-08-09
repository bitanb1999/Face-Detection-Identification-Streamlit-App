# Face-Detection-Identification-Streamlit-App
This project performs face detection and identification using the Streamlit library and the face_recognition library. It takes input from a video source, detects faces in each frame, and groups similar faces together. The grouped faces are displayed in a Streamlit UI.


## Table of Contents
- [Overview](#overview)
- [Installation and Usage](#installation)
- [Streamlit Deployment](#streamlit-deployment)

## Overview

The Face Detection and Identification project is a web application built using the face_recognition library and the Streamlit framework. It enables users to upload a video, perform face detection, and identify similar faces, grouping them together. The application displays the original video with detected and grouped faces in an interactive and user-friendly interface.
### Key Features
1. Video Upload and Display: Users can upload a video in the mp4 format through the web application's user interface. The original video is displayed in the application, allowing users to see the faces present in the video.

2. Face Detection: The application uses the face_recognition library to detect faces in the uploaded video frames. Two detection models are available: HOG (Histogram of Oriented Gradients) and CNN (Convolutional Neural Network). Users can choose between these models to optimize either speed or accuracy.

3. Face Identification and Grouping: Detected faces that are similar to each other are grouped together. This grouping is based on calculating the distance between face encodings. Faces with a similarity above a specified threshold are considered part of the same group.

4. Interactive Display: The application provides an interactive interface that displays the original video along with detected faces. Grouped faces are displayed together, making it easy for users to identify and compare similar faces.

5. Download Grouped Faces: Users can download the grouped faces as a zip file. The zip file contains directories for each group, with images of the faces in that group.

6. Error Handling: The application includes proper error handling mechanisms to manage issues such as incorrect video formats, face detection failures, and errors during face identification.

### Streamlit UI

![image](https://github.com/bijayshah726/Face-Detection-Identification-Streamlit-App/assets/89373352/4d4439ce-6b9a-4178-8115-df4fd338aae8)

### Output from demo video
![image](https://github.com/bijayshah726/Face-Detection-Identification-Streamlit-App/assets/89373352/1617e0b5-a6eb-4fd0-8bec-65ce7587e212)
![image](https://github.com/bijayshah726/Face-Detection-Identification-Streamlit-App/assets/89373352/7ed5d3b6-6326-4f6d-85f3-58fd8cb3b92d)



## Installation and Usage

### Web App
   You can directly view the app and use it for prediction and visualization.
   https://face-detection-identification.streamlit.app/

### Local Machine
   #### Prerequisites

- An Operating System like Windows, OsX or Linux
- A working [Python](https://www.python.org/) installation.
- a Shell
  - [Git Bash](https://git-scm.com/downloads) for Windows 8.1
  - [wsl](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) for For Windows 10
- an Editor
  - [VS Code](https://code.visualstudio.com/) (Preferred) or [PyCharm](https://www.jetbrains.com/pycharm/).
- [Git cli](https://git-scm.com/downloads)

#### Installation

Clone the repo

```bash
git clone https://github.com/bijayshah726/Face-Detection-Identification-Streamlit-App.git
```

cd into the project root folder

```bash
cd Face-Detection-Identification-Streamlit-App
```

##### Create virtual environment

###### via python

Then you should create a virtual environment named .venv

```bash
python -m venv .venv
```

and activate the environment.

On Linux, OsX or in a Windows Git Bash terminal it's

```bash
source .venv/Scripts/activate
```

or alternatively

```bash
source .venv/bin/activate
```

In a Windows terminal it's

```bash
.venv/Scripts/activate.bat
```

###### or via anaconda

Create virtual environment named MNIST-app

```bash
conda create -n Face-Detection-Identification-Streamlit-App python3
```

and activate environment.

```bash
activate Face-Detection-Identification-Streamlit-App
```

If you are on windows you need to install some things required by GeoPandas by following [these instructions](https://geoffboeing.com/2014/09/using-geopandas-windows/).

Then you should install the local requirements

```bash
pip install -r requirements.txt
```

#### Build and run the Application Locally

```bash
streamlit run app.py  #opens the default browser where the streamlit app can be viewed
#Upload a video with multiple faces to see the grouping effect
```


  


## Streamlit Deployment
Sign up or log in to Streamlit Sharing.

Fork or clone this GitHub repository to your GitHub account.

Connect your GitHub repository to Streamlit Sharing.

Deploy the app on Streamlit Sharing (more details [here](https://docs.streamlit.io/streamlit-community-cloud/share-your-app))


Demo Video from [Pexels](https://www.pexels.com/video/people-walking-in-the-corridor-with-coffee-while-having-a-break-3254006/)

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests if you encounter any problems or have improvements to suggest.
   
