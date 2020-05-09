# MCA_Project
# Audio and Textual Analysis:

All the pre-extracted features have been saved in the folder ./MELD.Features.Models/features. These features are in pickle files with can be imported in our code. \
Copy these features into ./MELD.Features.Models/pickles/ or directly use them from features folder by changing the paths in code. \
We can specify the type of modality we want to use in the init section in baseline.py \
We can also use the pre made models which are saved in the folder ./MELD.Features.Models/models. Add the path of the directory in the code. \
Run baseline.py to get the results.

# Video Analysis:

You can use detect.py file to extract the faces from the video. \
We have already extracted the features using emotion_detection.py. We have saved the extracted features using pickle files. \
You can directly run the code with the help of the pre extracted features using run_svm.py

# MCA Original.ipynb
This file contains the code for the baseline MELD paper[12], which is executed on the balanced dataset.
For excuting the file load all the components of ipynb file.
Run the main cell and change the modality (text/audio) accordingly.

# train_frames_make.py, train_frames_filter.py, train_frames_join.py
Files to preprocess videos, and convert videos into frames, and store them as np arrays.

# mca_vgg.py
Use VGG to extract features from frames and trains a NN to classify them into 4 emotions

# MCA_Project_Audio_Text_Attention.ipynb
This is the file that runs the text audio and bimodal modality with attention layer implemented. 
You need to run the main cell and remove the comment for the specific modality that you need to run.
This code work for tensorflow 1.1x version only.
The dataset has been balanced to make the quanities of all emotions equal.
