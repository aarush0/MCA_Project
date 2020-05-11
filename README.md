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

# VGG MoreBalanced.ipynb
This is the code for the video modality involvong the VGG. We have use the balanced dataset in this case. We use LSTM in this case.

# Pickle files
1. Audio_model.h5 - model for audio modality 
2. Text_model.h5 - model for text modality \
3. Bimodal_model.h5 - model for bimodal modality \
4. Video_model.h5 - model for video modality \
5. Test_text_mask.pkl - mask test for text \ 
6. Test_text_x.pkl - x test for text \
7. Test_text_y.pkl - y test for text \
8. Test_audio_mask.pkl - mask test for audio \ 
9. Test_audio_x.pkl - x test for audio \
10. Test_audio_y.pkl - y test for audio \
11. Test_video_mask.pkl - mask test for video \ 
12. Test_video_x.pkl - x test for video \
13. Test_video_y.pkl - y test for video \
14. Test_bimodal_mask.pkl - mask test for bimodal \ 
15. Test_bimodal_x.pkl - x test for bimodal \
16. Test_bimodal_y.pkl - y test for bimodal \

# Data
Link to all pickle files curated by us: https://drive.google.com/open?id=1rLmKQBN4QOtRgJY5ag2f-R2_g4UYyOVb

# Original Data
Link to original MELD features : https://drive.google.com/drive/folders/1y4nj9rBMHyEvfLNMcpoKsXm9cGdAcZpy
