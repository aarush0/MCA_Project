# MCA_Project
# Audio and Textual Analysis:

All the pre-extracted features have been saved in the folder ./MELD.Features.Models/features. These features are in pickle files with can be imported in our code. \
Copy these features into ./MELD.Features.Models/pickles/ or directly use them from features folder by changing the paths in code. \
We can specify the type of modality we want to use in the init section in baseline.py \
We can also use the pre made models which are saved in the folder ./MELD.Features.Models/models. Add the path of the directory in the code. \
Run baseline.py to get the results.

# Video Analysis:

You can use detect.py file to extract the faces from the video. \
We have already extracted the features using emotion_detection.py. We have save the extracted features using pickle files. \
You can directly run the code with the help of the pre extracted features using run_svm.py

