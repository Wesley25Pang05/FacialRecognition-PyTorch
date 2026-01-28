I created a facial recognition system with TensorFlow and deep learning. I implemented face detection and alignment, and extracted a previously trained CNN for facial embeddings and identities using an SVM. The main areas of the project were data cleaning, model inference, and the total machine learning workflow.

**Files:**

train.py - Using photos from the data/train folder, it creates a pkl file containing a trained face recognition model to differentiate between those faces. Currently, it is using a pre-trained face detection model.
test-cam.py - Needs a pkl file from the train.py file. When run, it opens your camera to locate faces and draw a box around them. It then shows the confidence and name of the person it is closest to from the recognition model. If the person in the camera does not match one of the individuals in the files, it will be classified as Unknown, meaning it has less than 85% confidence.
test-folder.py - Needs a pkl file from the train.py file. When run, it opens the data/test folder. It will edit each valid image, either .jpg, .jpeg, or .png, to match a face in the recognition model. It uses the format: Name_Confidence_OrderOfEditting, so if the first image edited was Pang with a confidence of 0.98 it will say Pang_0.98_1.

classifier.py - Trains a classifier using sklearn and saves the model as face_svm.pkl.
detect.py - Uses a trained face detection model to locate where all the faces are using Pillow, Torch and numpy.
embeds.py - Uses PyTorch, pretrained models to get the embedding with tensor.

**How to use**
In progress
