from sklearn.svm import SVC
import joblib

def train_svm(embeddings, labels, save_path="face_svm.pkl"):
    classifier = SVC(kernel='linear', probability=True) # Using a model for machine learning
    classifier.fit(embeddings, labels) # Classifier trainer
    joblib.dump(classifier, save_path) # Saves the model
    return classifier