import cv2
import numpy as np
from PIL import Image
import os

if __name__ == "__main__":
    path = 'images'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print('\n[INFO] Training...')
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    def get_images_and_labels(path):
        """
        Load face images and corresponding labels from the given directory path.

        Args:
            path (str): Directory path containing face images.

        Returns:
            list: List of face samples.
            list: List of corresponding labels
        """
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []

        for image_path in image_paths:
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(image_path)[-1].split('-')[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

        return face_samples, ids


    faces, ids = get_images_and_labels(path)

    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer.yml')

    print('\n[INFO] {0} faces trained. Exiting Program'.format(len(np.unique(ids))))