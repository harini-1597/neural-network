import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def process_image_to_dataframe(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img_resized = cv2.resize(img, (28, 28))
    _, img_binary = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY)
    img_normalized = img_binary / 255.0
    img_flattened = img_normalized.flatten()
    df = pd.DataFrame([img_flattened])
    return df

def display_prediction(model, df):
    df = np.array(df)
    predictions = model.predict(df.reshape(1, 28 * 28))
    predicted_label = np.argmax(predictions)
    print(f"Digit Predicted: {predicted_label}")

# image_path = 'img_6.png'
# img = cv2.imread(image_path,0)
# df = process_image_to_dataframe(image_path)
# # print(df.shape)  

model = load_model('digit_classifier.h5')
# cv2.imshow('image',img)
# display_prediction(model,df)
# cv2.waitKey(0)

cap = cv2.VideoCapture(0) 

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Webcam', gray_frame)
        df = process_image_to_dataframe(gray_frame)
        display_prediction(model, df)
        
        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
