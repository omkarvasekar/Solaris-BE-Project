import streamlit as st
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# Title
st.title("Solar Yeild Calculator")
st.write("Upload a .tiff image, and the model will predict the output.")

# Load the model
@st.cache_resource
def load_keras_model():
    model = load_model(r"C:\Users\vasek\Downloads\FinalModelTrainTest.keras")  # Replace with your model file
    return model

model = load_keras_model()
uploaded_file = st.file_uploader("Upload a .tiff image", type=["tiff", "tif"])

#Load Classifier

with open("ClassifierModel.json", "r") as json_file:
    model_json = json_file.read()
    Classifier = model_from_json(model_json)

print("Model architecture loaded successfully!")

Classifier.load_weights("model.weights.h5")

print("Model weights loaded successfully!")

Classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("Model compiled successfully!")

#Function for prediction for classifier
def PredictClassifier(Classifier,uploaded_file):
    image=img_to_array(load_image(uploaded_file))
    return Classifier.predict(image)





# Upload TIFF file



# if uploaded_file:
#     st.write(PredictClassifier(Classifier=Classifier,uploaded_file=uploaded_file))

if uploaded_file:
    # Open and process the TIFF image
    try:
        # Read the .tiff image using rasterio
        with rasterio.open(uploaded_file) as src:
            tiff_image = src.read([1, 2, 3])  # Read first 3 bands as RGB
            tiff_image = tiff_image.transpose(1, 2, 0)  # Read the first band (grayscale or single channel)
            st.write(f"Image Dimensions: {tiff_image.shape}")
            
            # Display the image
            

            # Preprocess the image
            
            # Preprocess the image
            st.write("Preprocessing the image for the model...")
            # Resize the image to match the input shape of the model
            resized_image = np.array(Image.fromarray(tiff_image.astype('uint8')).resize((512, 512)))
            resized_image = tiff_image / 255.0  # Normalize pixel values

        

            
            input_image = np.expand_dims(resized_image, axis=0)
            st.image(input_image, caption="resized Image")
            # Predict using the model
            st.write("Making predictions...")
            
            prediction = model.predict(input_image)[0, ..., 0] 
            st.write(prediction.shape)
            threshold =0.5 # Adjust threshold as necessary
            binary_mask = (prediction > threshold).astype(np.uint8)

            # plt.subplot(1, 2, 2)
            # plt.title("Predicted Mask")
            # plt.imshow(binary_mask, cmap='gray')
            # plt.axis('off')
            # plt.show()
            fig, ax = plt.subplots(1, 1, figsize=(2, 2))  # Adjust size as needed
            ax.set_title("Predicted Mask")
            ax.imshow(binary_mask ,cmap='gray')
            ax.axis('off')
            def calculate_area(predicted_mask, pixel_resolution):
                installable_pixels = np.sum(predicted_mask > 0.5)  # Threshold mask to binary
                area = installable_pixels * pixel_resolution  # Convert to real-world area
                return area
    
            st.title(f"Total Rooftop Area: {calculate_area(prediction,0.01):.2f} m²")


            # Render the plot in Streamlit
            st.pyplot(fig)
        
           


    except Exception as e:
        st.error(f"Error processing the TIFF image: {e}")

else:
    st.info("Please upload a .tiff file to get started.")
