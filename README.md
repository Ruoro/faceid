# FaceID with Custom Model

by John Ruoro
This Jupyter Notebook contains a Python program that uses computer vision to detect faces with a custom model and input from the webcam. The program uses the OpenCV and TensorFlow libraries to create and train the custom model, and to detect faces in real-time video input from the webcam.

### Installation
To run the program, you'll need to have the following libraries installed:

- OpenCV
- TensorFlow
- NumPy
You can install these libraries using pip:
'''console
pip install opencv-python tensorflow numpy
'''
You'll also need to have Jupyter Notebook installed on your computer. You can download it from the official website: https://jupyter.org/install

Once you have everything installed, download the faceid.ipynb file and save it to a directory of your choice.

### How to Use
To use the program, open up Jupyter Notebook and navigate to the directory where you saved the faceid.ipynb file. Then, open the file and run each cell in order.

The program will prompt you to capture images of your face to use as training data for the custom model. Follow the prompts to capture images of your face from different angles and with different expressions. The program will use these images to train the model.

Once the model is trained, the program will use the webcam to detect faces in real-time video input. It will draw a bounding box around each detected face and label it with the name of the person it recognizes.

### How the Custom Model Works
The custom model is a deep neural network that uses TensorFlow to classify images of faces as belonging to one of several people. It is based on a pre-trained model called MobileNet, which is optimized for use on mobile devices with limited computational resources.

The custom model is trained using a technique called transfer learning, which involves using a pre-trained model as a starting point and fine-tuning it with new data. In this program, the pre-trained MobileNet model is used as a feature extractor to generate a set of features for each input image. These features are then fed into a fully connected layer that outputs a probability distribution over the possible classes (i.e. the different people whose faces the model has been trained to recognize).

The custom model is trained using a dataset of images of the people whose faces it will be asked to recognize. The images are labeled with the name of the person they belong to. During training, the model adjusts its weights and biases to minimize the difference between its predicted probability distributions and the true labels.

### Future Improvements
Here are some ideas for how the program could be improved in the future:

- Add an option to save the trained model to a file, so it can be reused later without having to train it again.
- Add an option to recognize faces in images or videos, rather than just in real-time webcam input.
- Experiment with different pre-trained models or different configurations of the custom model to see if they lead to better performance.
