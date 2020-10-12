#This class is exactly like neural transfer however within a class
"""
Created on Sun Jul 12 21:53:59 2020
@author: kun-je, Adanna Obibuaku
NST project in spyder
Thisis project is done using "A Neural Algorithm of Artistic Style
by. Leon A. Gatys,  Alexander S. Ecker, Matthias Bethge" as a reference
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.preprocessing.image as img
from tensorflow.keras.applications.vgg19 import decode_predictions
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input


PATH =  os.path.dirname(__file__)
MAIN_PATH =  os.path.join(PATH, "static/utils/")
MODEL = VGG19()
IMG_WIDTH = 224
IMG_HEIGHT = 224
LEARNING_RATE = 0.2
CHANNEL = 3

class Neural():

    def __init__(self, alpha, beta, content_layers, style_layers, style_path, content_path, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_path = content_path
        self.style_path = style_path
    
        self.style_path = os.path.join(MAIN_PATH, self.style_path)
        self.content_path = os.path.join(MAIN_PATH, 'input', self.content_path)
        self.c_image, self.g_image, self.s_image = self.tensor_inputs(self.content_path, self.content_path, self.style_path)
        
    def load_image(self, image_path):
        """
            Description:
                As we are using a pre-trained version VGG16 we have to resize and normalise
                the inputs.
            Args:
                image_path (str): This takes a given an image path
            Returns:
                <class 'numpy.ndarray'> : This would convert the given image into array
                <class 'PIL.Image.Image'>: This would convert the given image into PIL format
        """
        image_array = img.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        image = img.img_to_array(image_array)
        image = image.reshape((1, IMG_HEIGHT, IMG_WIDTH, CHANNEL))
        image = preprocess_input(image)
        return tf.convert_to_tensor(image), image_array

    def tensor_inputs(self, c_image_path, g_image_path, s_image_path):
        """
            Description:
                This is used to take return the tensor image of our content image,
                generate image and style image
            Args:
                c_image_path ():
                g_image_path ():
                s_image_path ():
            Returns:
        """
        c_image = self.load_image(c_image_path)[0]
        g_image = self.load_image(g_image_path)[0]
        s_image = self.load_image(s_image_path)[0]
        return c_image, g_image, s_image
    
    def deprocess_img(self, image):
        """
            Description:
                This is used to reverse the depressing of the image. This is used in order
                to get the image.
            Args:
                image (<class 'numpy.ndarray'>) : Take in the given image in a preprocess format
            Returns:
        """
        temp_image = image
        temp_image = temp_image[0] # Gets one image, from samples
        temp_image = temp_image.reshape((IMG_HEIGHT, IMG_WIDTH, CHANNEL)) # converts it into 3-dimentions
        temp_image[:,:,0] += 103.939 #This adds the mean rgb back to the image, which the preprocess to off
        temp_image[:,:,1] += 116.779
        temp_image[:,:,2] += 123.68
        temp_image = temp_image[:,:,::-1]
        temp_image = np.clip(temp_image, 0, 255)
        return temp_image.astype('uint8')

    def save_image(self, file_name, array_image):
        """
            Description:
                This saves a given tensor image and saves the generated file into
                an output folder
            Args:
                file_name (string): This takes in the given file name
                array_image (): This takes in the given array
        """
        file_name = os.path.join(MAIN_PATH, "output/", file_name)
        img.save_img(file_name, self.deprocess_img(array_image))
        return True

    def MSE(self, matrix_content, matrix_generated):
        """
            Args:
                matrix_content (<class 'numpy.ndarray'>):
                matrix_generated (<class 'numpy.ndarray'>):
            Returns:
                int: A number made by perform substraction operation from each matrix (tensor), followed by
                    squared operation with each substraction operation. The operation reduce mean is then applied.
        """
        return tf.reduce_mean(tf.square(matrix_content - matrix_generated))
    
    def get_layer(self, c_image, s_image, g_image, layer_name):
        """
            Description:
                This returns the activation of the input image.
            Args:
                image (<class 'numpy.ndarray'>): A given image array
                layer_name (str): A given layer name within the cnn model
            Returns:

                <class 'numpy.ndarray'> :
        """
        tensor_image = tf.concat([c_image, s_image, g_image], axis = 0) #put images within one array
        layer = tf.keras.Model(inputs=MODEL.inputs, outputs=MODEL.get_layer(layer_name).output)
        feature = layer(tensor_image) #This will return the activations of the function
        return feature

    def get_feature(self, c_image, s_image, g_image, layer_name):
        """
            Description:
                This function takes in the tensor repersentations c_image, s_image and g_image
                and returns their feauture activations.
            Args:
                c_image (): This is a tensor repersentation of the content image
                s_image (): This is a tensor repersentation of the style image
                g_image (): This is a tensor repersentation of the generated image
            Returns:
                : features of content image
                : features of style image
                : features of generated image


        """
        layer_feature = self.get_layer(c_image, s_image, g_image, layer_name)
        c_feature = layer_feature[0, :, :, :]
        s_feature = layer_feature[1, :, :, :]
        g_feature = layer_feature[2, :, :, :]
        return c_feature, s_feature, g_feature


    def content_loss_function(self, c_feature, g_feature):
        #todo need to change doc string as type was changed
        """
            Args:
                layer_name (str): To take in the layer name

            Returns:
                int: The loss content. A low integer denotes the content is similar
                to the generated image. A high integer denotes the content is not similar
                to the generated image
        """
        WEIGHT = 0.5
        loss = self.MSE(g_feature, c_feature)
        return WEIGHT*loss


    def gram_matrix(self, tensor):
        """
            Args:
                tensor (tensor): take 3D tensor
            Returns:
                gram (tensor) : gram matrix which is 2D array of the multiplication
                of the reshape matrix and its transpose
        """
        m_shape = []
        m_shape.append(tensor.shape[2])
        m_shape.append(tensor.shape[0]*tensor.shape[1])
        tensor = tf.reshape(tensor,m_shape)
        gram = tf.matmul(tensor,tf.transpose(tensor))
        return gram



    def style_loss_function(self, s_feature, g_feature):
        """
            Args:
                c_image_path (str): To take the style image path
                g_image_path (str): To take the generate image path
            Returns:
                int: The loss content. A low integer denotes the content is similar
                to the generated image. A high integer denotes the content is not similar
                to the generated image
        """

        #finding gram matrix of s and g image from perticular layer
        generated_gram = self.gram_matrix(g_feature)
        style_gram = self.gram_matrix(s_feature)

        img_size = IMG_HEIGHT * IMG_WIDTH

        loss = self.MSE(generated_gram, style_gram)/(4*(CHANNEL**2)*(img_size**2))
        return loss

    def total_loss_function(self, c_image,s_image,g_image,alpha,beta):
        """
            Args:
                c_image_path (str): To take the content image path
                s_image_path (str): To take the style image path
                g_image_path (str): To take the generate image path
            Returns:
                int: The totoal loss of style and content.
        """
        content_loss = 0
        c_feature, s_feature, g_feature = self.get_feature(c_image, s_image, g_image, self.content_layers[0]) 
        content_loss += self.content_loss_function(c_feature, g_feature)

        style_loss = 0
        for layer in self.style_layers:
            c_feature, s_feature, g_feature = self.get_feature(c_image, s_image, g_image, layer) 
            style_loss += self.style_loss_function(s_feature, g_feature)

        content_loss *= alpha
        style_loss *= beta

        #total loss
        loss = style_loss + content_loss
        return loss

    def gradient_total_loss(self, c_image, s_image, g_image, alpha, beta):
        """
            Description:
                The purpose of this function is to find the current gradient at 
                the generate image variable (g_image),
            Args:
                c_image ():
                s_image ():
                g_image ():
            Returns:    
        """
        with tf.GradientTape() as tape:
            tape.watch(g_image)
            loss =  self.total_loss_function(c_image,s_image,g_image,alpha,beta)
        dy_dx = tape.gradient(loss, g_image)
        return loss, dy_dx

    def regression_total_loss(self):
            """
                Description:
                    The purpose of this is to use an optimization algorthium called gradient 
                    descent to minimise our loss function (total loss function) in the direction
                    of the steepest descent. In other words where the graph is at its lowest point.
                    In this circumstances we want c_image, s_image to remain static while within 
                    every loop we countisouly change g_image to minimise the loss of our total loss function.
                Args:
                    c_image ()
                    s_image ()
                    g_image ()
                Returns 

            """
            opt = tf.keras.optimizers.Adam(learning_rate=0.2)
            self.g_image  = tf.Variable(self.g_image) 
            iteration = 8000
            for _ in range(iteration+1):
                loss, dy_dx = self.gradient_total_loss(self.c_image, self.s_image, self.g_image, self.alpha, self.beta)
                print("\t Iteration: %d\t Loss: %f" % (_, loss)) 
                opt.apply_gradients([(dy_dx, self.g_image)]) # Apply gradient to varaiable
                if _ % 10 == 0:
                    fname = "img_%d.jpg" % (_)
                    self.save_image(fname, self.g_image.numpy())
