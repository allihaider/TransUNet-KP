import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import cv2

def load_and_preprocess_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_outputs)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_outputs)
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image, style_targets, content_targets, optimizer, style_weight, content_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

def generate_synthetic_image(content_image, style_image, mask, num_iterations=1000):
    content_image = load_and_preprocess_image(content_image)
    style_image = load_and_preprocess_image(style_image)
    mask = cv2.imread(mask, 0) / 255.0
    mask = cv2.resize(mask, (224, 224))

    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    extractor = tf.keras.Model(inputs=vgg.input, outputs=model_outputs)

    style_features = extractor(style_image)
    content_features = extractor(content_image)

    style_targets = {name: gram_matrix(value) for name, value in zip(style_layers, style_features[:len(style_layers)])}
    content_targets = {name: value for name, value in zip(content_layers, content_features[len(style_layers):])}

    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    style_weight = 1e-2
    content_weight = 1e4

    for n in range(num_iterations):
        train_step(image, style_targets, content_targets, opt, style_weight, content_weight)
        if n % 100 == 0:
            print(f"Iteration {n}")

    result = deprocess_image(image.numpy())
    result = result * mask[:, :, np.newaxis] + content_image[0] * (1 - mask[:, :, np.newaxis])
    return array_to_img(result)

# Usage
content_image_path = 't1.jpg'
style_image_path = 't2.jpg'
mask_path = './synthetic_data/cp_aug/masks/synthetic_2_labels.png'

synthetic_image = generate_synthetic_image(content_image_path, style_image_path, mask_path)
synthetic_image.save('synthetic_image.jpg')
