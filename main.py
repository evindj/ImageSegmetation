import os.path
import tensorflow as tf
import helper
import project_tests as tests
import tensorflow.contrib.slim as slim


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer3_out)
    """
    # Load the VGG-16 model in the default graph

    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    # Access the graph
    print("The list of operations")
    for op in sess.graph.get_operations():
        print(op.name)
        print(op.values())


    # Retrieve VGG inputs
    image = sess.graph.get_tensor_by_name('image_input:0')
    l3 = sess.graph.get_tensor_by_name('layer3_out:0')
    l4 = sess.graph.get_tensor_by_name('layer4_out:0')
    l7 = sess.graph.get_tensor_by_name('layer7_out:0')
    keep = sess.graph.get_tensor_by_name('keep_prob:0')

    return image, keep,l3, l4, l7


tests.test_load_vgg(load_vgg, tf)

def skip_layer(layer,output):
    one = slim.conv2d(layer,output,[3,3])
    relu = tf.nn.relu(one)
    skip_layer = tf.add(layer,relu)
    return tf.nn.relu(skip_layer)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    NOT SURE HOW TO HANDLE THE SHAPES OF THE SKIP CONNECTIONS.

    """
    skip_vgg_3 = skip_layer(vgg_layer3_out,256)
    interim = tf.matmul(skip_vgg_3,vgg_layer4_out)
    skip_vgg_4 = skip_layer(interim,512)
    interim = tf.matmul(skip_vgg_4,vgg_layer7_out)
    output = slim.fully_connected(interim, num_classes, activation_fn=tf.nn.softmax)

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.matmul(nn_last_layer, correct_label)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=correct_label, logits=logits, name='xentropy')
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)

    # TODO: Implement function
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for i in  range(epochs):
        batches = get_batches_fn(batch_size)
        for batch in batches:
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: keep_prob})

    # TODO: Implement function
    pass
#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        vgg_path = ""
        image, keep,l3, l4, l7 load_vgg(sess, vgg_path)
        network = layers(l3, l4, l7, 2)
        logits, train_op, cross_entropy_loss = optimize(network, image, learning_rate, num_classes)
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # TODO: Build NN using load_vgg, layers, and optimize function


        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

"""
if __name__ == '__main__':
    run()
"""
