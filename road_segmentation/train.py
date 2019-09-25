import argparse
import glob
import os
import time

import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers

import segmentation_models

HEIGHT = 256
WIDTH = 256


def generator_for_filenames(*filenames):
    """
    Wrapping a list of filenames as a generator function
    """

    def generator():
        for f in zip(*filenames):
            yield f

    return generator


def preprocess(image, segmentation):
    """
    A preprocess function the is run after images are read. Here you can do augmentation and other
    processesing on the images.
    """

    # Set images size to a constant
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    segmentation = tf.image.resize(segmentation, [HEIGHT, WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32) / 255
    segmentation = tf.cast(segmentation, tf.int64)

    return image, segmentation


def read_image_and_segmentation(img_f, seg_f):
    """
    Read images from file using tensorflow and convert the segmentation to appropriate formate.
    :param img_f: filename for image
    :param seg_f: filename for segmentation
    :return: Image and segmentation tensors
    """
    img_reader = tf.io.read_file(img_f)
    seg_reader = tf.io.read_file(seg_f)
    img = tf.image.decode_png(img_reader, channels=3)
    seg = tf.image.decode_png(seg_reader)[:, :, 2:]
    seg = tf.where(seg > 0, tf.ones_like(seg), tf.zeros_like(seg))
    return img, seg


def kitti_dataset_from_filenames(image_names, segmentation_names, preprocess=preprocess, batch_size=8, shuffle=True):
    """
    Convert a list of filenames to tensorflow images.
    :param image_names: image filenames
    :param segmentation_names: segmentation filenames
    :param preprocess: A function that is run after the images are read, the takes image and
    segmentation as input
    :param batch_size: The batch size returned from the function
    :return: Tensors with images and corresponding segmentations
    """
    dataset = tf.data.Dataset.from_generator(
        generator_for_filenames(image_names, segmentation_names),
        output_types=(tf.string, tf.string),
        output_shapes=(None, None)
    )

    if (shuffle):
        dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(read_image_and_segmentation)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch_size)

    return dataset


def kitti_image_filenames(dataset_folder, training=True):
    sub_dataset = 'training' if training else 'testing'
    segmentation_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'gt_image_2', '*road*.png'),
                                   recursive=True)
    image_names = [f.replace('gt_image_2', 'image_2').replace('_road_', '_') for f in segmentation_names]
    return image_names, segmentation_names


def vis_mask(image, mask, alpha=0.4):
    """Visualize mask on top of image, blend using 'alpha'."""

    # Note that as images are normalized, 1 is max-value
    red = tf.zeros_like(image) + tf.constant([1, 0, 0], dtype=tf.float32)
    vis = tf.where(mask, alpha * image + (1 - alpha) * red, image)

    return vis


def main(train_dir):
    # Divide into train and test set.
    train_start_idx, train_end_idx = (0, 272)
    val_start_idx, val_end_idx = (272, 287)

    train_epochs = 10
    batch_size = 4

    # Getting filenames from the kitti dataset
    image_names, segmentation_names = kitti_image_filenames('data_road')

    preprocess_train = preprocess
    preprocess_val = preprocess

    # Get image tensors from the filenames
    train_set = kitti_dataset_from_filenames(
        image_names[train_start_idx:train_end_idx],
        segmentation_names[train_start_idx:train_end_idx],
        preprocess=preprocess_train,
        batch_size=batch_size
    )
    # Get the validation tensors
    val_set = kitti_dataset_from_filenames(
        image_names[val_start_idx:val_end_idx],
        segmentation_names[val_start_idx:val_end_idx],
        batch_size=batch_size,
        preprocess=preprocess_val,
        shuffle=False
    )

    # model = segmentation_models.simple_model((HEIGHT, WIDTH, 3))
    model = segmentation_models.unet((HEIGHT, WIDTH, 3))

    optimizer = optimizers.Adam(lr=1e-4)
    loss_fn = losses.BinaryCrossentropy(from_logits=False)

    print("Summaries are written to '%s'." % train_dir)
    writer = tf.summary.create_file_writer(train_dir, flush_millis=3000)
    summary_interval = 10

    train_accuracy = metrics.BinaryAccuracy(threshold=0.5)
    train_loss = metrics.Mean()
    train_precision = metrics.Precision()
    train_recall = metrics.Recall()
    val_accuracy = metrics.BinaryAccuracy(threshold=0.5)
    val_loss = metrics.Mean()
    val_precision = metrics.Precision()
    val_recall = metrics.Recall()
    step = 0
    start_training = start = time.time()
    for epoch in range(train_epochs):

        print("Training epoch: %d" % epoch)
        for image, y in train_set:
            with tf.GradientTape() as tape:
                y_pred = model(image)
                loss = loss_fn(y, y_pred)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # update metrics and step
            train_loss.update_state(loss)
            train_accuracy.update_state(y, y_pred)
            train_precision.update_state(y, y_pred)
            train_recall.update_state(y, y_pred)
            step += 1

            if step % summary_interval == 0:
                duration = time.time() - start
                print("step %d. sec/batch: %g. Train loss: %g" % (
                    step, duration / summary_interval, train_loss.result().numpy()))
                # write summaries to TensorBoard
                with writer.as_default():
                    tf.summary.scalar("train_loss", train_loss.result(), step=step)
                    tf.summary.scalar("train_accuracy", train_accuracy.result(), step=step)
                    tf.summary.scalar("train_precision", train_precision.result(), step=step)
                    tf.summary.scalar("train_recall", train_recall.result(), step=step)
                    vis = vis_mask(image, y_pred >= 0.5)
                    tf.summary.image("train_image", vis, step=step)

                # reset metrics and time
                train_loss.reset_states()
                train_accuracy.reset_states()
                train_precision.reset_states()
                train_recall.reset_states()
                start = time.time()

        # Do validation after each epoch
        for i, (image, y) in enumerate(val_set):
            y_pred = model(image)
            loss = loss_fn(y, y_pred)
            val_loss.update_state(loss)
            val_accuracy.update_state(y, y_pred)
            val_precision.update_state(y, y_pred)
            val_recall.update_state(y, y_pred)

            with writer.as_default():
                vis = vis_mask(image, y_pred >= 0.5)
                tf.summary.image("val_image_batch_%d" % i, vis, step=step, max_outputs=batch_size)

        with writer.as_default():
            tf.summary.scalar("val_loss", val_loss.result(), step=step)
            tf.summary.scalar("val_accuracy", val_accuracy.result(), step=step)
            tf.summary.scalar("val_precision", val_precision.result(), step=step)
            tf.summary.scalar("val_recall", val_recall.result(), step=step)
        val_loss.reset_states()
        val_accuracy.reset_states()
        val_precision.reset_states()
        val_recall.reset_states()

    print("Finished training %d epochs in %g minutes." % (
        train_epochs, (time.time() - start_training) / 60))
    # save a model which we can later load by tf.keras.models.load_model(model_path)
    model_path = os.path.join(train_dir, "model.h5")
    print("Saving model to '%s'." % model_path)
    model.save(model_path)
    print(model.summary())


def parse_args():
    """Parse command line argument."""

    parser = argparse.ArgumentParser("Train segmention model on Kitti dataset.")
    parser.add_argument("train_dir", help="Directory to put logs and saved model.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    main(args.train_dir)
