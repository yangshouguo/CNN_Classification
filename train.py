import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helper import DataHelper
from TextClassif import TextCNN
from tensorflow.contrib import learn

tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation (default is 0.2)")
tf.flags.DEFINE_string("data_file", "../dataset_singlefunc/",
                       "Data source for the data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
#                        "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 4, "Dimensionality of character embedding (default: 4)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5,6", "Comma-separated filter sizes (default: '3,4,5,6')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.8)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("hidden_dim", 64, "hidden layer dimension default(500)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 20, "Number of checkpoints to store (default: 20)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
data_helper = DataHelper()

class_num = 5 # 类别数目

def preprocess():
    print('loading data...')

    x_text, y = data_helper.load_data_and_labels(FLAGS.data_file)

    #one-hot encode
    y_oh = tf.one_hot(y, depth=class_num)
    with tf.Session() as sess:
        y_encode = sess.run(y_oh)
        # print(sess.run(y_oh))




    # # build vocabulary
    # max_document_length = max([len(x.split(' ')) for x in x_text])
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # x = np.array(list(vocab_processor.fit_transform(x_text)))

    # randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x_text[shuffle_indices]
    y_shuffled = y_encode[shuffle_indices]

    # split train/test set
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x_text, y_encode, x_shuffled, y_shuffled

    # print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    # print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev

def train(x_train, y_train, x_dev, y_dev):

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_device_placement = FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                seq_len=x_train.shape[1],
                seq_width=x_train.shape[2],
                num_class = class_num,
                hidden_size = FLAGS.hidden_dim,
                # vocabsize=len(vocab_processor.vocabulary_),
                embedding_size= FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters
            )
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3) #learning rate
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)



        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print('wirting to {}\n'.format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        # loss / acc / gradient
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        # vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        import datetime
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer= None
                       ):

            feed_dict = {cnn.input_x:x_batch, cnn.input_y:y_batch,
                         cnn.dropout_keep_prob:1.0}

            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print("dev_size is {} ; {}: step {}, loss {:g}, acc{:g}".format(len(x_batch) ,time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

            return accuracy

        #Generate batches
        batches = data_helper.batch_iter(list(zip(x_train, y_train)),FLAGS.batch_size,
                                              FLAGS.num_epochs)
        #dev batches
        dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size, 1, shuffle=False)

        #Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")

                acc_array = []
                for dev_batch in dev_batches:
                    x_batch, y_batch = zip(*dev_batch)
                    acc_array.append(dev_step(x_batch, y_batch))

                print("average accuracy of test data is {}".format(np.mean(np.array(acc_array))))

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model chechpoint to{}\n".format(path))

        #testing

        acc_array = []
        for dev_batch in dev_batches:
            x_batch, y_batch = zip(*dev_batch)
            acc_array.append(dev_step(x_batch, y_batch))

        print("average accuracy of test data is {}".format(np.mean(np.array(acc_array))))


def main(arg):
    x_train, y_train, x_dev, y_dev = preprocess()
    train(x_train, y_train, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run() # parse FLAG and run main
