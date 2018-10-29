#
#     Copywrite 2017 Alan Steremberg and Arthur Conner
#

import os
import tensorflow as tf
from tensorflow.python.platform import gfile
#from tensorflow.python.tools import freeze_graph
import tensorflow.contrib.lite as tflite
from tensorflow.python.tools import optimize_for_inference_lib

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_config
from tensorflow.keras.models import Sequential, Model
import argparse


def freeze_graph(model_dir, output_node_names):
  """Extract the sub graph defined by the output nodes and convert
  all its variables into constant
  Args:
      model_dir: the root folder containing the checkpoint state file
      output_node_names: a string, containing all the output node's names,
                          comma separated
                        """
  if not tf.gfile.Exists(model_dir):
    raise AssertionError(
      "Export directory doesn't exists. Please specify an export "
      "directory: %s" % model_dir)

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  # We retrieve our checkpoint fullpath
  checkpoint = tf.train.get_checkpoint_state(model_dir)
  input_checkpoint = checkpoint.model_checkpoint_path

  # We precise the file fullname of our freezed graph
  absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
  output_graph = absolute_model_dir + "/frozen_model.pb"

  # We clear devices to allow TensorFlow to control on which device it will load operations
  clear_devices = True

  # We start a session using a temporary fresh Graph
  with tf.Session(graph=tf.Graph()) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    print(f"Merging weights from {input_checkpoint}")
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, # The session is used to retrieve the weights
      tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
      output_node_names.split(",") # The output node names are used to select the usefull nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

  return output_graph_def


def convert(prevmodel, export_path, export_name):

    graph1 = tf.Graph()
    with graph1.as_default():
      sess1 = tf.Session()
      with sess1.as_default():
        previous_model = load_model(prevmodel)
        previous_model.summary()
        config = previous_model.get_config()
        weights = previous_model.get_weights()

    graph2 = tf.Graph()
    with graph2.as_default():
      sess2 = tf.Session()
      with sess2.as_default():
        K.set_learning_phase(0)
        try:
          model= Sequential.from_config(config)
        except:
          model= Model.from_config(config)
        model.set_weights(weights)

        with open(os.path.join(export_path, export_name + '.input_output.txt'), 'w') as fid:
          model_input_name = model.input.name
          #model_input_node_name
          model_output_name = model.output.name
          print(f"Input name: {model_input_name}")
          print(f"Output name: {model_output_name}")
          fid.write(f"{model_input_name}\n")
          fid.write(f"{model_output_name}\n")

        model_output_node_name=model.output.name.split(':')[0]

        graph_file = os.path.join(export_path, export_name + ".graph.pbtxt")
        ckpt_file = os.path.join(export_path, export_name + ".ckpt")
        saver = tf.train.Saver()
        tf.train.write_graph(sess2.graph_def, '', graph_file)
        save_path = saver.save(sess2, ckpt_file)

        # freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
          sess2, # The session is used to retrieve the weights
          graph2.as_graph_def(), # The graph_def is used to retrieve the nodes
          model_output_node_name.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        frozen_graph_path = os.path.join(export_path, export_name + ".frozen.pb")
        with tf.gfile.GFile(frozen_graph_path, "wb") as f:
          f.write(frozen_graph_def.SerializeToString())

    #tf.reset_default_graph()
    #frozen_graph_def = freeze_graph(export_path, model_output_node_name)

    tf.reset_default_graph()
    train_writer = tf.summary.FileWriter(export_path)
    train_writer.add_graph(frozen_graph_def)
    train_writer.flush()

    # optimize for inference
    optimized_graph_def = optimize_for_inference_lib.optimize_for_inference(
      frozen_graph_def,
      [model_input_name.split(':')[0]],
      [model_output_name.split(':')[0]],
      tf.float32.as_datatype_enum)
      #tf.int32.as_datatype_enum

    optimized_graph_path = os.path.join(export_path, export_name + ".optimized_for_inference.pb")
    with tf.gfile.GFile(optimized_graph_path, "wb") as f:
      f.write(optimized_graph_def.SerializeToString())

    # tflite
    # ## from frozen graph
      converter = tflite.TocoConverter.from_frozen_graph(
        optimized_graph_path,
        [model_input_name.split(':')[0]],
        [model_output_name.split(':')[0]],
        {model_input_name.split(':')[0]: [1, 32, 32, 3]})
    #
    ## from keras model file
    # only available in tensorflow >=1.11
    #converter = tflite.TocoConverter.from_keras_model_file(prevmodel)
    converter.post_training_quantize = True
    #converter.inference_type = tf.quint8
    #converter.inference_input_type = tf.float32
    tflite_quantized_model = converter.convert()
    optimized_graph_path = os.path.join(export_path, export_name + ".tflite")
    open(optimized_graph_path, "wb").write(tflite_quantized_model)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras Tensorflow Converter')
    parser.add_argument(
        'model',
        type=str,
        help='Path to the keras model'
    )
    parser.add_argument(
        'frozen',
        type=str,
        help='Path to the frozen output'
    )
    parser.add_argument(
        'freezegraph',
        type=str,
        help='Path to the freeze_graph binary'
    )
    args = parser.parse_args()

    #convert(args.model,args.frozen,args.freezegraph)

