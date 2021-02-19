'''
A class and function for generating confusion matrix images that can be called
directory (plot_confusion) or set up as a TensorBoard callback

Inspired from a stackoverflow post by MLNinja
https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
we added a few throwing stars and ... :-)

Requires tensorflow-plot, see comments above import tfplot
Note that tfplot tries to set a non-interactive graphics rending engine for 
matplotlib and will produce a warning that it cannot change rendering engines
if the graphics backend is already initialized.
You may safely ignore the warning.
'''

import os.path
import itertools
import pathlib

import tensorflow.keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib


# Requires tensorflow-plot module developed by Jongwook Choi
# Install: pip install tensorflow-plot
import tfplot

def plot_confusion(predictions, truth, labels):
    """plot_confusion(predictions, truth, labels)
    Plot a confusion matrix for a set of predicted class with the possibility
    of masked values
    
    predictions - prediction class list
    truth - truth class list
    labels - Names of categories

    Returns tuple:
        (confusion_matrix, fig_handle, axes_handle, image_handle)
    """ 

    N = len(labels)
    label_indices = [x for x in range(N)]

    confusion = confusion_matrix(truth, predictions)

    # Build the figure and axes            
    fig = plt.figure(figsize=(4.5,4.5), dpi=320,
                    facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    
    
    # Plot the confusion matrix
    # Show the heat map relative to the number of true examples
    # so that the most frequent decisions are always highlighted
    relative = confusion / confusion.sum(axis=1)[:,None]
    im = ax.imshow(relative, cmap='Oranges')
   
    # Label the axes
    tick_marks = np.arange(N)
    
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(labels, fontsize=3.5, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    
    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=4, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    
    # Add counts in boxes
    for i, j in itertools.product(
        range(confusion.shape[0]), range(confusion.shape[1])):
        
        ax.text(j, i, format(confusion[i, j], 'd') if confusion[i,j] !=0 
                else '.', 
                horizontalalignment="center", fontsize=2,
                verticalalignment='center', color= "black")
        
    fig.set_tight_layout(True)
    
    return confusion, fig, ax, im

    
class ConfusionTensorBoard(Callback):
    """
    A class for generating confusion matrix images that can be displayed
    in TensorBoard
    
    Inspired from a stackoverflow post by MLNinja
    https://stackoverflow.com/questions/41617463/tensorflow-confusion-matrix-in-tensorboard
    we added a few throwing stars and ... :-)
    
    Usage:
    from keras import backend as K
    
    # A tensorflow session must already be started (this is certainly
    # true after model compilation)
    # In this example, we assume that model is a compiled model
    # and corpus is a Corpus object.
    
    # TensorBoard confusion matrices
    confusion = ConfusionTensorBoard(log_dir, corpus.get_phonemes(), 
                                     K.get_session())
    confusion.add_callbacks(model)  # fetch labels/outputs
    
    """
    
    summary_types = ['batch', 'epoch']
    
    def __init__(self, logdir, labels, writer, 
                 tag='confusion', summaries='epoch'):
        """"ConfusionTensorBoard(logdir, tag, summaries)
            
            logdir - TensorBoard log directory
            labels - List of label classes
            writer - tensorflow.summary.FileWriter instance
            tag - A name
            summaries - Create image on "batch" or "epoch" (default).
                Use a list to provide summaries for both ['batch', 'epoch']
                
        Note that several variables are set by the callback mechanism,
        some examples:
            validation_data - list with examples and targets
            model
        """
        
        # Call parent class constructor and save tag
        super().__init__()  
        self.logdir = logdir  # TensorBoard log directory
        
        # Location of TensorBoard images
        self.imgdir = os.path.join(logdir, 'summaries', 'img')
        pathlib.Path(self.imgdir).mkdir(parents=True, exist_ok=True)
        # Start a writer        
        self.imgwriter = tf.summary.FileWriter(self.imgdir, K.get_session().graph)
        
        self.labels = labels
        self.label_indices = [l for l in range(len(self.labels))]
        
        self.writer = writer
        
        self.tag = tag
                
        
        
        if isinstance(summaries, str):
            summaries = [summaries]
        bad = [s for s in summaries if s not in self.summary_types]
        if len(bad) > 0:
            raise RuntimeError(
                'summaries must be %s'%(", ".join(self.summaries)))
             
        self.summaries = summaries
        
        # Set up variables that tensorflow will write to in its fetch
        # operation
        # Last batch may not be of same size, so do not validate shape
        self.var_labels = tf.Variable(0., validate_shape=False)
        self.var_predictions = tf.Variable(0., validate_shape=False)
        self.var_mask = tf.Variable(False, dtype=np.bool, validate_shape=False)
        self.mask_present = False
        
    def add_callbacks(self, model):
        """construct_callbacks(model)
        Given a keras model with tensorflow as the underlying implementation,
        add the callbacks to Tensorflow to obtain the predictions and labels
        
        This must be done *before* the model is fit.
        """
        
        # Add to an existing fit list if appropriate
        try:
            fit_list = model._function_kwargs['fetches']
        except KeyError:
            fit_list = []
        
        fit_list.extend([tf.assign(self.var_labels, model.targets[0], 
                                  validate_shape=False),
                         tf.assign(self.var_predictions, model.outputs[0],
                                   validate_shape=False)])
        # If a mask layer has been added, we will need to ignore some
        # inputs when creating the confusion matrix.  Find the Masking
        # layer and add it to the fit_list
        # Right now, we are looking for the first Masking layer, but there
        # may be times when this is inappropriate (cannot think of any right
        # now, but keep it in mind)
        for l in model.layers:
            present = isinstance(l, keras.layers.Masking)
            if present:
                self.mask_present = True
                fit_list.append(tf.assign(self.var_mask, l.output_mask,
                                          validate_shape=False ))
            
        #fit_list.append(lambda : self.update(model))
        model._function_kwargs['fetches'] = fit_list
        
    def update(self, model):
        "update(model) - Save current label (target) and result"
        tf.assign(self.var_labels, model.targets[0], validate_shape=False),
        tf.assign(self.var_predictions, model.outputs[0], validate_shape=False)
        print('here')
        
        
    def __confusion(self, step, tag=""):
        """_confusion(step, tag)
        Log confusion matrix for Tensorflow
        Uses current labels and predictions (representative of the last batch
        of training data) 
        """
        
        # Retrieve values of the tensors
        label_t = K.eval(self.var_labels)
        pred_t = K.eval(self.var_predictions)
        
        interactive = matplotlib.is_interactive()
        if interactive:
            matplotlib.interactive(False)
        
        _conf, fig, _ax, _im = \
            plot_confusion(pred_t, label_t, self.labels)
        
        # Create summary to write to logs
        summary = tfplot.figure.to_summary(
            fig, tag="%s_%s_%d"%(self.tag, tag, step))
        self.imgwriter.add_summary(summary, step)
        
        if interactive:
            matplotlib.interactive(interactive)  # restore state
        
        
    def on_batch_end(self, batch, logs={}):
        if 'batch' in self.summaries:
            self.__confusion(batch, 'b')
        pass
        
    def on_epoch_end(self, epoch, logs={}):
        "on_epoch_end(epoch, logs) - Create confusion matrix for epoch"
        
        # todo - Should figure out how to generate a confusion
        # matrix for validation data.  self.validation_data contains
        # both the examples ([0]) and the labels ([1]), but predicting
        # on them is causing issues.
        if 'epoch' in self.summaries:
            self.__confusion(epoch, 'e')            
        pass
        
        
        
    
    
    
