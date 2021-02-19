'''
Created on Dec 2, 2017

@author: mroch
'''

from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K

def build_model(specification, name="model"):
    """build_model - specification list
    Create a model given a specification list
    Each element of the list represents a layer and is formed by a tuple.
    
    (layer_constructor, 
     positional_parameter_list,
     keyword_parameter_dictionary)
    
    Example, create M dimensional input to a 3 layer network with 
    20 unit ReLU hidden layers and N unit softmax output layer
    
    [(Dense, [20], {'activation':'relu', 'input_dim': M}),
     (Dense, [20], {'activation':'relu', 'input_dim':20}),
     (Dense, [N], {'activation':'softmax', 'input_dim':20})
    ]
    
    Wrappers are supported by creating a 4th item in the tuple/list
    that consists of a tuple with 3 items:
        (WrapperType, [positional args], {dictionary of arguments})
        
    The WrapperType is wrapped around the specified layer which is assumed
    to be the first argument of the constructor.  Additional positional
    argument are taken from the second item of the tuple and will *follow*
    the wrapped layer argument.  Dictionary arguments
    are applied as keywords.
    
    For example:
    (Dense, [20], {'activation':'relu'}, (TimeDistributed, [], {}))
    
    would be equivalent to calling TimeDistributed(Dense(20, activation='relu'))
    If TimeDistributed had positional or named arguments, they would be placed 
    inside the [] and {} respectively.  Remember that the wrapped layer (Dense)
    in this case is *always* the first argument to the wrapper constructor.

    Author - Marie A. Roch, 12/2017
    """
    
    K.name_scope(name)
    model = Sequential()

    for item in specification:
        layertype = item[0]
        # Construct layer and add to model
        # This uses Python's *args and **kwargs constructs
        #
        # In a function call, *args passes each item of a list to 
        # the function as a positional parameter
        #
        # **args passes each item of a dictionary as a keyword argument
        # use the dictionary key as the argument name and the dictionary
        # value as the parameter value
        #
        # Note that *args and **args can be used in function declarations
        # to accept variable length arguments.
        layer = layertype(*item[1], **item[2])
        
        if len(item) > 3:
            # User specified wrapper
            wrapspec = item[3]
            # Get type, positional args and named args
            wraptype, wrapposn, wrapnamed = wrapspec
            wlayer  = wraptype(layer, *wrapposn, **wrapnamed)
            model.add(wlayer)
        else:
            # No wrapper, just add it.
            model.add(layer)
        
    return model