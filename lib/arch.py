
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, BatchNormalization

networks = {
    'DenseX4_L2' : lambda layer_width, indim, outdim, penalty : [
        (Input, [], {'shape':indim}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l2(penalty)}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l2(penalty)}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l2(penalty)}),
        (Dense, [outdim], {'activation':'softmax'}),
        ],
    'DenseX5_L2' : lambda layer_width, indim, outdim, penalty, drop : [
        (Input, [], {'shape':indim}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l2(penalty)}),
        (Dropout, [drop], {}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l2(penalty)}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l2(penalty)}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l2(penalty)}),
        #(BatchNormalization, (), {}),
        (Dense, [outdim], {'activation':'softmax'}),
        ],
    'DenseX3_Dropout' : lambda layer_width, indim, outdim, Pdrop, penalty : [
        (Input, [], {'shape':indim}),
        (Dropout, [Pdrop], {}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l2(penalty)}),
        (Dropout, [Pdrop], {}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l2(penalty)}),
        (Dense, [outdim], {'activation':'softmax'}),
        ],
    'DenseX4_L1' : lambda layer_width, indim, outdim, penalty : [
        (Input, [], {'shape':indim}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l1(penalty)}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l1(penalty)}),
        (Dense, [layer_width], {'activation':'relu',
                                'kernel_regularizer':regularizers.l1(penalty)}),
        (Dense, [outdim], {'activation':'softmax'}),
        ],
}