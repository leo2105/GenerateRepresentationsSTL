from tensorflow.keras.layers import Input, Flatten, Dense, LSTM, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Reshape, \
    Conv2DTranspose, InputLayer, MaxPool2D, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D
from random import randint
import tensorflow
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import os
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
from sklearn.decomposition import PCA
import keras.losses
from matplotlib import pyplot as plt

class Representations:
    #constructor
    def __init__(self, auxiliary_train, auxiliary_test, X_target_CK, X_target_JAFFE):
        self.auxiliary_train = auxiliary_train
        self.auxiliary_test = auxiliary_test
        self.X_target_CK = X_target_CK
        self.X_target_JAFFE = X_target_JAFFE

    def Generate_all(self, stride=2, activation='relu', filter_convolution=3, filters=[16, 32, 64, 128, 256],
                      output_activation='linear', size_input_data=[96, 96, 1],
                      n_hidden=500, n_encoder=5, optimizer=SGD(lr=0.001, momentum=0.9, decay=1e-6),
                      kernel_initializer=None, padding='same', epochs=20, verbose=2, batch_size=60,
                      seeds_rep = True, hidden_rep = False, arch_rep = False, number_of_repr = 50, const = 0.001, techs = 'S'):
        auxiliary_train = self.auxiliary_train
        auxiliary_test = self.auxiliary_test
        X_target_CK = self.X_target_CK
        X_target_JAFFE = self.X_target_JAFFE

        self.size_input_data = size_input_data
        self.filters = filters
        self.filter_convolution = filter_convolution
        self.activation = activation
        self.output_activation = output_activation
        self.stride = stride
        self.n_hidden = n_hidden
        self.n_encoder = n_encoder
        self.optimizer = optimizer
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.seeds_rep = seeds_rep
        self.hidden_rep = hidden_rep
        self.arch_rep = arch_rep
        self.number_of_repr = number_of_repr
        max_layers = n_encoder
        total_filters = filters
        self.const = const
        self.techs = techs
        weights_cae = []



        x_unlabeled = [] #variavel utilizada para armazenar o vetor latente da representacao criada
        input_img = Input(shape=(size_input_data[0], size_input_data[1], 1))

        name_tech = ''

        if seeds_rep == True:
            name_tech = name_tech + 'S'
        if hidden_rep == True:
            name_tech = name_tech + 'L'
        if arch_rep == True:
            name_tech = name_tech + 'A'

        self.lista_rep = [150, 200, 250, 300, 400, 500, 1000, 1500, 2000, 2500]
        #for j in range(number_of_repr):
        for j in range(52):
            print(f"Representation: {j}")
            if seeds_rep == True:
                value_seeds = randint(0, 10000)
                print(value_seeds)
                tensorflow.random.set_seed(value_seeds)
            else:
                tensorflow.random.set_seed(42)

            if hidden_rep == True:
                value_hidden = randint(150, 2500)
                #value_hidden = self.lista_rep[j]
                #print(value_hidden)
                tensorflow.random.set_seed(value_hidden)
                hidden = value_hidden
                
            else:
                hidden = n_hidden

            # generate encoder
            for i in range(n_encoder):
                if i == 0:
                    autoencoder = Conv2D(filters[i], (filter_convolution, filter_convolution), strides=stride,
                                         activation=activation,
                                         padding='same', kernel_initializer='he_uniform')(input_img)
                else:
                    autoencoder = Conv2D(filters[i], (filter_convolution, filter_convolution), strides=stride,
                                         activation=activation,
                                         padding='same', kernel_initializer='he_uniform')(autoencoder)

            # generate latent vector
            autoencoder = Flatten()(autoencoder)
            autoencoder = Dense(hidden, activation=activation, name='hidden_layer')(autoencoder)
            autoencoder_decoder = Dense(
                units=filters[n_encoder - 1] * int(size_input_data[0] / (2 ** (len(filters)))) * int(
                    size_input_data[0] / (2 ** (len(filters)))), activation=activation)(autoencoder)

            # generate decoder
            for i in range(n_encoder, -1, -1):
                if i == (n_encoder):
                    autoencoder_decoder = Reshape((int(size_input_data[0] / (2 ** (len(filters)))),
                                                   int(size_input_data[0] / (2 ** (len(filters)))), filters[-1]))(
                        autoencoder_decoder)
                elif i == 0:
                    autoencoder_decoder = Conv2DTranspose(filters[i], (filter_convolution, filter_convolution),
                                                          strides=stride,
                                                          activation=activation, padding='same',
                                                          kernel_initializer='he_uniform')(autoencoder_decoder)

                else:
                    autoencoder_decoder = Conv2DTranspose(filters[i], (filter_convolution, filter_convolution),
                                                          strides=stride,
                                                          activation=activation, padding='same',
                                                          kernel_initializer='he_uniform')(autoencoder_decoder)

            autoencoder_decoder = Conv2D(1, (filter_convolution, filter_convolution), activation=output_activation,
                                         padding='same')(autoencoder_decoder)

            if j == 0:
                autoencoder_model = Model(input_img, autoencoder_decoder)
            else:
                autoencoder_model = Model(inputs=[input_img], outputs=[autoencoder_decoder])

            print(autoencoder_model.summary())

            if x_unlabeled != []:
                min_value = []
                min_value.append(x_unlabeled.shape[0])
                min_value.append(x_unlabeled.shape[1])
                min_value.append(hidden)
                valor_minimo = min(min_value)

            def customized_loss(autoencoder_model):
                self.autoencoder_model = autoencoder_model
                def lossFunction(y_true, y_pred):
                    if (j == 0):
                        loss = tf.square(tf.subtract(y_true, y_pred))
                        return tf.reduce_mean(loss, axis=1)
                    else:
                        '''
                        layer_name = 'hidden_layer'
                        get_3rd_layer_output = K.function([self.autoencoder_model.input],
                                                          [self.autoencoder_model.get_layer(layer_name).output])
                        layer_output = get_3rd_layer_output([auxiliary_test])
                        layer_output = np.array(layer_output)
                        layer_output = layer_output.reshape(layer_output.shape[1], layer_output.shape[2])

                        pca = PCA(n_components=min(min_value))
                        pca.fit(x_unlabeled)
                        x_unlabeled_pca = pca.transform(x_unlabeled)

                        pca2 = PCA(n_components=min(min_value))
                        pca2.fit(layer_output)
                        x_layer_output = pca2.transform(layer_output)

                        #K.print_tensor(pca2, message='pca2 = ')

                        hidden_dif = tf.square(x_layer_output - x_unlabeled_pca)
                        K.print_tensor(hidden_dif, message='hidden_dif = ')

                        hidden_dif = tf.reduce_mean(hidden_dif)
                        K.print_tensor(hidden_dif, message='hidden_dif_mean = ')
                        hidden_dif = const / (hidden_dif)

                        loss = tf.square(tf.subtract(y_true, y_pred))

                        loss_final = loss + hidden_dif
                        return tf.reduce_mean(loss_final, axis=1)
                        '''
                        dif_weights = []
                        max_lenvalues = []
                        weights_lasts = []
                        dist = []
                        size_weights = len(weights_cae)

                        #teste3 = tf.Variable(K.size(size_weights))
                        #K.print_tensor(teste3, message='size_weights = ')

                        #teste3 = tf.Variable((size_weights))
                        #K.print_tensor(teste3, message='size_weights = ')

                        for i in range(size_weights):
                            weights_lasts.append(weights_cae[i])

                        weights_complete_loss = autoencoder_model.get_layer('hidden_layer').get_weights()
                        weights_lasts.append(weights_complete_loss[1])

                        size_weights = len(weights_lasts)
                        
                        #teste3 = tf.Variable(K.size(size_weights2))
                        #K.print_tensor(size_weights2, message='size_weights2 = ')

                        #weights_lasts = np.asarray(weights_lasts)

                        #teste4 = tf.Variable(K.int_shape(weights_lasts))
                        #K.print_tensor(teste4, message='weights_lasts = ')
                        for i in range(size_weights):
                            max_lenvalues.append(len(weights_lasts[i]))

                        max_value = max(max_lenvalues)

                        for i in range(size_weights):
                            if max_lenvalues[i] < max_value:
                                weights_lasts[i] = np.concatenate((weights_lasts[i], np.zeros((max_value - max_lenvalues[i]))), axis=None)

                        for i in range(size_weights - 1):
                            diff = weights_lasts[i] - weights_lasts[-1]
                            dif_weights.append(diff)

                        for i in range(size_weights - 1):
                            dist.append(np.sqrt(np.dot(dif_weights[i], dif_weights[-1])))

                        sum_total = np.sum(dist)

                        #K.print_tensor(sum_total, message='sum_total = ')

                        hidden_dif = const / sum_total

                        loss = tf.square(tf.subtract(y_true, y_pred))

                        loss_final = loss + hidden_dif
                        return tf.reduce_mean(loss_final, axis=1)
                return lossFunction

            keras.losses.mean_squared_error= customized_loss
            tf.executing_eagerly()

            autoencoder_model.compile(optimizer=optimizer, loss=customized_loss(autoencoder_model), run_eagerly=True)
            earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)

            history = autoencoder_model.fit(auxiliary_train, auxiliary_train,
                                  epochs=epochs,
                                  verbose=verbose,
                                  batch_size=batch_size,
                                  validation_data=(auxiliary_test, auxiliary_test))
                                  #callbacks = [earlyStop])
            '''           
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            '''

            weights_complete = autoencoder_model.get_layer('hidden_layer').get_weights()
            weights_cae.append(weights_complete[1])

            '''
            if j != 0:
                dif_weights = []
                weights_sum = []
                max_lenvalues = []
                dist = []
                size_weights = len(weights_cae)

                for i in range (size_weights):
                    max_lenvalues.append(len(weights_cae[i]))

                max_value = max(max_lenvalues)

                for i in range (size_weights):

                    if max_lenvalues[i] < max_value:
                        weights_cae[i] = np.concatenate((weights_cae[i], np.zeros((max_value - max_lenvalues[i]))), axis=None)

                for i in range(size_weights-1):
                    diff = weights_cae[i] - weights_cae[-1]
                    dif_weights.append(diff)

                for i in range(size_weights-1):
                    dist.append(np.sqrt(np.dot(dif_weights[i], dif_weights[-1])))

                sum_total = np.sum(dist)
                print("sum_total", sum_total)
            '''

            '''
            # save model
            l = j
            model_json = autoencoder_model.to_json()
            filename = "%s.json" % l
            filename_h = "%s.h5" % l
            if os.path.exists('./temp_autoencoder/' + str(number_of_repr) + ' REP/' + name_tech) == False:
                os.makedirs("./temp_autoencoder/" + str(number_of_repr) + ' REP/' + name_tech)
            with open("./temp_autoencoder/" + str(number_of_repr) + ' REP/' + name_tech + '/' + filename,
                      "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            autoencoder_model.save_weights(
                "./temp_autoencoder/" + str(number_of_repr) + ' REP/' + name_tech + '/' + filename_h)
            print("Saved model to disk")
            '''

            # variável x_unlabeled armazena o vetor latente da representacao criada (h-last)
            feature_extraction = Sequential()
            for k in range(n_encoder):
                if k == 0:
                    feature_extraction.add(Conv2D(filters[k], (filter_convolution, filter_convolution), strides=stride,
                                                  activation=activation, padding='same',
                                                  weights=autoencoder_model.layers[k + 1].get_weights(),
                                                  input_shape=(
                                                      size_input_data[0], size_input_data[1], size_input_data[2],)))
                else:
                    feature_extraction.add(Conv2D(filters[k], (filter_convolution, filter_convolution), strides=stride,
                                                  activation=activation, padding='same',
                                                  weights=autoencoder_model.layers[k + 1].get_weights()))
            feature_extraction.add(Flatten())

            feature_extraction.add(
                Dense(hidden, activation=activation, weights=autoencoder_model.layers[k + 3].get_weights()))
            print(feature_extraction.summary())

            x_JAFFE = feature_extraction.predict(X_target_JAFFE)
            x_CK = feature_extraction.predict(X_target_CK)
            x_unlabeled = feature_extraction.predict(auxiliary_test)


            if os.path.exists('./temp2/JAFFE/' + str(number_of_repr) + ' REP/' + techs) == False:
                os.makedirs('./temp2/JAFFE/' + str(number_of_repr) + ' REP/' + techs)
            if os.path.exists('./temp2/CK/' + str(number_of_repr) + ' REP/' + techs) == False:
                os.makedirs('./temp2/CK/' + str(number_of_repr) + ' REP/' + techs)

            number = j
            np.save('./temp2/JAFFE/' + str(number_of_repr) + ' REP/' + techs + '/' + "images_%s" % number, x_JAFFE)
            np.save('./temp2/CK/' + str(number_of_repr) + ' REP/' + techs + '/' + "images_%s" % number, x_CK)

            if n_encoder == 1:
                n_encoder = max_layers
                filters = total_filters
            elif arch_rep == True:
                filters = filters[:-1]
                n_encoder = n_encoder - 1