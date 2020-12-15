#Designed by lcy
import tensorflow as tf
import os

from keras.callbacks import Callback

from utilities import save_network, load_network
from keras.models import Sequential
from topology import Convolutional, Pooling, Dropout, Block
from random import randint, choice
from copy import deepcopy


class Network:
    __slots__ = ('name', 'block_list', 'fitness', 'model')

    def __init__(self, it):
        self.name = 'parent_' + str(it) if it == 0 else 'net_' + str(it)
        self.block_list = []
        self.fitness = None
        self.model = None

    def build_model(self):
        model = Sequential()                                
        for block in self.block_list:
            for layer in block.get_layers():                
                try:
                    layer.build_layer(model)
                except:
                    print("\nINDIVIDUAL ABORTED, CREATING A NEW ONE\n")
                    return -1
        return model

    def train_and_evaluate(self, model, dataset):
        print("Training", self.name)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(dataset['x_train'],
                            dataset['y_train'],
                            batch_size=dataset['batch_size'],
                            epochs=dataset['epochs'],
                            validation_data=(dataset['x_test'], dataset['y_test']),
                            shuffle=True)

        self.model = model                                    
        self.fitness = history.history['val_loss'][-1]        

        print("SUMMARY OF", self.name)
        print(model.summary())
        print("FITNESS: ", self.fitness)

        model.save(self.name + '.h5')                       
        save_network(self)                                  

    def asexual_reproduction(self, it, dataset):

        # 如果个体存在就读取它
        if os.path.isfile('net_' + str(it) + '.h5'):
            print("\n-------------------------------------")
            print("Loading individual net_" + str(it))
            print("--------------------------------------\n")
            individual = load_network('net_' + str(it))
            model = tf.keras.models.load_model(individual.name + '.h5')
            print("SUMMARY OF", individual.name)
            print(model.summary())
            print("FITNESS: ", individual.fitness)
            return individual

        # 否则通过变异父代来产生个体
        individual = Network(it)

        print("\n-------------------------------------")
        print("\nCreating individual", individual.name)
        print("--------------------------------------\n")

        individual.block_list = deepcopy(self.block_list)           

        print("----->Strong Mutation")
        individual.block_mutation(dataset)                          # 块变异
        individual.layer_mutation(dataset)                          # 层变异
        individual.parameters_mutation()                            # 参数变异

        model = individual.build_model()

        if model == -1:
            return self.asexual_reproduction(it, dataset)

        individual.train_and_evaluate(model, dataset)

        return individual

    def block_mutation(self, dataset):
        print("Block Mutation")

        print([(block.index, block.type) for block in self.block_list])

        bl = [block.index for block in self.block_list if block.type == 1]

        if len(bl) == 0:
            print("Creating a new block with two Convolutional layers and a Pooling layer")
            self.block_list[1].index = 2
            layerList1 = [
                Convolutional(filters=pow(2, randint(5, 8)),
                              filter_size=(3, 3),
                              stride_size=(1, 1),
                              padding='same',
                              input_shape=dataset['x_train'].shape[1:]),
                Convolutional(filters=pow(2, randint(5, 8)),
                              filter_size=(3, 3),
                              stride_size=(1, 1),
                              padding='same',
                              input_shape=dataset['x_train'].shape[1:])
            ]
            layerList2 = [
                Pooling(pool_size=(2, 2),
                        stride_size=(2, 2),
                        padding='same')
            ]
            b = Block(1, 1, layerList1, layerList2)
            self.block_list.insert(1, b)
            return

        block_idx = randint(1, max(bl))         # 随机选择一个块
        block_type_idx = randint(0, 1)          # 1是Conv2D; 0是Pooling或者Dropout
        mutation_type = randint(0, 1)           # 1是remove; 0是add


        layerList = self.block_list[block_idx].layerList1 if block_type_idx else self.block_list[block_idx].layerList2
        length = len(layerList)

        if mutation_type:                                       # 移除
            if length == 1:
                del self.block_list[block_idx]
            elif block_type_idx:
                pos = randint(0, length - 1)
                print("Removing a Conv2D layer at", pos)
                del layerList[pos]
            else:
                pos = randint(0, length - 1)
                print("Removing a Pooling/Dropout layer at", pos)
                del layerList[pos]
        else:                                                   # 添加
            if block_type_idx:
                print("Inserting a Convolutional layer")
                layer = Convolutional(filters=pow(2, randint(5, 8)),
                                      filter_size=(3, 3),
                                      stride_size=(1, 1),
                                      padding='same',
                                      input_shape=dataset['x_train'].shape[1:])
                layerList.insert(randint(0, length - 1), layer)
            else:
                if randint(0, 1):                               # 1是Pooling; 0是Dropout
                    print("Inserting a Pooling layer")
                    layer = Pooling(pool_size=(2, 2),
                                    stride_size=(2, 2),
                                    padding='same')
                    layerList.insert(randint(0, length - 1), layer)
                else:
                    print("Inserting a Dropout layer")
                    rate = choice([0.15, 0.25, 0.35, 0.50])
                    layer = Dropout(rate=rate)
                    layerList.insert(randint(0, length - 1), layer)

    def layer_mutation(self, dataset):
        print("Layer Mutation")

        # 随机选择一个块
        bl = [block.index for block in self.block_list if block.type == 1]

        if len(bl) == 0:
            return

        block_idx = randint(1, max(bl))
        block_type_idx = randint(0, 1)      # 1是Conv2D; 0是Pooling或Dropout

        layerList = self.block_list[block_idx].layerList1 if block_type_idx else self.block_list[block_idx].layerList2

        if len(layerList) == 0:
            if block_type_idx:
                layer = Convolutional(filters=pow(2, randint(5, 8)),
                                      filter_size=(3, 3),
                                      stride_size=(1, 1),
                                      padding='same',
                                      input_shape=dataset['x_train'].shape[1:])
                self.block_list[block_idx].layerList1.append(layer)
                return
            else:
                layer = Pooling(pool_size=(2, 2),
                                stride_size=(2, 2),
                                padding='same')
                self.block_list[block_idx].layerList2.append(layer)

        idx = randint(0, len(layerList) - 1)
        layer = layerList[idx]

        if layer.name == 'Conv2D':
            print("Splitting Conv2D layer at index", idx)
            layer.filters = int(layer.filters * 0.5)
            layerList.insert(idx, deepcopy(layer))
        elif layer.name == 'MaxPooling2D' or layer.name == 'AveragePooling2D':
            print("Changing Pooling layer at index", idx, "with Conv2D layer")
            del layerList[idx]
            conv_layer = Convolutional(filters=pow(2, randint(5, 8)),
                                       filter_size=(3, 3),
                                       stride_size=(2, 2),
                                       padding=layer.padding,
                                       input_shape=dataset['x_train'].shape[1:])
            layerList.insert(idx, conv_layer)

    def parameters_mutation(self):
        print("Parameters Mutation")
        for block in self.block_list:
            for layer in block.get_layers():
                if randint(0, 1):
                    layer.mutate_parameters()
