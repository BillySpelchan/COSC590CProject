# Importing the libraries
import numpy as np
import random
import math
import os.path

# Importing the Keras libraries
from keras.models import Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

#import matplotlib.pyplot as plt
#import pandas as pd
LEVEL_SIZE_TO_GENERATE = 120

TILES = "-oES?Q<>[]X"
TILE_EMPTY = 0
TILE_COIN = 1
TILE_ENEMY = 2
TILE_BREAKABLE = 3
TILE_QUESTION = 4
TILE_POWERUP = 5
TILE_LEFT_PIPE = 6
TILE_RIGHT_PIPE = 7
TILE_LEFT_PIPE_TOP = 8
TILE_RIGHT_PIPE_TOP = 9
TILE_BRICK = 10

PATH_TILES = "-oES?Q<>[]X*@"
TILE_EMPTY_MARIO = 11
TILE_COIN_MARIO = 12

ACTION_NONE = 0
ACTION_MOVING = 1
ACTION_JUMP1 = 2
ACTION_JUMP2 = 4
ACTION_JUMP3 = 8
ACTION_FALLING = 16


class Path_info:
    def __init__(self, level, mario_start_x, mario_start_y):
        self.source_level = level
        self.num_rows = level.shape[0]
        self.num_cols = level.shape[1]
        print ("level size is ", self.num_rows, self.num_cols)
        self.path_info = np.zeros((3, self.num_rows, self.num_cols))
        self.path_info[0,:,:] = level[:,:]
        self.path_info[1,mario_start_y,mario_start_x] = 1    # start position has walking action
        self.find_paths(mario_start_x, mario_start_y)
        self.tile_counts = None
        
    def find_paths(self, mario_start_x, mario_start_y):     
        end_x = self.num_cols     
        for col in range(mario_start_x+1, end_x): 
            freefall = False
            for row in range (0,13): # bottom row is death so ...
                if self.path_info[0,row, col] < 2: # if player can enter tile
                    state = 0
                    if (int(self.path_info[1, row, col-1]) & 1) == 1:
                        if self.path_info[0, row+1, col] < 2:
                            freefall = True
                    t = int(self.path_info[1, row+1, col-1])
                    if (t & 1) == 1: state = state | 2
                    if (t & 2) == 2: state = state | 4
                    if (t & 4) == 4: state = state | 8
                    if (t & 8) == 8: freefall = True
                    if (row < 12):
                        t= int(self.path_info[1, row+2, col-1])
                        if (t & 1) == 1: state = state | 4
                        if (t & 2) == 2: state = state | 8
                        if (t & 4) == 4: freefall = True
                    if (row < 11):
                        t= int(self.path_info[1, row+3, col-1])
                        if (t & 1) == 1: state = state | 8
                        if (t & 2) == 2: freefall = True
                    if (row < 10):
                        t= int(self.path_info[1, row+3, col-1])
                        if (t & 1) == 1: freefall = True
                    if (row > 0):
                        t= int(self.path_info[1, row-1, col-1])
                        if t>0: freefall = True
                    if (freefall): 
                        if (self.path_info[0, row+1, col] > 1):
                            state = state | 1
                            freefall = False
                        else:
                            state = state | 16
                    self.path_info[1, row, col] = state
            print(path_column_to_string(self.path_info, col), self.is_column_a_gap(col), self.is_column_an_obstacle(col))     

    def count_tile_usage(self):
        if self.tile_counts is None:
            self.tile_counts = np.zeros((11))
        for row in range(0, self.num_rows):
            for col in range(0, self.num_cols):
                self.tile_counts[int(self.path_info[0,row, col])] += 1
        for count in self.tile_counts:
            print(count)
            
    def is_completable(self):
        completable = False
        for row in range(0, self.num_rows):
            if self.path_info[1, row, self.num_cols-1] > 0:
                completable = True
                break
        return completable
    
    def num_empty_tiles(self):
        if self.tile_counts is None: self.count_tile_usage()
        return self.tile_counts[0]
    
    def num_reachable_tiles(self):
        reachable_count = 0
        for col in range(0, self.num_cols): 
            for row in range (0,13):
                if (self.path_info[1,row,col] > 0):
                   reachable_count += 1 
        return reachable_count
    
    def num_interesting_tiles(self):
        if self.tile_counts is None: self.count_tile_usage()        
        interesting_count = 0
        for tile_type in range(TILE_ENEMY, TILE_BRICK):
            interesting_count += self.tile_counts[tile_type]
        return interesting_count

    def is_column_a_gap(self, col):
        gap = self.path_info[0, self.num_rows-1, col] == TILE_EMPTY
        return gap
    
    def is_column_an_obstacle(self, col):
        if (col == 0): return False
        lowest_path = 0;
        for row in range(1,self.num_rows):
            if (int(self.path_info[1, row, col]) & ACTION_MOVING) > 0:
                lowest_path = row
        if (int(self.path_info[1, lowest_path, col-1]) & ACTION_MOVING) > 0:
            return False
        return True
    
    def calculate_leniency(self):
        if self.tile_counts is None: self.count_tile_usage()
        # the number of enemies plus the number of gaps minus the number of rewards.
        enemy_count = self.tile_counts[TILE_ENEMY]
        reward_count = self.tile_counts[TILE_COIN] + \
                self.tile_counts[TILE_POWERUP] + \
                self.tile_counts[TILE_QUESTION]
        gap_count = 0
        for col in range (2, self.num_cols):
            if self.is_column_a_gap(col): gap_count += 1
        return enemy_count + gap_count - reward_count
    
    def number_of_potential_jumps(self):
        potential_jumps = 0
        for col in range(0, self.num_cols): 
            for row in range (0,13):
                if (int(self.path_info[1,row,col]) and 1) == 1:
                    if self.path_info[0,row+1, col] > 1:
                        potential_jumps += 1
        return potential_jumps
    
    def number_of_required_jumps(self):
        gap_count = 0
        in_large_gap = False
        large_gap_count = 0
        obs_count = 0
        in_large_obs = False
        large_obs_count = 0
        for col in range (2, self.num_cols):
            if self.is_column_a_gap(col): 
                if (in_large_gap):
                    large_gap_count += 1
                else:
                    gap_count += 1
                    large_gap_count = 1
                    in_large_gap = True
            else:
                in_large_gap = False
                if (large_gap_count > 6):
                    gap_count += math.ceil((large_gap_count-6)/6)
                large_gap_count = 0
            
            if self.is_column_an_obstacle(col): 
                if (in_large_obs):
                    large_obs_count += 1
                else:
                    obs_count += 1
                    large_obs_count = 1
                    in_large_obs = True
            else:
                in_large_obs = False
                if (large_obs_count > 6):
                    obs_count += math.ceil((large_obs_count-6)/6)
                large_obs_count = 0
        return gap_count + obs_count
        
    def print_test_results(self):
        print ("Completable: ", self.is_completable())
        print ("Empty tiles: ", self.num_empty_tiles())
        print ("Reachable Tiles: ", self.num_reachable_tiles())
        print ("Interesting Tiles: ", self.num_interesting_tiles())
        print ("Leniency: ", self.calculate_leniency())
        print ("Number of potential jumps: ", self.number_of_potential_jumps())
        print ("Number of required jumps: ", self.number_of_required_jumps())

    def write_test_results_to_csv(self, csv_filename):
        num_tiles = self.path_info.shape[1] * self.path_info.shape[2]
        f = open(csv_filename, "a+")
        if self.is_completable():
            f.write('1,')
        else:
            f.write('0,')
        f.write(str(self.num_empty_tiles() / num_tiles))
        f.write(',')
        f.write (str(self.num_reachable_tiles() / num_tiles))
        f.write(',')
        f.write (str(self.num_interesting_tiles() / num_tiles))
        f.write(',')
        f.write (str( self.calculate_leniency()))
        f.write(',')
        f.write (str(self.number_of_potential_jumps()))
        f.write(',')
        f.write (str( self.number_of_required_jumps()))
        f.write('\n')
        f.close()

    def make_path_level(self):
        path_level = np.zeros((self.source_level.shape[0], 
                               self.source_level.shape[1]))
        for row in range(0, self.source_level.shape[0]):
            for col in range(0, self.source_level.shape[1]):
                tile = self.path_info[0, row, col]
                path = self.path_info[1, row, col]
                if (int(path) and 15) > 0:
                    tile += 11
                path_level[row,col] = tile
        return path_level
    
def convert_path_level_to_normal_level(path_level):
    normal_level = np.zeros((path_level.shape[0], 
                               path_level.shape[1]))
    for row in range(0, path_level.shape[0]):
        for col in range(0, path_level.shape[1]):
            tile = path_level[row, col]
            if tile >= 11:
                tile -= 11
            normal_level[row,col] = tile
    return normal_level
    
    

# --------------------------------------------------------------

"""
Simple method to take level segment and convert it into the input necessary
for the neural network.
"""
def grab_inputs_from_level(level, x, y, size, num_tiles):
    tx = x
    ty = y
    data = np.zeros((size, num_tiles))
    for t in range(0,size):
        tile = level[ty][tx]
        data[t][int(tile)] = 1
        ty -= 1
        if (ty < 0):
            ty = 13
            tx += 1
    return data

def turn_output_into_tile(array, byRandom=False):
    if byRandom:
        weight = random.random()
        picked = 0
        indx = 0
        while (weight > 0) and (indx < len(TILES)):
            if weight <= array[indx]:
                picked = indx
                weight = 0
            else:
                weight -= array[indx]
                indx += 1
    else:
        picked = 0
        for indx in range(1,len(array)):
            if array[indx] > array[picked]:
                picked = indx
    return picked


def build_training_set(level, start_col, end_col, chunk_size, num_tiles):
    true_end = end_col - int(chunk_size / 14)
    if (true_end <= start_col):
        true_end = start_col+1
    num_tests = (true_end - start_col) * 14
    training = np.zeros((num_tests, chunk_size, num_tiles))
    results = np.zeros((num_tests, num_tiles))
    test_id = 0
    for x in range(start_col, true_end):
        for y in range(0, 14):
            temp = grab_inputs_from_level(level, x, y, chunk_size + 1, num_tiles)
            training[test_id] = temp[0:chunk_size,:]
            results[test_id] = temp[chunk_size]
            test_id += 1
    return training, results

class TrainingManager:
    def __init__(self, memory, num_tiles):
        self.test_sets = list()
        self.memory = memory
        self.num_tiles = num_tiles
        self.train_set = None
        self.train_result = None
        self.test_set = None
        self.test_results = None
        
    def add_level_to_test_set(self, level, start_col, end_col):
        self.test_sets.append(build_training_set(level, start_col, end_col, 
                                                 self.memory, 
                                                 self.num_tiles ))
        
    def make_test_set(self, hold_rate = .1):
        total_tests = 0
        for test in self.test_sets:
            total_tests = total_tests + test[0].shape[0]
        # todo properly split
        self.train_set = np.zeros((total_tests, self.memory, self.num_tiles))
        self.train_results = np.zeros((total_tests, self.num_tiles))
        current = 0
        for test in self.test_sets:
            for indx in range(0,test[0].shape[0]):
                self.train_set[current] = test[0][indx]
                self.train_results[current] = test[1][indx]
                current += 1
        print("number of tests is ", current)
        
# --------------------------------------------------------------


class BasicModel:
    def __init__(self, memory):
        self.memory = memory

    def predict_next_tile(self, raw_seed_input):
        return raw_seed_input[self.memory-14]


# --------------------------------------------------------------
def build_simple_model(dropout = 0.05):
    model = Sequential()
    #input layer and first (only in this model) 
    model.add(SimpleRNN(units = 200, return_sequences = True, input_shape = (200, len(TILES))))
    model.add(Dropout(dropout))
    model.add(SimpleRNN(units = 200, return_sequences = True))
    model.add(Dropout(dropout))
    model.add(SimpleRNN(units = 200, return_sequences = False))
    model.add(Dropout(dropout))
    #ouput layers
    model.add(Dense(units = len(TILES)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    return model


def build_lstm_model(dropout = 0.05):
    model = Sequential()
    #input layer and first (only in this model) 
#    model.add(LSTM(units = 200, return_sequences = False, input_shape = (200, len(TILES))))
    model.add(LSTM(units = 200, return_sequences = True, input_shape = (200, len(TILES))))
    if (dropout > 0):
        model.add(Dropout(dropout))
    model.add(LSTM(units = 200, return_sequences = True))
    if (dropout > 0):
        model.add(Dropout(dropout))
    model.add(LSTM(units = 200))
    #ouput layers
    model.add(Dense(units = len(TILES)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    return model


class RNNModel:
    def __init__(self, memory, saved_model_name=None):
        self.memory = memory
        if saved_model_name is None:
            self.model = build_simple_model()
        else:
            self.model = load_model(saved_model_name)
        #training, results = build_training_set(level, 0, 100, 200, len(TILES))
        #self.model.fit(training, results, batch_size=32, epochs=3)
    
    def train(self, training_manager, epochs_to_run):
        test = training_manager.train_set
        res = training_manager.train_results
        self.model.fit(test, res, batch_size=32, epochs=epochs_to_run)
        
    def save(self, saved_model_name="rnn.h5"):
        self.model.save(saved_model_name)
        
    def predict_next_tile(self, raw_seed_input):
        seed = np.zeros((1, self.memory, len(TILES)))
        for cntr in range(0, self.memory):
            seed[0,cntr, int(raw_seed_input[cntr])] = 1
        prediction = self.model.predict(seed, batch_size=1)        
        return turn_output_into_tile(prediction[0], True)
  

class LSTMModel:
    def __init__(self, memory, saved_model_name=None):
        self.memory = memory
        if saved_model_name is None:
            self.model = build_lstm_model()
        else:
            self.model = load_model(saved_model_name)
        #training, results = build_training_set(level, 0, 100, 200, len(TILES))
        #self.model.fit(training, results, batch_size=32, epochs=3)
    
    def train(self, training_manager, epochs_to_run):
        test = training_manager.train_set
        res = training_manager.train_results
        self.model.fit(test, res, batch_size=32, epochs=epochs_to_run)
        
    def save(self, saved_model_name="lstm.h5"):
        self.model.save(saved_model_name)
        
    def predict_next_tile(self, raw_seed_input):
        seed = np.zeros((1, self.memory, len(TILES)))
        for cntr in range(0, self.memory):
            seed[0,cntr, int(raw_seed_input[cntr])] = 1
        prediction = self.model.predict(seed, batch_size=1)        
        return turn_output_into_tile(prediction[0], True)
    
class LSTMPathModel:
    def __init__(self, memory, saved_model_name=None):
        self.memory = memory
        if saved_model_name is None:
            self.model = Sequential()
            self.model.add(LSTM(units = 200, return_sequences = True, 
                                input_shape = (200, len(PATH_TILES))))
            self.model.add(Dropout(.1))
            self.model.add(LSTM(units = 200, return_sequences = True))
            self.model.add(Dropout(.1))
            self.model.add(LSTM(units = 200))
            #ouput layers
            self.model.add(Dense(units = len(PATH_TILES)))
            self.model.add(Activation("softmax"))
            self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
        else:
            self.model = load_model(saved_model_name)
    
    def train(self, training_manager, epochs_to_run):
        test = training_manager.train_set
        res = training_manager.train_results
        self.model.fit(test, res, batch_size=32, epochs=epochs_to_run)
        
    def save(self, saved_model_name="lstm_path.h5"):
        self.model.save(saved_model_name)
        
    def predict_next_tile(self, raw_seed_input):
        seed = np.zeros((1, self.memory, len(PATH_TILES)))
        for cntr in range(0, self.memory):
            seed[0,cntr, int(raw_seed_input[cntr])] = 1
        prediction = self.model.predict(seed, batch_size=1)        
        return turn_output_into_tile(prediction[0], True)
    
# --------------------------------------------------------------

class LevelGenerator:
    def __init__(self, seed_level):
        self.seed_level = seed_level
        
    def generate_level(self, model, chunk_size, columns):
        generated_level = np.zeros((14, columns))
        running_chunk = np.zeros((chunk_size))
        predict = 0
        count = 0
        for col in range (0,columns):
            for row in range (0,14):
                if count < chunk_size:
                    predict = self.seed_level[13-row, col]
                    count += 1
                else:
                    predict = model.predict_next_tile(running_chunk)
                for cntr in range (0, chunk_size-1):
                    running_chunk[cntr] = running_chunk[cntr+1]
                running_chunk[chunk_size-1] = predict
                generated_level[13-row, col] = predict
        return generated_level


# --------------------------------------------------------------                  
                
    
def load_level(level_name):
    f = open(level_name, "r")
    raw_level = f.readlines()
    f.close()
    #print(raw_level)
    num_cols = len(raw_level[0])-1
    print("Debug - level columns = ", num_cols)

    level_data = np.zeros((14, num_cols))
    for row in range(0,14) :
        for col in range(0,num_cols) :
            tile = TILES.index(raw_level[row][col])
            #todo error correction for invalid tiles
            level_data[row][col] = tile;
    return level_data

def save_level(level, level_name):
    f = open(level_name, "w")
    for row in range(0,14):
        text = ""
        for col in range (0, level.shape[1]):
            text += PATH_TILES[int(level[row, col])]
        text += "\n"
        f.write(text)
    f.close

def path_column_to_string(path_info, col):
    s = ""
    for row in range (0,14):
        if(path_info[1, 13-row, col] != 0):
            if path_info[0, 13-row, col] < 2:
                s = s + '*'
            else:
                s = s + 'invalid'
        else:
            s = s + TILES[int(path_info[0, 13-row, col])]
    return s


def is_tile_empty(level, x, y):
    if y > 13:
        return True
    if (level[y, x] < 2):
        return True
    return False;
 


def coordinate_to_string_position(x,y, snaking = False, offset=0 ):
    ty=y
    if snaking and (x % 2 == 1):
        ty = 13 - y
    p = x * 14 + ty + offset
    return p

def string_position_to_coordinate(pos, snaking=False):
    x = pos // 14
    y = pos % 14
    if snaking and (x % 2 == 1):
        y = 13 - y
    return x,y


def testing():
    for t in range(200):
        x,y = string_position_to_coordinate(t)
        n = coordinate_to_string_position(x,y)
        if n != t:
            print("Problem with coordinate/position conversion non-snaking")
            print(x,y,n)
    for t in range(200):
        x,y = string_position_to_coordinate(t,True)
        n = coordinate_to_string_position(x,y,True)
        if n != t:
            print("Problem with coordinate/position conversion non-snaking")
            print(x,y,n)


def generate_human_levels_csv():
    path = Path_info(level, 1, 12)
    path.print_test_results()
    path.write_test_results_to_csv("original_levels.csv")
    path = Path_info(level2, 1, 12)
    path.print_test_results()
    path.write_test_results_to_csv("original_levels.csv")
    path = Path_info(level3, 1, 12)
    path.print_test_results()
    path.write_test_results_to_csv("original_levels.csv")    

def batch_generate_levels(model, seed, batch_size, save_base, save_levels=False):
    gen_level = LevelGenerator(seed)
    for cntr in range(0,batch_size):
        print("*** ", save_base," Generating ",  (cntr+1), " of ", batch_size)
        generated = gen_level.generate_level(model, 200, LEVEL_SIZE_TO_GENERATE)
        if save_levels:
            save_level(generated, save_base+str(cntr)+".txt")
        # normal levels unnefected by following, but to be generic...
        normal_level = convert_path_level_to_normal_level(generated)
        path = Path_info(normal_level, 1, 12)
        path.print_test_results()
        path.write_test_results_to_csv(save_base+".csv")
    print(save_base," Generation completed!")

def build_and_test_RNN(training_manager, batch_size):
    if os.path.exists("rnn.h5"):
        print("The model already exists")
        rnn_model = RNNModel(200, "rnn.h5")
    else:
        print("need to create the model")
        rnn_model = RNNModel(200)
        rnn_model.train(training_manager, 10)
        rnn_model.save()
    batch_generate_levels(rnn_model, level, batch_size, "rnn")    
   
def build_and_test_LSTM(training_manager, batch_size):
    if os.path.exists("lstm.h5"):
        print("The model already exists")
        lstm_model = LSTMModel(200, "lstm.h5")
    else:
        print("need to create the model")
        lstm_model = LSTMModel(200)
        lstm_model.train(training_manager, 10)
        lstm_model.save()
    batch_generate_levels(lstm_model, level, batch_size, "lstm")    

def build_and_test_LSTMPath(training_manager, batch_size):
    if os.path.exists("lstm_path.h5"):
        print("The model already exists")
        lstmpath_model = LSTMPathModel(200, "lstm_path.h5")
    else:
        print("need to create the model")
        lstmpath_model = LSTMPathModel(200)
        lstmpath_model.train(training_manager, 10)
        lstmpath_model.save()
    batch_generate_levels(lstmpath_model, level, batch_size, "lstm_path")    

# level is my seed level    
level = load_level("levels/mario-1-1.txt")
level2 = load_level("levels/mario-1-2.txt")
level3 = load_level("levels/mario-1-3.txt")

# optional mehthod used once to generate our baseline csv
# UNCOMMENT to run
#generate_human_levels_csv()


# set up the training manager for non-path models
training_manager = TrainingManager(200, len(TILES))
training_manager.add_level_to_test_set(level, 0, level.shape[1]-15)
training_manager.add_level_to_test_set(level2, 0, level2.shape[1]-15)
training_manager.add_level_to_test_set(level3, 0, level3.shape[1]-15)
training_manager.make_test_set()

path_training_manager = TrainingManager(200, len(PATH_TILES))
level1p = Path_info(level, 1, 12).make_path_level()
path_training_manager.add_level_to_test_set(level1p, 0, level1p.shape[1]-15)
level2p = Path_info(level2, 1, 12).make_path_level()
path_training_manager.add_level_to_test_set(level2p, 0, level2p.shape[1]-15)
level3p = Path_info(level3, 1, 12).make_path_level()
path_training_manager.add_level_to_test_set(level3p, 0, level3p.shape[1]-15)
path_training_manager.make_test_set()

'''
my_level = gen_level.generate_level(model, 200, 150)
path = Path_info(my_level, 1, 12)
path.count_tile_usage()
path.print_test_results()
'''

#path_level = path.make_path_level()
#normal_level = convert_path_level_to_normal_level(path_level)
#save_level(normal_level, "test.txt")

# original code for testing baching
'''
model = BasicModel(200)
batch_generate_levels(model, level, 5, "basic_test", True)
'''
build_and_test_RNN(training_manager, 20)
build_and_test_LSTM(training_manager, 20)
build_and_test_LSTMPath(path_training_manager, 20)


'''
testing()
'''
