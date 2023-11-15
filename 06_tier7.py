import os
import argparse

os.environ["SM_FRAMEWORK"] = "tf.keras"


import random
from itertools import combinations
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import segmentation_models as sm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["Unet", "Linknet"])
parser.add_argument("--trainset", type=int, help="option for the trainset: 0,1,2,3,4,5...,29. 999 for ALLIN.")

args = parser.parse_args()
MODEL = args.model
TRAINSET = args.trainset

LEADTIME = 1
BACKBONE = "vgg19"

if LEADTIME == 1:
    EPOCHS = 20 # same epochs to consider bigger VGG19
    BATCH = 32
elif LEADTIME == 12:
    EPOCHS = 15
    BATCH = 8


keys = np.load("./keys.npy", allow_pickle=True).item()
eval1000 = np.load("./eval1000.npy").tolist()
data_path = "./train_npy/"

if TRAINSET == 999:
    months_train = [i for i in range(1,13)]
    months_valid = [2,5,8,11]
else:
    # compute all combinations of training months
    list0 = [1, 5, 9, 12]
    list1 = [6, 7, 8]
    list2 = [2, 3, 4, 10, 11]

    combinations_list = []
    for combo1 in combinations(list1, 2):
        for combo2 in combinations(list2, 3):
            new_list = list0 + list(combo1) + list(combo2)
            combinations_list.append(new_list)

    # months for training and validation
    months_train = combinations_list[TRAINSET]
    months_valid = [i for i in range(1,13) if i not in months_train]

# keys for training and validation
keys_train = list()
keys_valid = list()

if LEADTIME == 1:
    mkey = 3
elif LEADTIME == 12:
    mkey = 16

for m in months_train:
    keys_train.extend(keys[m][mkey])

for m in months_valid:
    keys_valid.extend(keys[m][mkey])


# Generators for training
class DLSequence(Sequence):

    def __init__(self, index_instance, data_path=data_path, batch_size=8, lookback=2, shuffle=False):
        
        # init
        self.data_path = data_path
        self.index_instance = index_instance
        self.batch_size = batch_size
        self.lookback = lookback
        self.shuffle = shuffle

        # the idea is to add rotated examples to the untransformed ones
        # to that, we want to duplicate index instance to the amount of possible rotations
        # i.e. the index would be 4 times larger
        # Also, for each index item, there will be rotation factor.

        self.index_extended = self.index_instance * 4
        self.rotations_extended = []
        for i in range(4):
            self.rotations_extended.extend([i for j in range(len(self.index_instance))])


        if self.shuffle == True:
            
            np.random.seed(42)
            permuted_indices = np.random.permutation(len(self.index_extended))
            
            self.index_extended = np.array(self.index_extended)[permuted_indices].tolist()
            self.rotations_extended = np.array(self.rotations_extended)[permuted_indices].tolist()

            #np.random.shuffle(self.index_instance)
            
            print("Keys are shuffled")
        else:
            print("Keys are sequential")


    def __padding(self, array, from_shape=252, to_shape=256):
        # calculate how much to pad in respect with native resolution
        padding = int( (to_shape - from_shape) / 2)
        # for input shape as (batch, W, H, channels)
        array_transformed = np.pad(array, ((0,0),(padding,padding),(padding,padding),(0,0)), mode="constant", constant_values=0)
        
        return array_transformed


    def __transform(self, array, how="log0"):

        if how == "log0": # 0--6
            array_transformed = np.log(array+1)
        if how == "log1": # -1--6
            array_transformed = np.log(array+1/np.e)

        return array_transformed
    

    def __rotation(self, batch_of_arrays, batch_of_factors):

        # rotation factor is from [0,1,2,3]
        rotated_batch = np.empty_like(batch_of_arrays)

        for i in range(len(batch_of_arrays)):
            
            rotated_array = np.rot90(batch_of_arrays[i], k=batch_of_factors[i], axes=(0, 1))
            
            rotated_batch[i] = rotated_array

        return rotated_batch



    def __len__(self):
        #return int(np.ceil(len(self.index_instance) / self.batch_size))
        return int(np.ceil(len(self.index_extended) / self.batch_size))

    
    def __getitem__(self, idx):
        
        # create a keys for particular batch
        #keys_for_batch = self.index_instance[idx * self.batch_size : (idx + 1) * self.batch_size]
        keys_for_batch = self.index_extended[idx * self.batch_size : (idx + 1) * self.batch_size]
        rots_for_batch = self.rotations_extended[idx * self.batch_size : (idx + 1) * self.batch_size]   
        
        # batch construction
        # shape: batch, channels, W, H
        batch_x = np.array([np.array([np.load(os.path.join(self.data_path, f"{k}.npy")) for k in kb[:self.lookback] ]) for kb in keys_for_batch])
        batch_y = np.array([np.array([np.load(os.path.join(self.data_path, f"{k}.npy")) for k in kb[self.lookback:] ]) for kb in keys_for_batch])
        

        # batch re-ordering
        # from: batch, channels, W, H
        # to: batch, W, H, channels
        batch_x = np.moveaxis(batch_x, 1, -1)
        batch_y = np.moveaxis(batch_y, 1, -1)

        # padding
        batch_x = self.__padding(batch_x)
        batch_y = self.__padding(batch_y)

        # replace -1 flag in data by 0
        batch_x[batch_x == -1] = 0
        batch_y[batch_y == -1] = 0

        # transform data for training
        batch_x = self.__transform(batch_x)
        batch_y = self.__transform(batch_y)

        # rotation of batches
        batch_x = self.__rotation(batch_x, rots_for_batch)
        batch_y = self.__rotation(batch_y, rots_for_batch)
        
        return batch_x, batch_y

    def on_epoch_end(self):
        # Method called at the end of every epoch.
        # Only if we want to change our data between epochs
        pass


# construction of generators
if LEADTIME == 1:
    seq_train = DLSequence(keys_train, batch_size=BATCH, lookback=2, shuffle=True)
    seq_valid = DLSequence(keys_valid, batch_size=BATCH, lookback=2, shuffle=False)
elif LEADTIME == 12:
    seq_train = DLSequence(keys_train, batch_size=BATCH, lookback=4, shuffle=True)
    seq_valid = DLSequence(keys_valid, batch_size=BATCH, lookback=4, shuffle=False)


# model initialization
if LEADTIME == 1:
    input_frames = 2
elif LEADTIME == 12:
    input_frames = 4


if MODEL == "Unet":
    model = sm.Unet(backbone_name=BACKBONE,
                    encoder_weights=None, 
                    classes=LEADTIME, 
                    activation="linear",
                    input_shape=(256, 256, input_frames))

elif MODEL == "Linknet":
    model = sm.Linknet(backbone_name=BACKBONE,
                       encoder_weights=None, 
                       classes=LEADTIME, 
                       activation="linear",
                       input_shape=(256, 256, input_frames))


model.compile(optimizer="Adam", 
              loss="mean_squared_error")


# Set-up Callbacks
checkpoint_filepath = f'./models/{MODEL}_{BACKBONE}_{LEADTIME}_{TRAINSET}_tier7_repr.keras'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               save_weights_only=False,
                                                               monitor='val_loss',
                                                               mode='min',
                                                               save_best_only=True)

reduceLR_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                         min_delta=0.0001,
                                                         mode="min", 
                                                         patience=2, 
                                                         factor=0.1)

earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                      min_delta=0.0001,
                                                      mode="min", 
                                                      patience=10)

h = model.fit(seq_train, 
              validation_data=seq_valid, 
              epochs=EPOCHS, 
              callbacks=[reduceLR_callback], 
              verbose=2, 
              workers=8)

model.save(checkpoint_filepath)

print(f"Tier 7: {MODEL}, trainset: {TRAINSET}.")

###################################################################################################

# Evaluation on EVAL1000 keys

def get_input_data(list_instance, lookback=input_frames):

    data = []

    for timestamp in list_instance[-lookback:]:
        
        data.append(np.load(os.path.join(data_path, f"{timestamp}.npy"), allow_pickle=True))

    # the shape is channels, W, H
    data = np.array(data)

    # preprocessing
    # no data flag -1 to 0
    data[data == -1] = 0

    # move the channels to the end
    # --> W, H, channels
    data = np.moveaxis(data, 0, -1)  

    # pad to 256
    padding = 2
    data = np.pad(data, ((padding,padding),(padding,padding),(0,0)), mode="constant", constant_values=0)

    # log-transform
    data = np.log(data+1)

    # add virtual batch axis
    data = data[np.newaxis, ::, ::, ::]

    return data


def get_output_data(list_instance):

    data = []
    
    for timestamp in list_instance:
        
        data.append(np.load(os.path.join(data_path, f"{timestamp}.npy"), allow_pickle=True))

    data = np.array(data)

    # no postprocessing as we would like to keep -1 flag
    
    return data


def predict_nowcast(model_instance, input_data, lead_time=LEADTIME):

    if LEADTIME == 1:
        
        pred_ts = 12
        
        nwcst = []

        for _ in range(pred_ts):
            # make prediction
            pred = model_instance.predict(input_data, verbose=0)
            # append prediction to holder
            nwcst.append(pred)
            # append prediction to the input shifted on one step ahead
            input_data = np.concatenate([input_data[::, ::, ::, 1:], pred], axis=-1)
        
        # shape is 12, 1, 256, 256, 1
        nwcst = np.array(nwcst)
    
    elif LEADTIME == 12:
        
        # shape should be 1, 256, 256, 12
        nwcst = model_instance.predict(input_data, verbose=0)
        # moveaxis for channels
        nwcst = np.moveaxis(nwcst, -1, 0)
   
    return nwcst


def postprocess_output(array):

    # remove singular axes
    array = np.squeeze(array)

    # cut to 252*252
    padding=2
    to_shape=252
    array = array[::, padding:padding+to_shape, padding:padding+to_shape]

    # inverse log-transform
    array = np.exp(array) - 1

    # supress all Nans and infs
    array[~np.isfinite(array)] = 0

    array = np.where(array<0, 0, array)
    array = np.where(array>500, 500, array)

    return array


def eval_single_instance(keys_instance, model_instance):

    inp_keys = keys_instance[:4]
    out_keys = keys_instance[4:]

    # read input data
    # and preprocess it
    inp_data = get_input_data(inp_keys)

    # read output data as is (with -1 flag)
    out_data = get_output_data(out_keys)

    # producing nowcast
    nowcast = predict_nowcast(model_instance, inp_data)

    # postprocessing
    nowcast = postprocess_output(nowcast)

    return nowcast, out_data


def evaluate_on_val(keys_instance, model_instance):

    rmses = np.zeros((12,), dtype=float)
    
    for i, item in enumerate(keys_instance):

        output, target = eval_single_instance(item, model_instance)
        
        rmses += np.sum((np.square(target - output)) * (target != -1), axis=(1, 2))
            
    rmses /= len(keys_instance)
    
    return np.mean(np.sqrt(rmses))


#eval1000_rmse = evaluate_on_val(random.sample(eval1000, 150), model)
eval1000_rmse = evaluate_on_val(eval1000, model)

print(f"Eval1000 RMSE: {eval1000_rmse}")
