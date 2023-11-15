import os
import numpy as np
import tensorflow as tf
import h5py


LEADTIME = 1

model_path = "./models_final/"

"""
model_names = ["Unet_vgg16_1_19_tier5.keras"]
output_names= ["tier5_Unet_19.hdf5"]
LOOKBACKS = [4]
"""

model_names = ["Unet_vgg16_1_19_tier3.keras", 
               "Unet_vgg16_1_21_tier3.keras", 
               "Unet_vgg16_1_24_tier3.keras", 
               "Unet_vgg16_1_19_tier5.keras", #wrongly produced number 20
               "Unet_vgg16_1_23_tier5.keras", 
               "Unet_vgg19_1_24_tier7.keras", 
               "Linknet_vgg19_1_25_tier7.keras", 
               "Linknet_vgg19_1_27_tier7.keras"]

output_names = ["tier3_Unet_19.hdf5",
                "tier3_Unet_21.hdf5",
                "tier3_Unet_24.hdf5",
                "tier5_Unet_19.hdf5",
                "tier5_Unet_23.hdf5",
                "tier7_Unet_24.hdf5",
                "tier7_Linknet_25.hdf5",
                "tier7_Linknet_27.hdf5"]

LOOKBACKS = [2, 2, 2, 4, 4, 2, 2, 2] # 4 for tier5, 2 for other tiers


# get test keys
with h5py.File("2022-test-public.hdf5", mode="r") as test:
    test_keys = sorted(test.keys())
    input_keys = [test_keys[i*4:i*4+4] for i in range(int(len(test_keys)/4))]


def get_data(list_instance, lookback):

    data = []

    for timestamp in list_instance[-lookback:]:
        
        data.append(np.array(test[timestamp]['intensity']))

    # the shape is channels, W, H
    data = np.array(data)

    # preprocessing
    # both Nan (-2e6) and zeros (-1e6) go to zeros
    data[data == -1e6] = 0
    data[data == -2e6] = 0

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

def predict_nowcast(model_instance, input_data, LEADTIME=1):

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

    # get rid of negative values
    array = np.where(array<0, 0, array)

    # get rid of physically unplausible values
    array = np.where(array>500, 500, array)

    # convert to float16
    array = array.astype("float16")

    return array

def get_output_list(list_instance, pred_ts=12):

    last_ts = int(list_instance[-1])

    return [str(last_ts + 600*(i+1)) for i in range(pred_ts)]


for model_name, output_name, LOOKBACK in zip(model_names, output_names, LOOKBACKS):

    model = tf.keras.models.load_model(os.path.join(model_path, model_name))
    with h5py.File("2022-test-public.hdf5", mode="r") as test:
        with h5py.File(f"./outputs/{output_name}", mode="a") as output_file:
            
            for input_list in input_keys:

                # create a list of output ts
                output_list = get_output_list(input_list)

                # get an input data
                input_data = get_data(input_list, LOOKBACK)

                # make a predition
                nowcast = predict_nowcast(model, input_data)

                # postprocessing
                nowcast = postprocess_output(nowcast)

                # write to a file
                for i, out_ts in enumerate(output_list):

                    output_file.create_group(out_ts)
                    output_file[out_ts].create_dataset('intensity', data=nowcast[i])