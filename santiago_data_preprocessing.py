
######################################################################################################
############ A MODULE WHERE THE NECESSARY FUNCTIONS TO PROCESS THE AUDIO FILES ARE INCLUDED ##########
######################################################################################################

# All rights reserved to: Santiago Álvarez-Buylla Puente
# Last modification: 24/06/2018

import numpy as np
import librosa
import math


def extend_file(input_file, sampling_rate, desired_length_seconds):

#A helper function to extend shorter-than-desired-duration audio files into a desired-duration file. It appends the same
#audio file at the end until achieving desired length
    #Inputs:  
        # input_file - 1D array of the audio wav file (resampled)
        # sampling_rate - sampling rate of the file
        # desired_file_duration - desired length of the audio files (in seconds)
    #Returns:
        # extended_file, an array containing the extended input_file
        
    desired_length_samples = int(sampling_rate * desired_length_seconds) 
    needed_length  = desired_length_samples - len(input_file)
    how_many = math.ceil(needed_length / len(input_file))
    extended_file = input_file
    
    for i in range(how_many):
        extended_file = np.concatenate((extended_file,input_file))

    extended_file = extended_file[0:desired_length_samples]
    extended_file.reshape(desired_length_samples,1)
    
    return extended_file


def voss(nrows, ncols=16, seed = 1):

#Generates pink noise using the Voss-McCartney algorithm. code taken from: https://www.dsprelated.com/showarticle/908.php
    #Inputs:
        #nrows: number of values to generate
        #rcols: number of random sources to add
    #Returns:
        #NumPy array

    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values



def normalize(input_file, amp = 1):
    """Normalizes a wave array so the maximum amplitude is +amp or -amp.
    ys: wave array
    amp: max amplitude (pos or neg) in result
    returns: wave array
    """
    high, low = np.absolute(np.amax(input_file)), np.absolute(np.amin(input_file))
    
    return amp * input_file / np.maximum(high, low)




def add_noise(input_file, db_SNR, index):
    '''Add pink noise to an audio file at a SNR specified 
    by the user
    Dependency: function voss'''
    
    #Generate normalized pink noise
    noise = normalize(voss(len(input_file)))
    
    # Shift it so that it has mean 0
    noise = noise - np.mean(noise)
    
    #RMS value of the noise
    rms_noise = np.sqrt(np.mean(noise**2))

    #RMS value of the input file
    rms_file = np.sqrt(np.mean(input_file**2)) 
    
    # For numerical stability: Avoid taking log(0)
    if rms_file == 0.0:
        print('problem in file %s' %index)
        rms_file = 10**(-8)

    #RMS value required for the noise to provide the specified db_SNR
    rms_noise_required = 10**(-db_SNR/10 + math.log10(rms_file))
    
    #Coefficient to multiply the noise vector
    coeff = rms_noise_required / rms_noise

    #Scale the noise vector with the calculated coeff
    noise_scaled = noise * coeff
    
    #Add the scaled noise to the input file
    input_file_noised = input_file + noise_scaled
    
    return input_file_noised


def zero_pad(input_file, sampling_rate, desired_length_seconds, noise = False, noise_all = False, db_SNR = 0, index = 0):

#A helper function to extend shorter-than-desired-duration audio files into a desired-duration file. 
    #Inputs: 
        # input_file - 1D array of the audio wav file (resampled)
        # sampling_rate - sampling rate of the file
        # desired_length_seconds - desired length of the audio file 
        # noise - Boolean. Whether to apply noise or not
        # noise_all - Boolean. Whether to apply noise to the whole extended file or only to original file
        # db_SNR - If noise = True, Signal to Noise Ratio (in dB) to be applied to the file. 
    #Returns:
        # zero_padded_file, a vector containing the zero-padded input_file
            
    desired_length_samples = sampling_rate * desired_length_seconds 
    needed_length  = int(desired_length_samples - len(input_file)) 
    
    if noise == True:
        if noise_all == True:
            
            input_file, noise_scaled = add_noise(input_file, db_SNR, index) 
            times_add_noise = math.ceil(needed_length / len(noise_scaled))
            noise_to_add = []

            for i in range(times_add_noise):
                noise_to_add.append(noise_scaled)

            noise_to_add = np.squeeze(np.array(noise_to_add).reshape(1,-1))
            zero_padded_file = np.concatenate((input_file,noise_to_add[0:needed_length]), axis = 0)
        
        else:
            input_file = add_noise(input_file, db_SNR, index)
            zero_padded_file = np.concatenate((input_file,np.zeros(needed_length)), axis = 0)
        
    else:
        zero_padded_file = np.concatenate((input_file, np.zeros(needed_length)), axis = 0) 
        
    
    return zero_padded_file
        
    

def get_spectrogram(x, window_size, hop_length):
    
#A function to calculate the power spectrogram of an audio file. The values of the spectrogram are in dB
#relative to the maximum value of each spectrogram. 
    #Inputs:
        # x - Array containing the single channel input audio file, size = (sampling_rate*duration, number_of_examples)
        # window_size - the time window size of the STFT frames, in samples
        # hop length - the step between consecutive windows, in samples
    #Returns:
        # features - An array of size = (frequency_bins_fft, time_frames, number_of_examples)

    output = []    
          
    for l in range(x.shape[1]): # i.e. number of examples 
        
        out,_ = librosa.core.spectrum._spectrogram(x[:,l], n_fft = window_size, hop_length = hop_length, power = 2)
        
        ref = np.amax(out)
        
        spectrogram_in_db = librosa.power_to_db((out), ref=ref) 
        
        output.append(spectrogram_in_db)
 
    features = np.transpose(np.array(output), axes =(1,2,0))

        
    return features


def get_mel_spectrogram(x, sampling_rate, number_mel_bands, window_size, hop_length):
    
#A function that calculates the spectrogram (S) and then builds a Mel filter (mel_basis = filters.mel(sr, n_fft, **kwargs)) and
#returns np.dot(mel_basis, S)
    #Inputs: 
        # x - matrix with all the audio clips from get_files_and_resample, dimensions (duration*sampling_rate_new, number_of_examples)
        # sampling_rate - sr of the audio clips (it matters for the computation of the different features)
        # number_mel_bands - the number of mel frequency bands to compute 
        # window_size - the time window size of the STFT frames, in samples
        # hop length - the step between consecutive windows, in samples
    #Returns:
        # features - 3D matrix of dimensions (number_mel_bands, number_time_frames , number_of_examples) containing...
        # ...the values of the mel-spectrogram

    output = []    
          
    for l in range(x.shape[1]): # i.e. number of examples 
        
        out = librosa.feature.melspectrogram(x[:,l], sampling_rate , n_fft = window_size, hop_length = hop_length, n_mels = number_mel_bands, power = 2)
        
        
        ref = np.amax(out)
        
        spectrogram_in_db = librosa.power_to_db((out), ref=ref) 
        
        output.append(spectrogram_in_db)
 
    features = np.transpose(np.array(output), axes =(1,2,0))

        
    return features



def order_files(x, y, class_ids):
# A helper function to order the audio files in order to make easier their 
# posterior mixing. The files are ordered by their class id. From 0 to 9
    #Inputs:
        # x - array with all the audio clips from get_files_and_resample, size = (duration*sampling_rate_new, number_of_examples)
        # y - array with the label ids

    ordered_list_files = []
    ordered_list_labels = []

    for i in range(len(class_ids)):
        
        for ii in range(x.shape[1]):
            
            if class_ids[i] == y[ii]:
                
                ordered_list_files.append(x[:,ii])
                ordered_list_labels.append(y[ii])

    ordered_x = np.array(ordered_list_files)
    ordered_y = np.array(ordered_list_labels)

    return np.transpose(ordered_x), ordered_y



def mix_files(x, y, examples_per_class = 28):
# A function to mix the files in pairs. The mixing is performed in such a way that
# for ten files corresponding to ten different classes in one folder, 45 (9+8+7+...+1) combinations are created
    #Inputs:
        # x - array of size (audio_samples, examples)
        # y - array containing the labels of each example,  size = (examples,)
        # class_ids - list containing 
        # examples_per_class - integer, representing the number of audio examples per class and folder that are to be combined
    #Returns:
        # array with the mixed audio files, array of size = (audio_samples, 28*45)
        # y_list - list containing the mixed classes ids, in the way [class_id_1-class_id_2,...]
    
    x_list = []
    y_list = []

    number_classes = 10

    for index in range(number_classes):       # The three for loops informally explained:
                                              # For each class in the present folder, run through every example of that class
        for i in range(examples_per_class):   # and combine it with one example of each other class (thus the 3rd loop)

            for ii in range(number_classes):

                if (index * examples_per_class + i) >= (ii * examples_per_class + i):
                    continue

                x_list.append(x[:, index * examples_per_class + i] + x[:, ii * examples_per_class + i])

                y_list.append(str(y[index * examples_per_class + i]) + '-' + str(y[ii * examples_per_class + i]))
        
    return np.transpose(np.array(x_list)), y_list



def one_hot_encode_mine(y_mixed):
# A function to one hot encode the labels of the combined files. That is, for every label, two "1" will appear in the 
# array, such as for example: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    #Inputs:
        #y_mixed - a list containing the combined ids of the audio examples in the form [3-2, 4-8, ...] for example
    #Returns:
        #y_mixed_one_hot - an array containing the one hot encoded class ids, size = (number_of_examples, 10)
    
    y_mixed_one_hot = []
    
    for i in range(len(y_mixed)):
        
        lista = [0,0,0,0,0,0,0,0,0,0]
        
        a = y_mixed[i].split('-')[0]
        b = y_mixed[i].split('-')[1]
        
        lista[int(a)] = 1
        lista[int(b)] = 1
        
        y_mixed_one_hot.append(np.array(lista))

    y_mixed_one_hot = np.array(y_mixed_one_hot)

    return y_mixed_one_hot



def split_into_clips (input_file, sampling_rate, clip_duration):
    
# A helper function to split files into equal length segments. Note: not used in the project
    #Inputs: 
        # input_file - 1D array of the audio wav file (resampled)
        # sampling_rate - sampling rate of the file
        # clip_duration - length of the clips we want to obtain (in seconds)
    #Returns:
        # clips, an array of dimensions (samples per clip, number_of_clips)
        
    samples_per_clip = int(sampling_rate*clip_duration)
    number_of_clips = len(input_file)//samples_per_clip  #if shorter than duration, will be discarded since number_of_clips = 0
    clips = np.zeros((samples_per_clip,number_of_clips))
    
    for l in range(number_of_clips):
        clips[:,l] = input_file[samples_per_clip*l:samples_per_clip*(l+1)] 
    
    return clips


def split_ordered_files(x_ordered, y_ordered, test_percentage = 10, num_classes = 10):
    
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    
    test_examples = x_ordered.shape[1] * test_percentage / 100
    train_examples = x_ordered.shape[1] - 2 * test_examples
    
    examples_per_class = int(x_ordered.shape[1] / num_classes)
    train_examples_per_class = int(train_examples / num_classes)
    test_examples_per_class = int(test_examples / num_classes)
    
    for i in range(num_classes):
      
        x_train.append(x_ordered[:, (i * examples_per_class) : (i * examples_per_class) + train_examples_per_class])
        y_train.append(y_ordered[(i * examples_per_class) : (i * examples_per_class) + train_examples_per_class])
        
        x_val.append(x_ordered[:, (i * examples_per_class) + train_examples_per_class : (i * examples_per_class) + train_examples_per_class + test_examples_per_class])
        y_val.append(y_ordered[(i * examples_per_class) + train_examples_per_class : (i * examples_per_class) + train_examples_per_class + test_examples_per_class])
        
        x_test.append(x_ordered[:, (i * examples_per_class) + train_examples_per_class + test_examples_per_class : (i * examples_per_class) + train_examples_per_class + 2 * test_examples_per_class])
        y_test.append(y_ordered[(i * examples_per_class) + train_examples_per_class + test_examples_per_class : (i * examples_per_class) + train_examples_per_class + 2 * test_examples_per_class])
        
    
    
    return np.array(x_train), np.array(y_train).flatten(), np.array(x_val), np.array(y_val).flatten(), np.array(x_test), np.array(y_test).flatten()



