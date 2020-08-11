import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random


def add_gaussian_noise(n, scale=1, start=0, bottom=float("-inf"), top=float("inf")):
    """Generate a numpy array populated with Gaussian noise values.
    
    Parameters:
    n (int): Number of time points
    scale (float): The standard deviation of the Gaussian distribustion from which the noise is sampled.
    start (int): The time point at which the noise begins.  Previous time points are set to 0.
    bottom (float): The lower limit of the noise values.
    top (float): The upper limit of the noise values.
    """
    
    t = np.zeros(n)
    for i in range(start,n):
        t[i] = min(max(np.random.normal(loc=0, scale=scale),bottom),top)
    return np.array(t)

def add_sine_wave(n, period):
    """Generate a numpy array of length 'n', populated with values from a sine wave."""
    
    start = 2*np.pi*random.random()
    return np.array([np.sin(start + 2*np.pi*i/(period-1)) for i in range(n)])

def add_chaotic_drift(n, scale=0.1):
    """Generate a numpy array of length 'n' whose values drift.
    
    The amount that each value drifts from the previous value is sampled from a continuous uniform distribution
    with a min of -0.5*scale and a max of 0.5*scale.
    """
    
    t = [(random.random()-0.5)*scale for i in range(n)]
    for i in range(1,len(t)):
        t[i] = t[i-1] + t[i]
    return np.array(t)

def add_anchored_drift(n, top=1, bottom=-1):
    """Generate a numpy array of length 'n' whose values drift but stay within the range 'top' to 'bottom'."""
    
    t = np.zeros(n)
    delta = np.random.normal(loc=0, scale=0.35)
    for i in range(1,n):
        sign_ = 1 if t[i-1]>=0 else -1
        move_away_probability = 0.5 - sign_*t[i-1]/2
        scale = max(1/50, 0.7*move_away_probability)
        distance = abs(np.random.normal(loc=0, scale=scale))
        move_away = 1 if np.random.binomial(n=1,p=move_away_probability) else -1
        delta = 0.9 * delta + (1-0.9) * sign_ * distance * move_away
        t[i] = t[i-1] + delta
        if t[i] > top or t[i] < bottom:
            t[i] = t[i-1]
    return t

def add_random_shifts(n):
    """Generate a numpy array of length 'n' whose values shift up and down randomly."""
    
    t = np.zeros(n)
    i=1
    while i < n:
        if np.random.binomial(n=1, p=0.005):
            duration = min((n-i), int(np.random.gamma(1.5,4)))
            direction = (1 if (t[i-1]<=0) else -1) * (1 if np.random.binomial(n=1, p=0.8) else -1)
            height = np.random.gamma(3,0.7)
            for j in range(duration):
                t[i+j] = t[i-1] + height * j/duration * direction
            i+=duration
        else:
            t[i] = t[i-1]
            i+=1
    return t

def add_compressor(n, height):
    """Generate a numpy array of length 'n' whose values simulate a compressor cycle.
    
    The array values move up and down cyclically.  The period of each cycle can change slightly over time,
    but the 'height' remains constant. 
    """
    
    baseline = max(np.random.gamma(6,1), 0.5)
    num_cycles = round(n/baseline*1.5)
    drift = add_anchored_drift(num_cycles) * baseline/10
    period = np.random.normal(288 / random.randint(1,3))
    cycle = add_sine_wave(n=num_cycles, period=period) * baseline/40
    intervals = [baseline+drift[i]+cycle[i] for i in range(num_cycles)]
    t = np.zeros(n)
    end_of_current_cycle = 0
    end_of_prev_cycle = 0
    next_cycle = 0
    for i in range(1, n):
        while i > end_of_current_cycle:
            end_of_prev_cycle = end_of_current_cycle
            end_of_current_cycle = end_of_current_cycle + intervals[next_cycle]
            next_cycle += 1
        halfway = end_of_prev_cycle + (end_of_current_cycle-end_of_prev_cycle)/2
        if i <= halfway:
            t[i] = height * (i - end_of_prev_cycle) / (halfway - end_of_prev_cycle)
        else: t[i] = height * (end_of_current_cycle - i) / (end_of_current_cycle - halfway)
    return t

def spawn_defrost(height, num_readings_up_average, num_readings_down_average):
    """Generate a numpy array whose values simulate the temperatures of a single defrost event."""
    
    num_readings_up = int(max(round(np.random.normal(loc=num_readings_up_average, scale=num_readings_up_average/5)),1))
    num_readings_down = int(max(round(np.random.normal(loc=num_readings_down_average, scale=num_readings_down_average/5)),1))
    t_up = [height/num_readings_up*i for i in range(1,num_readings_up+1)]
    t_down = [(height-height/num_readings_down*i) for i in range(1,num_readings_down)]
    t = t_up+t_down
    return np.array(t)



def add_defrosts(n, height_baseline, anomalous=False, all_anomalous=True, end_in_defrost=False):
    """Generate a numpy array of length 'n' whose values simulate a defrost cycle.
    
    Each defrost event is simulated separately, which creates a small amount of variation between defrost events
    in the same array.
    Each defrost event's height and each interval between two defrost is determined by applying a small amount
    of random variation to a baseline value that remains constant throughout the array.
    A defrost event is called 'anomalous' when gaussian noise is added to its height and to the interval between
    it and the next defrost event.
    If anomalous is set to True, then setting all_anomalous to True makes all defrosts anomalous.
    If anomalous is set to True and all_anomalous is set to False, then only the second half of the defrosts 
    will be anomalous.
    
    Parameters:
    n (int): Number of time points
    anomalous (bool): If True, make at least some of the defrosts are anomalous (inconsistent in height and period).
    all_anomalous (bool): If True, make all defrosts anomalous, rather than just the second half.
    end_in_defrost (bool): If True, a defrost will be occurring during the last time point in the array.
    """
    
    t = np.empty(shape=(0))
    labels = []
    interval_baseline = max(np.random.gamma(3,22), 12)
    anomaly_range = (0 if (not anomalous) else (n if all_anomalous else (int(n/2))))
    num_readings_up_average = round(min(max(np.random.gamma(1,5),1), interval_baseline/4))
    num_readings_down_average = round(min(max(np.random.gamma(1,5),1), interval_baseline/4))
    if end_in_defrost:
        height = height_baseline + add_gaussian_noise(1, bottom=-height_baseline/2, top=height_baseline*2, \
                                                      scale=(height_baseline/2 if anomalous else height_baseline/5))
        defrost = spawn_defrost(height, num_readings_up_average, num_readings_down_average)
        cutoff_defrost = defrost[random.randint(0,(len(defrost)-1)):]
        t = np.append(t, cutoff_defrost)
        labels = labels + ([True]*len(cutoff_defrost))
        interval = int(interval_baseline + \
                       add_gaussian_noise(1, bottom=-interval_baseline, \
                                          scale=(interval_baseline/2 if anomalous else interval_baseline/20)))
        t = np.append(t, np.zeros(interval))
        labels = labels + ([False]*interval)
    else:
        cutoff_interval = random.randint(0,round(interval_baseline))
        t = np.append(t, np.zeros(cutoff_interval))
        labels = labels + ([False]*cutoff_interval)
    while len(t) < n:
        height_noise_scale = (height_baseline/2 if (len(t)<anomaly_range) else height_baseline/5)
        height = height_baseline + add_gaussian_noise(1, bottom=-height_baseline/2, top=height_baseline*2, \
                                                      scale=height_noise_scale)
        defrost = spawn_defrost(height, num_readings_up_average, num_readings_down_average)
        t = np.append(t, defrost)
        labels = labels + ([True]*len(defrost))
        interval_noise_scale = (interval_baseline/2 if anomalous else interval_baseline/20)
        interval = int(interval_baseline + add_gaussian_noise(1, bottom=-interval_baseline, scale=interval_noise_scale))
        t = np.append(t, np.zeros(interval))
        labels = labels + ([False]*interval)
    t = np.flip(t[0:n])
    labels = np.flip(labels[0:n])
    return t, labels, interval_baseline

def add_random_events(n):
    """Generate a numpy array of length 'n' whose values simulate random spikes in a time series.
    
    The random spikes can go either up or down, and their heights fluctuate more than defrost heights.
    """
    
    num_events = int(abs(np.random.normal(loc=int(n/50), scale=int(n/50))))
    t = np.zeros(n)
    for i in range(num_events):
        height = np.random.gamma(1,1)
        num_readings_up = round(max(np.random.gamma(1.5,3),1))
        num_readings_down = round(max(np.random.gamma(1.5,3),1))
        defrost = spawn_defrost(height=height, num_readings_up=num_readings_up, num_readings_down=num_readings_down)
        event = [0]
        for up in range(num_readings_up):
            event = event + [event[up-1] + np.random.normal(loc=height/num_readings_up, scale = height/num_readings_up)]
        for down in range(num_readings_down):
            event = event + [event[num_readings_up+down-1] + np.random.normal(loc=height/num_readings_up, \
                                                                              scale = height/num_readings_up)]
        direction = (1 if np.random.binomial(n=1,p=0.5) else -1)
        event = event*direction
        location = random.randint(0,n-1-len(event))
        t[location:(location+len(event))] = event
    return np.array(t)

def add_fake_defrosts(n, height_baseline):
    """Generate a numpy array of length 'n' whose values simulate counterfeit defrosts in a time series.
    
    Each counterfeit defrost's height is determined by applying a small amount of random variation to a 
    baseline value, the 'height_baseline', which remains constant throughout the array.
    """
    
    num_defrosts = int(abs(np.random.normal(loc=int(n/200), scale=int(n/200))))
    t = np.zeros(n)
    for i in range(num_defrosts):
        height = np.random.normal(loc=height_baseline, scale=(height_baseline/5))
        num_readings_up_average = round(max(np.random.gamma(1.5,2.5),1))
        num_readings_down_average = round(max(np.random.gamma(1.5,2.5),1))
        defrost = spawn_defrost(height, num_readings_up_average, num_readings_down_average)
        location = random.randint(0,n-1-len(defrost))
        t[location:(location+len(defrost))] = defrost
    return np.array(t)

def add_fake_defrost_to_end(n, height_baseline):
    """Generate a numpy array of length 'n' that ends in a defrost.  All other values are zero"""
    
    t = np.zeros(n)
    height = np.random.normal(loc=height_baseline, scale=(height_baseline/5))
    num_readings_up_average = round(max(np.random.gamma(1.5,2.5),1))
    num_readings_down_average = round(max(np.random.gamma(1.5,2.5),1))
    defrost = spawn_defrost(height, num_readings_up_average, num_readings_down_average)
    cutoff = random.randint(1,len(defrost))
    t[-cutoff:] = defrost[:cutoff]
    return t

def generate_simulation(n, defrosts=None, compressor=None, chaotic_drift=None, gaussian_noise=None, \
                        random_events=None, fake_defrosts=None, random_shifts=None, anomalous=False, all_anomalous=True, \
                        end_in_defrost=False, end_in_fake_defrost=False, chaotic_drift_scale=0.2, \
                        noise_scale=0.5, verbose=False):
    """Simulate a time series of refrigeration temperature readings.
    
    Simulate a refrigeration temperature time series by built by adding together component time series, each of 
    which are simulated separately.  Component time series include a compressor cycle, a defrost cycle, drift, 
    gaussian noise, random events, counterfeit defrosts, and random shifts.  Each of these components is 
    associated with a boolean parameter that controls whether or not the component is added to the time series.
    If any of these parameters are left as None (their default), then that component will have some random chance 
    of being included.
    
    Parameters:
    n (int): Length of the time series.
    defrosts (bool or int): If True or 1, add a defrost cycle to the time series.
    compressor (bool or int): If True or 1, add a compressor cycle to the time series.
    chaotic_drift (bool or int): If True or 1, add drift to the time series.
    gaussian_noise (bool or int): If True or 1, add gaussian noise to the time series.
    random_events (bool or int): If True or 1, add random spikes up and down to the time series.
    fake_defrosts (bool or int): If True or 1, add counterfeit defrosts to the time series.
    random_shifts (bool or int): If True or 1, add random shifts up and down to the time series.
    anomalous (bool): If True, make at least some of the defrosts are anomalous (inconsistent in height and period).
    all_anomalous (bool): If True, make all defrosts anomalous, rather than just the second half.
    end_in_defrost (bool): If True, a defrost will be occurring during the last time point in the array.
    end_in_fake_defrost (bool): If True, a counterfeit defrost will be occurring during the last time point in the array.
    chaotic_drift_scale (float): The variance of the distribution from which the drift step sizes are sampled.
    noise_scale (float): The variance of the Gaussian distribution from which the noise is sampled.
    verbose (bool): If True, print the components that were added to the time series.
    
    Returns:
    simulated_run (numpy array): Simulated time series of temperature readings.
    labels (numpy array): Simulated time series of defrost labels.
    interval_baseline (float): Defrost periods' baseline value before random variance is added.
    """
    
    if (end_in_defrost and end_in_fake_defrost):
        raise ValueError('Cannot end in both a fake and real defrost')
    if defrosts == None:
        defrosts = np.random.binomial(n=1,p=0.9)
    if compressor == None:
        compressor = np.random.binomial(n=1,p=0.5)
    if chaotic_drift == None:
        chaotic_drift = np.random.binomial(n=1,p=0.5)
    if gaussian_noise == None:
        gaussian_noise = np.random.binomial(n=1,p=0.5)
    if random_events == None:
        gaussian_noise = np.random.binomial(n=1,p=0.5)
    if ((fake_defrosts == None) and defrosts):
        fake_defrosts = np.random.binomial(n=1,p=0.5)
    if random_shifts == None:
        random_shifts = np.random.binomial(n=1,p=0.5)

    simulated_run = add_anchored_drift(n)
    if chaotic_drift:
        simulated_run = simulated_run + add_chaotic_drift(n, scale=chaotic_drift_scale)
    if compressor:
        comp_cycle_height = max(np.random.gamma(1,2), 0.25)
    if gaussian_noise:
        random_noise_scale = abs(np.random.normal(loc=noise_scale, scale=noise_scale/2))
        if verbose: print('random_noise_scale is ', random_noise_scale)
        simulated_run = simulated_run + add_gaussian_noise(n, scale=random_noise_scale)
    if random_events:
        simulated_run = simulated_run + add_random_events(n)
    if random_shifts:
        simulated_run = simulated_run + add_random_shifts(n)
    
    height_baseline = 1.5 + np.random.gamma(1.5,3)
    if compressor: height_baseline = max(height_baseline, 2*comp_cycle_height)
    if gaussian_noise: height_baseline = max(height_baseline, 5*random_noise_scale)
    interval_baseline = 0
    if defrosts:
        d, labels, interval_baseline = add_defrosts(n, height_baseline=height_baseline, anomalous=anomalous, \
                                                    all_anomalous=all_anomalous, end_in_defrost=end_in_defrost)
        simulated_run = simulated_run + d
        if compressor:
            if np.random.binomial(n=1,p=0.5):
                simulated_run = simulated_run + add_compressor(n, height=comp_cycle_height)*np.invert(labels)
            else: simulated_run = simulated_run + add_compressor(n, height=comp_cycle_height)
    else:
        labels = np.array([0]*n)
        if compressor: simulated_run = simulated_run + add_compressor(n, height=comp_cycle_height)
    if fake_defrosts:
        fd = add_fake_defrosts(n=n, height_baseline=height_baseline)
        simulated_run = simulated_run + fd
    if end_in_fake_defrost:
        fd = add_fake_defrost_to_end(n=n, height_baseline=height_baseline)
        simulated_run = simulated_run + fd
    if verbose:
        print('chaotic_drift={}, compressor={}, gaussian_noise={}, fake_defrosts={},\
        random_shifts={}'.format(chaotic_drift, compressor, gaussian_noise, fake_defrosts, random_shifts))
    return simulated_run, labels, interval_baseline

def simulate_refrigerator_data(num_simulations, time_series_len, x0y0_ratio=0.5, x1y1_ratio=0.4, x1y0_ratio=0.1):
  """Generate an entire dataset of simulated refrigeration temperature time series.
  
  Parameters:
  num_simulations (int): The number of simulated time series in the dataset.
  time_series_len (int): The length of each simulated time series.
  x0y0_ratio (float): The portion of simulated time series that will not end in a defrost or a counterfeit defrost.
  x1y1_ratio (float): The portion of simulated time series that will end in a defrost.
  x1y0_ratio (float): The portion of simulated time series that will end in a counterfeit defrost.
  
  Returns:
  X (numpy array): An array of simulated refrigeration temperature time series.
  Y (numpy array): An array of defrost label time series.
  P (numpy array): An array of the baseline defrost interval values used to build each time series.
  """
  
  num_x0y0 = int(num_simulations*x0y0_ratio)
  num_x1y1 = int(num_simulations*x1y1_ratio)
  num_x1y0 = int(num_simulations*x1y0_ratio)

  X=np.zeros(((num_x0y0+num_x1y1+num_x1y0), time_series_len))
  Y=np.zeros(((num_x0y0+num_x1y1+num_x1y0), time_series_len))
  p=np.zeros((num_x0y0+num_x1y1+num_x1y0))
  
  for i in range(num_x0y0):
    X[i,:], Y[i,:], p[i] = generate_simulation(n=time_series_len, end_in_defrost=False, end_in_fake_defrost=False)
  for i in range(num_x1y1):
    ind = num_x0y0+i
    X[(ind),:],Y[(ind),:],p[ind] = generate_simulation(n=time_series_len, end_in_defrost=True, end_in_fake_defrost=False)
  for i in range(num_x1y0):
    ind = num_x0y0+num_x1y1+i
    X[(ind),:],Y[(ind),:],p[ind] = generate_simulation(n=time_series_len, end_in_defrost=False, end_in_fake_defrost=True)
  P = np.reshape(np.array(p), (len(p),1))
  Z = np.concatenate((X,Y,P), axis=1)
  np.random.shuffle(Z)
  X = Z[:,0:X.shape[1]]
  Y = Z[:,X.shape[1]:-1]
  P = Z[:,-1]
  del Z
  return (X,Y,P)

def predict_consecutive_readings(readings, look_back, model):
    """Classify consecutive time series values using the sliding window method.
    
    Parameters:
    readings (numpy array): The input time series.
    look_back(int): The size of the sliding window
    model (tensorflow.keras.Model): The model that will make the classifications.
    
    Returns:
    A numpy array of classification values.
    """
    
    input = []
    num_predictions = len(readings) - look_back
    test_predictions = [-1]*num_predictions
    for i in range(num_predictions):
        x = readings[i:i+look_back]
        x = (x-np.mean(x))/np.std(x)
        input.append(x)
    input = np.array(input)
    input = np.reshape(input, (input.shape[0],input.shape[1],1))
    return model.predict_classes(input)