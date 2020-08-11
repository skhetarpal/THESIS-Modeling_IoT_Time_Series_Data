import numpy as np


def generate_binary_run(len_run, noise=False, period=None, mean_period=None, period_range=None, start=None,\
                        force_end_in_1=False, force_end_in_0=False, force_end_in_fake_1=False):
  """Generate a binary time series with a seasonal component for a toy dataset.
  
  If 'period' is 'None', then the seasonal component's period will be set to 'period'.
  If 'period' is 'None' and 'mean_period' is not 'None', then the seasonal component period will be sampled 
  from a Gaussian distribution centered at 'mean_period'.  
  If 'period' is 'None', 'mean_period' is 'None', and 'period_range' is not 'None', then the seasonal 
  component period will be sampled from a uniform distribution with a min of 'period_range[0]' and a max of
  'period_range[1]'.
  If 'period', 'mean_period', and 'period_range' are all 'None', then the seasonal component period will be 
  sampled from a Gaussian distribution centered at 'len_run'/10.
  
  'force_end_in_1', 'force_end_in_0', and 'force_end_in_fake_1' are used to control the characteristics of
  the last time point in the time series.  See parameter descriptions below.
  Between 'force_end_in_1', 'force_end_in_0', and 'force_end_in_fake_1', only one can be set to True.
  If all three parameters are set to 'False', then the placement of the seasonal component will be random.
  
  Parameters:
  len_run (positive int): The desired length of the time series.
  noise (bool): Whether to add noide to the time series.
  period (positive int): The desired period of the seasonal component.
  mean_period (int): Center of Guassian distribution from which the period is sampled.
  period_range (tuple or list): Min and max values of a uniform distribution from which the period is sampled.
  start (int): The time point at which the first 1 of the seasonal component occurs.
  force_end_in_1 (bool): If True, force the last of the seasonally occurring 1s to fall on the last time point.
  force_end_in_0 (bool): If True, prevent the last of the seasonally occurring 1s from falling on the last time point.
  force_end_in_fake_1 (bool): If True, prevent the last of the seasonally occurring 1s from falling on the last time 
                              point, but then set the last time to the value 1 anyways.

  Returns:
  X (numpy array): Time series values
  Y (numpy array): Seasonal component values
  period (int): Time series period
  """
  if start and (force_end_in_1 or force_end_in_0):
    raise ValueError('Cannot assign a start value if either force_end_in_1 or force_end_in_0 are True.')
  if (force_end_in_1 and force_end_in_0):
    raise ValueError('force_end_in_1 and force_end_in_0 cannot both be True.')
  if (force_end_in_fake_1 and force_end_in_0) or (force_end_in_fake_1 and force_end_in_1):
    raise ValueError('force_end_in_fake_1 and force_end_in_0 or force_end_in_1 cannot both be True.')
  if (force_end_in_fake_1 and start==0):
    raise ValueError('force_end_in_fake_1 and start=0 cannot both be True.')
  X = np.zeros(len_run)
  Y = np.zeros(len_run).astype(int)
  if period == None:
    if mean_period:
      period = int(max(6,(np.random.normal(loc=mean_period, scale=mean_period/3))))
    elif period_range:
      period = int(np.random.uniform(period_range[0], period_range[1]))
    else:
      period = int(max(6,(np.random.normal(loc=int(len_run/10), scale=int(len_run/10/3)))))
  #If start==None, assign a value to start
  if force_end_in_1:
    start = 0
  elif (force_end_in_0 or force_end_in_fake_1) and (start == None):
    start = int(np.random.uniform(1,period))
  elif start == None:
    start = int(np.random.uniform(0,period)) 

  if force_end_in_fake_1:
    noise=True
    X[0] = 1
  
  if noise:
    for i in range(len_run):
      if np.random.binomial(n=1,p=(1/period)):
        X[i]=1

  current = start
  while current < len_run:
    X[current] = 1
    Y[current] = 1
    current += period
  
  if (force_end_in_1 or force_end_in_0 or force_end_in_fake_1):
    X = np.flip(X)
    Y = np.flip(Y)
  
  return X,Y,period


def generate_binary_training_data(len_run, noise=True, period=None, mean_period=None, period_range=None, start=None, num_x0y0=0, num_x1y1=0, num_x1y0=0, num_unspecified_end=0):
  """Generate a toy dataset containing binary time series with seasonal components.
  
  If 'period' is 'None', then all seasonal component periods will be set to 'period'.
  If 'period' is 'None' and 'mean_period' is not 'None', then the seasonal component periods will be sampled 
  from a Gaussian distribution centered at 'mean_period'.  
  If 'period' is 'None', 'mean_period' is 'None', and 'period_range' is not 'None', then the seasonal 
  component periods will be sampled from a uniform distribution with a min of 'period_range[0]' and a max of
  'period_range[1]'.
  If 'period', 'mean_period', and 'period_range' are all 'None', then the seasonal component periods will be 
  sampled from a Gaussian distribution centered at 'len_run'/10.
  
  Parameters:
  len_run (positive int): The desired length of each time series.
  noise (bool): Whether to add noide to each time series.
  period (positive int): The desired period of every seasonal component.
  mean_period (int): Center of Guassian distribution from which the periods are sampled.
  period_range (tuple or list): Min and max values of a uniform distribution from which the periods are sampled.
  start (int): The time point at which the first 1 of the seasonal components occurs within each time series.
  num_x0y0 (int): The number of time series whose last seasonally occurring 1 falls on the last time point.
  num_x1y1 (int): The number of time series whose last seasonally occurring 1 doesn't fall on the last time point.
  num_x1y0 (int): The number of time series whose last seasonally occurring 1 doesn't fall on the last time point,
                  but whose last time is guaranteed to be set to the value 1 anyways.
  num_unspecified_end (int): The number of time series whose seasonal components can fall anywhere.

  Returns:
  X (numpy array): 2D array of time series values
  Y (numpy array): 2D array of seasonal component values
  P (numpy array): 1D array of time series periods
  """
  
  X=np.zeros(((num_x0y0+num_x1y1+num_x1y0+num_unspecified_end), len_run))
  Y=np.zeros(((num_x0y0+num_x1y1+num_x1y0+num_unspecified_end), len_run))
  p=np.zeros((num_x0y0+num_x1y1+num_x1y0+num_unspecified_end))
  for i in range(num_x0y0):
    X[i,:], Y[i,:], p[i] = generate_binary_run(len_run=len_run, noise=noise, period=period, mean_period=mean_period, period_range=None, start=start, force_end_in_1=False, force_end_in_0=True, force_end_in_fake_1=False)
  for i in range(num_x1y1):
    ind = num_x0y0+i
    X[(ind),:],Y[(ind),:],p[ind] = generate_binary_run(len_run=len_run, noise=noise, period=period, mean_period=mean_period, period_range=None, start=start, force_end_in_1=True, force_end_in_0=False, force_end_in_fake_1=False)
  for i in range(num_x1y0):
    ind = num_x0y0+num_x1y1+i
    X[(ind),:],Y[(ind),:],p[ind] = generate_binary_run(len_run=len_run, noise=noise, period=period, mean_period=mean_period, period_range=None, start=start, force_end_in_1=False, force_end_in_0=False, force_end_in_fake_1=True)
  for i in range(num_unspecified_end):
    ind = num_x0y0+num_x1y1+num_x1y0+i
    X[(ind),:],Y[(ind),:],p[ind] = generate_binary_run(len_run=len_run, noise=noise, period=period, mean_period=mean_period, period_range=None, start=start, force_end_in_1=False, force_end_in_0=False, force_end_in_fake_1=False)
  P = np.reshape(np.array(p), (len(p),1))
  Z = np.concatenate((X,Y,P), axis=1)
  np.random.shuffle(Z)
  X = Z[:,0:X.shape[1]]
  Y = Z[:,X.shape[1]:-1]
  P = Z[:,-1]
  del Z
  return (X,Y,P)
  
def predict_consecutive_binary_readings(readings, look_back, model):
    """Generate an array of forecasted values using the sliding window method
    
    Parameters:
    readings (array): The input time series
    look_back(int): The size of the sliding window
    model (tensorflow.keras.Model): The model that will make forecasts
    """
    
    input = []
    for i in range(len(readings)-look_back):
        x = readings[i:i+look_back]
        input.append(x)
    input = np.array(input)
    input = np.reshape(input, (input.shape[0],input.shape[1],1))
    return model.predict_classes(input)