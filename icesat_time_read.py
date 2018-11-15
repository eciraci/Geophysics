
"""
# -- Written by Enrico Ciraci - 11/13/2018
# -- 
# -- ICESat data are distributed employing the UTC (J2000) as time format
# -- UTC (J2000) => seconds since 2000-01-01 12:00:00 
# --
# -- This scripts uses the python.datetime library to convert the
# -- UTC (J2000) time values into digital time format [year]
# -- 229812558.824506              2007-04-14 08:49:18           2007.2794520547945
"""
# - python dependencies
from __future__ import print_function
import os
import numpy as np
import h5py
import datetime
import calendar


def read_h5_icesat(path_to_file):
	"""
	"""
	with h5py.File(path_to_file, 'r') as cfid:
		# - Transmit time of each shot in J2000 seconds
		utc_j2000_time = np.array(cfid['Data_40HZ']['Time']['d_UTCTime_40'][:])
		cfid.close()
	# -
	return{'utc_j2000_time': utc_j2000_time}


def write_txt(tseries, dtime, gtime, path_to_file):
    """
    Save time series in the selected output txt file (col1=gtime  col2=dtime  col3=mass monthly anomaly)
    :param tseries: mass change time series
    :param dtime: time expressed in digital format
    :param gtime: time expressed in GRACE time format
    :param path_to_file: absolute path to the output directory
    :return: None
    """
    with open(path_to_file, 'w') as wfid:
        wfid.write('G-Month     D-Month    Monthly Mass Anomaly  [Gt]\n')
        for tt in range(0, len(dtime)):
            wfid.write('%7i %6f %21.12e' % (gtime[tt], dtime[tt], tseries[tt])+"\n")


def main():
	# - path to input file
	input_path = os.path.join('.', 'GLAH14_634_2119_002_0421_0_01_0001.H5')
	# - import ICESat data
	utc_j2000_time = np.array(read_h5_icesat(input_path)['utc_j2000_time'])
	
	# - Use python datetime to convert the time values  from UTC (J2000)
	# - to [year] or digital-time format.

	# - set t00 equal to the time reference employed UTC (J2000) 
	t00 = datetime.datetime(2000, 1, 1, hour=12, minute=0)  
	digital_time = np.zeros(len(utc_j2000_time))

	# - save the obtained output inside a txt file
	path_to_file = os.path.join('.', 'time_conversion.txt')
	with open(path_to_file, 'w') as wfid:
		wfid.write('UTC (J2000)'.ljust(30) + 'Date'.ljust(30) + 'Digital Time [Year]\n')
	
		for t in range(len(digital_time)):
			# - onvert from UTC J2000 to python datetime object
			time_temp = t00 + datetime.timedelta(seconds=int(utc_j2000_time[t])) 
			# - temporal interval between the considered shot and the beginnig 
			# - of the year.
			t_yr = datetime.datetime(time_temp.year, 1, 1, hour=12, minute=0) 
			time_diff = time_temp - t_yr
			if calendar.isleap(time_temp.year):
				n_days = 366.
			else:
				n_days = 365.
			# - calculate digital time value
			# - NOTE: this step introduces a small rounding error in the conversion,
			# - time_diff.days returns in fact an integer numberand esidual hours and 
			# - seconds are therefore lost. At the same time, this approximation is
			# - neglectable when calculating the dhdt values.
			digital_time[t] = time_temp.year + (time_diff.days/n_days)
			# -
			wfid.write(np.str(utc_j2000_time[t]).ljust(30) + np.str(time_temp).ljust(30) + np.str(digital_time[t])+"\n")

if __name__ == '__main__':
    main()
