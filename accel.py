import re
import pprint
import string
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import linregress
#%matplotlib notebook

pList = []
dList = []
tList = []

def importAccelData(fileName, fileHeaderLength=10, labelLength=2):
    fileHeader = []
    dataLabel = []
    accelData = {}
    with open(fileName) as f:  # Extract header
        for lineNumber, line in enumerate(f):
            if lineNumber <= fileHeaderLength - labelLength:
                fileHeader.append(re.sub(
                    r"[\n\t]*", "",
                    line))  # RegEx to remove new-lines and tabs from header
            elif lineNumber > fileHeaderLength - labelLength and lineNumber < (
                    fileHeaderLength + labelLength) - 1:
                labelString = re.sub(r"[\n]*", "", line)
                dataLabel.append(string.split(labelString, sep='\t'))
            else:
                break

    accelData['FileName'] = fileHeader[0]
    accelData['TimeBase'] = fileHeader[1]
    accelData['NumberOfChannels'] = fileHeader[2]

    accelData['Ch1Label'] = fileHeader[3]
    accelData['Ch1Unit'] = fileHeader[4]
    accelData['Ch1Stop'] = dataLabel[1][1]

    accelData['Ch2Label'] = fileHeader[5]
    accelData['Ch2Unit'] = fileHeader[6]
    accelData['Ch2Stop'] = dataLabel[1][2]

    accelData['Ch3Label'] = fileHeader[7]
    accelData['Ch3Unit'] = fileHeader[8]
    accelData['Ch2Stop'] = dataLabel[1][3]

    accelData['Data'] = pd.read_csv(
        fileName,
        skiprows=(fileHeaderLength + labelLength),
        header=None,
        names=dataLabel[0],
        sep='\t')

    if dataLabel[0][0] == 'min': # Why the the accelerometer software change its export units from seconds to minutes? What is the point!
        accelData['Data'].rename(columns={'min':'sec'}, inplace=True)
        accelData['Data']['sec'] = accelData['Data']['sec'] * 60
        #df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'}, inplace=True)

    #pprint.pprint(accelData)

    return accelData

def plotAccel(data):

    x = data['Data']['sec']
    y1 = data['Data']['CH1']
    y2 = data['Data']['CH2']
    y3 = data['Data']['CH3']

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    #Convert to msec
    #xm = x * 2
    xm = x

    # Signal to plot
    axs[0].plot(xm, y1)
    axs[1].plot(xm, y2)
    axs[2].plot(xm, y3)

    for chart in range(3):
        axs[chart].grid(True)
        axs[chart].set_xlabel('time (s)')
        axs[chart].margins(x=0)

    axs[2].set_ylabel(data['Ch3Unit'])
    axs[2].set_title(data['Ch3Label'])

    axs[1].set_ylabel(data['Ch2Unit'])
    axs[1].set_title(data['Ch2Label'])

    axs[0].set_ylabel(data['Ch1Unit'])
    axs[0].set_title(data['Ch1Label'])

    fig.suptitle(data['FileName'])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def selectAccelData(data, lowerLimit, upperLimit):

    x = data['Data']['sec']
    y1 = data['Data']['CH1']
    y2 = data['Data']['CH2']
    y3 = data['Data']['CH3']

    # Plotting data selection
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    #Convert to msec
    #xm = x * 2
    xm = x

    # Signal to plot
    axs[0].plot(xm, y1)
    axs[1].plot(xm, y2)
    axs[2].plot(xm, y3)

    for chart in range(3):
        axs[chart].grid(True)
        axs[chart].set_xlabel('time (s)')
        axs[chart].margins(x=0)
        axs[chart].axvspan(lowerLimit, upperLimit, color='red', alpha=0.5) # Overlay selected area

    axs[2].set_ylabel(data['Ch3Unit'])
    axs[2].set_title(data['Ch3Label']  + ' Selected Range')

    axs[1].set_ylabel(data['Ch2Unit'])
    axs[1].set_title(data['Ch2Label']  + ' Selected Range')

    axs[0].set_ylabel(data['Ch1Unit'])
    axs[0].set_title(data['Ch1Label'] + ' Selected Range')

    fig.suptitle(data['FileName'] + ' Selected Range')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # New selected data, construct dictionary
    selectedData = data
    newData = {'CH1': data['Data']['CH1'].loc[lowerLimit:upperLimit].values,
               'CH2': data['Data']['CH2'].loc[lowerLimit:upperLimit].values,
               'CH3': data['Data']['CH3'].loc[lowerLimit:upperLimit].values,
               'sec': data['Data']['sec'].loc[lowerLimit:upperLimit].values}

    # make dataframe out of new np arrays
    newFrame = pd.DataFrame(newData)
    selectedData['Data'] = newFrame

    # Ammend filename to reflect data selection and add additional info keys to data dictionary
    selectedData['FileName'] = data['FileName'] + ' Selected from ' + str(lowerLimit) + 'ms to ' + str(upperLimit) + 'ms'
    selectedData['Ch1Label'] = data['Ch1Label'] + ' Selected from ' + str(lowerLimit) + 'ms to ' + str(upperLimit) + 'ms'
    selectedData['Ch2Label'] = data['Ch2Label'] + ' Selected from ' + str(lowerLimit) + 'ms to ' + str(upperLimit) + 'ms'
    selectedData['Ch3Label'] = data['Ch3Label'] + ' Selected from ' + str(lowerLimit) + 'ms to ' + str(upperLimit) + 'ms'
    selectedData['LowerLimit'] = lowerLimit
    selectedData['UpperLimt'] = upperLimit
    return selectedData

def spectrals(data):

    s1 = data['Data']['CH1'].values
    s2 = data['Data']['CH2'].values
    s3 = data['Data']['CH3'].values

    s1Avg = s1 - np.mean(s1)
    s2Avg = s2 - np.mean(s2)
    s3Avg = s3 - np.mean(s3)

    fig, axs = plt.subplots(3,1,figsize=(10, 10))

    sp1 = np.fft.fft(s1Avg)
    sp2 = np.fft.fft(s2Avg)
    sp3 = np.fft.fft(s3Avg)
    for chart in range(3):
        axs[chart].grid(True)
        axs[chart].set_xlabel('Frequency (Hz)')
        axs[chart].margins(x=0)

    axs[2].set_ylabel(r'$|F|$')
    axs[2].set_title(data['Ch3Label']  + ' Magnitude Spectra')

    axs[1].set_ylabel(r'$|F|$')
    axs[1].set_title(data['Ch2Label']  + ' Magnitude Spectra')

    axs[0].set_ylabel(r'$|F|$')
    axs[0].set_title(data['Ch1Label'] + ' Magnitude Spectra')

    fig.suptitle(data['FileName'] + ' Magnitude Spectra')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    axs[0].stem(np.real(sp1))
    axs[1].stem(np.real(sp2))
    axs[2].stem(np.real(sp3))

    # Deal with time base of data structure by keeping it the same length but changning each entry to NaN
    #timeEmpty = np.empty(data['Data']['sec'].size)
    #timeEmpty[:] = np.nan

    spectralData = data
    spectrals = {'CH1': sp1,
                 'CH2': sp2,
                 'CH3': sp2,
                 #'sec': timeEmpty
                  'sec': data['Data']['sec'].values}

    spectralFrame = pd.DataFrame(spectrals)
    spectralData['Data'] = spectralFrame

    # Ammend filename to reflect data selection and add additional info keys to data dictionary
    spectralData['FileName'] = data['FileName'] + ' Magnitude Spectra'

    return spectralData

def rmsAccel(x,y1,y2,y3):

    y1_m = y1-np.mean(y1)
    y2_m = y1-np.mean(y2)
    y3_m = y1-np.mean(y3)

    y1_rms = np.sqrt((np.sum(y1_m**2)/(x.size)))
    y2_rms = np.sqrt((np.sum(y2_m**2)/(x.size)))
    y3_rms = np.sqrt((np.sum(y3_m**2)/(x.size)))

    return[[y1_rms,0],[y2_rms,0,],[y3_rms,0]]

def sonicationRMS(data):

    x = data['Data']['sec']
    y1 = data['Data']['CH1']
    y2 = data['Data']['CH2']
    y3 = data['Data']['CH3']

    y1_m = y1-np.mean(y1)
    y2_m = y1-np.mean(y2)
    y3_m = y1-np.mean(y3)

    y1_rms = np.sqrt((np.sum(y1_m**2)/(x.size)))
    y2_rms = np.sqrt((np.sum(y2_m**2)/(x.size)))
    y3_rms = np.sqrt((np.sum(y3_m**2)/(x.size)))

    mag = np.sqrt((y1_rms)**2 + (y2_rms)**2 + (y3_rms)**2)

    return mag

def plotRMS(data, test, n, title='RMS Acceleration per', xLabel='Trial'):
    RMS = []
    RMS_si = []
    # Calculate RMS values
    for i in data:
        RMS.append(sonicationRMS(i))
    for entry in RMS:
        RMS_si.append(entry * 9.81)
    s = np.std(RMS)
    x_si = np.arange(0,n,1.0)
    slope_si, intercept_si, r_value_si, p_value_si, std_err_si = linregress(x_si,RMS_si)

    mean_si = sum(RMS_si)/len(RMS_si)
    s_si = np.std(RMS_si)
    label = None
    # g-Force_RMS Plot
    fig_si, axs_si = plt.subplots(figsize=(12,7))

    if test == 1:
        label = 'Distal'
    if test == 2:
        label = 'Proximal'
    if test == 3:
        label = 'Target'

    axs_si.set_title(str(title) + str(xLabel), fontsize=23)
    axs_si.set_ylabel(r'Acceleration ($m/s^2$ RMS)', fontsize=20)
    axs_si.set_xlabel(str(xLabel), fontsize=20)
    axs_si.grid(True)

    # 'y = ' + str(np.round(slope,3)) + 'x +' + str(np.round(intercept,3))

    fig_si.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.xticks(np.arange(0, 21, 1.0), fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(np.arange(0,n,1.0), RMS_si, 'bo-', markersize=12, linewidth=3, label= 'sd = ' + str(np.round(s,3)))
    plt.plot(x_si, intercept_si + slope_si*x_si, 'r--', label= r'$r^2$ = ' + str(np.round(r_value_si,3)) + '\n' + 'SEM = ' + str(np.round(std_err_si,3)))
    #plt.errorbar(x, intercept + slope*x, xerr=0, yerr=2.58*std_err)
    handles_si, labels_si = axs_si.get_legend_handles_labels()
    axs_si.legend(handles_si, labels_si, fontsize=15)
    plt.show()
    #plt.savefig('final.png')

def importAndPlot(fileName, test, a, b):
    global pList
    global dList
    global tList

    data = importAccelData(str(fileName))
    if test == 1:
        dList.append(selectAccelData(data, a, b))
        n = len(dList)
        plotRMS(dList,test,n)
    if test == 2:
        pList.append(selectAccelData(data, a, b))
        n = len(pList)
        plotRMS(pList,test,n)
    if test == 3:
        tList.append(selectAccelData(data, a, b))
        n = len(tList)
        plotRMS(tList,test,n)
