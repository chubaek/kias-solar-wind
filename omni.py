import pandas as pd

file = 'OMNI2_H0_MRG1HR_563544.txt'

with open(file, 'r') as f:
    lines = f.readlines()
    lines_list = []
    for line in lines:
        if line.startswith('#'):
            continue
        else:
            lines_list.append(line)

data_dict = {
    'datetime': [],
    'Speed (km/s)': [],
    'Density (1/cm^3)': [],
    'Temperature (K)': [],
    'B (nT)': [],
    'Sunspot Number': [],
    # 'Bx_GSE (nT)': [],
    # 'By_GSE (nT)': [],
    # 'Bz_GSE (nT)': [],
    # 'By_GSM (nT)': [],
    # 'Bz_GSM (nT)': [],
    # 'Flow Pressure (nPa)': [],
    # 'Electric Field (mV/m)': [],
    # 'Plasma Beta': [],
    # 'Alfven Mach Number': [],
    # 'F10.7 Flux (sfu)': [],
    # 'Kp Index': [],
}

for idx, line in enumerate(lines_list):
    if idx > 1:
        data = line.split()
        dt = pd.to_datetime(data[0] + ' ' + data[1], format='%d-%m-%Y %H:%M:%S.%f')
        data_dict['datetime'].append(dt)
        data_dict['B (nT)'].append(data[2])
        # data_dict['Bx_GSE (nT)'].append(data[3])
        # data_dict['By_GSE (nT)'].append(data[4])
        # data_dict['Bz_GSE (nT)'].append(data[5])
        # data_dict['By_GSM (nT)'].append(data[6])
        # data_dict['Bz_GSM (nT)'].append(data[7])
        data_dict['Temperature (K)'].append(data[8])
        data_dict['Density (1/cm^3)'].append(data[9])
        data_dict['Speed (km/s)'].append(data[10])
        # data_dict['Flow Pressure (nPa)'].append(data[11])
        # data_dict['Electric Field (mV/m)'].append(data[12])
        # data_dict['Plasma Beta'].append(data[13])
        # data_dict['Alfven Mach Number'].append(data[14])
        data_dict['Sunspot Number'].append(data[15])
        # data_dict['F10.7 Flux (sfu)'].append(data[16])
        # data_dict['Kp Index'].append(float(data[17])/10.0)


data_df = pd.DataFrame(data_dict)
# Official OMNI2 fill values: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
fill_values = {
    'B (nT)': 999.9,
    # 'Bx_GSE (nT)': 999.9,
    # 'By_GSE (nT)': 999.9,
    # 'Bz_GSE (nT)': 999.9,
    # 'By_GSM (nT)': 999.9,
    # 'Bz_GSM (nT)': 999.9,
    'Temperature (K)': 10000000.0,
    'Density (1/cm^3)': 999.9,
    'Speed (km/s)': 9999.0,
    # 'Flow Pressure (nPa)': 99.99,
    # 'Electric Field (mV/m)': 999.99,
    # 'Plasma Beta': 999.99,
    # 'Alfven Mach Number': 999.9,
    'Sunspot Number': 999.0,
    # 'F10.7 Flux (sfu)': 999.9,
    # 'Kp Index': 99.0,
}
numeric_columns = data_df.columns.drop('datetime')
data_df[numeric_columns] = data_df[numeric_columns].apply(pd.to_numeric)
data_df.replace(fill_values, pd.NA, inplace=True)
data_df.to_csv('sw_data.csv', index=False)
