# GNSS Data Processing Script

This Python script processes GNSS (Global Navigation Satellite System) data from Android devices, extracting information about satellite positions and receiver positions. It includes functions to read GNSS data from a file, parse the data, calculate satellite positions, estimate receiver positions, and generate output files including CSV files and a KML file.

## Dependencies

- `Python 3.x`
- `pandas`
- `numpy`
- `matplotlib`
- `navpy`
- `gnssutils`
- `simplekml`

* `git clone https://github.com/johnsonmitchelld/gnss-analysis.git`
* `mv gnss-analysis/gnssutils gnssutils`
* `sudo rm -r gnss-analysis`
* `pip install -r requirements.txt`

Make sure to install these dependencies using `pip` before running the script.

## Usage

To use this script, follow these steps:

1. Install the required dependencies mentioned above.
2. Place your GNSS data file (e.g., `walking.txt`) in the same directory as the script.
3. Modify the script's `main` function to specify the input file path.
4. Run the script by executing `python script.py`.

The script will process the GNSS data, calculate satellite and receiver positions, and generate output files including CSV files containing satellite and receiver positions, as well as a KML file for visualization.

## Functionality

- **read_file:** Reads GNSS data from the specified file path and parses Android fixes and GNSS measurements.
- **calculate_satellite_position:** Calculates satellite positions based on ephemeris data, timestamp, and GNSS measurements for one epoch.
- **generate_satellite_positions:** Generates satellite positions using GNSS measurements for each epoch.
- **least_squares:** Performs least squares estimation to calculate receiver position and clock bias.
- **calculate_pos_x_y_z:** Calculates receiver position in ECEF (Earth-Centered, Earth-Fixed) coordinates using GNSS measurements.
- **calculate_lla_based_on_pos_x_y_z:** Calculates receiver latitude, longitude, and altitude from ECEF coordinates.
- **generate_kml_from_lla:** Generates a KML file from latitude, longitude, and altitude data.

## Output Files

The script generates the following output files:
- `{filename}-sv_positions.csv`: CSV file containing satellite positions.
- `{filename}-calculated_positions.csv`: CSV file containing calculated receiver positions.
- `{filename}-android_positions.csv`: CSV file containing Android device fixes.
- `{filename}-KML.kml`: KML file for visualization of receiver positions.