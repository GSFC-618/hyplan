# Project Overview

HyPlan is an open-source Python library containing tools for planning airborne remote sensing campaigns. 

---

## Repository Contents

### Core Modules

- **geometry.py**: Functions for geometric calculations essential to flight planning and sensor modeling.
- **glint.py**: Functions to predict solar glint angles based on sensor view angles and solar position.
- **ground_footprint.py**: Functions to calculate ground footprint of frame camera sensors based on flight altitude and field of view.
- **flight_line.py**: Functions to generate and modify flight lines.
- **flight_box.py**: Functions for generating multiple flight lines that cover a geographic area.
- **terrain.py**: Functions for downloading terrain DEM data and calculating where the sensor field of view intersects the ground.
- **sensors.py**: Defines sensor specifications and functionalities.
- **swath.py**: Functions to compute swath coverage based on sensor field of view, altitude, and terrain elevation.
- **units.py**: Functions for unit conversions and handling.
- **sun.py**: Functions to calculate solar position for mission planning.
- **airports.py**: Functions for locating and analyzing nearby airports for mission logistics.
- **download.py**: Functions for downloading necessary datasets or dependencies.

### Configuration and Setup

- **setup.py**: Script for installing the package.
- **pyproject.toml**: Build configuration file.
- **requirements.txt**: Lists Python dependencies for the project.
- **LICENSE.md**: Licensing details.

### Documentation

- **README.md**: Overview and instructions for the repository.

---

## Installation

To set up the environment, clone the repository and install the dependencies:

```bash
# Clone the repository
git clone <repository_url>
cd <repository_name>

# Install dependencies
pip uninstall -y hyplan; pip install -e .
```

---

## Usage

### Example: Planning a Flight Mission

Need to add material here

## Contributing

Contributions are welcome! If you have suggestions or find issues, please open an issue or submit a pull request.

---

## License

HyPlan is licensed under the Apache License, Version 2.0. See the `LICENSE.md` file for details.

---

## Contact

For inquiries or further information, please contact Ryan Pavlick (ryan.p.pavlick@nasa.gov).
