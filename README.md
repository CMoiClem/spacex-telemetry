# SpaceX Telemetry Extractor

Automated telemetry data extraction from SpaceX Starship launch webcasts using OCR.

## Overview

This tool extracts real-time telemetry data from SpaceX launch webcast frames including:
- Stage 1 Speed (km/h)
- Stage 1 Altitude (km)
- Stage 2 Speed (km/h)
- Stage 2 Altitude (km)
- Timestamp (HH:MM:SS)

## Requirements

- Python 3.8+
- Tesseract OCR
- Dependencies from requirements.txt:
numpy opencv-python pytesseract pyyaml natsort


## Configuration

The `default_config.yaml` contains key parameters that can be manually modified if needed.

## Features

Image Processing
Grayscale conversion
Binary thresholding
Morphological operations for noise reduction
OCR Validation
Confidence threshold filtering
Sequence validation for altitude readings
Natural descent handling (e.g., 10km → 9km transitions)
Speed range and deviation checks
Detailed logging for debugging
Data Output
CSV format with timestamps
Handles missing/invalid readings
Maintains data consistency

## Usage

Image frames
The program needs individual frames of the webcast. 
Currently using https://github.com/CMoiClem/video-frame-extractor for that, but I will integrate them all together later. 



Arguments
--frames-dir: Directory containing video frame images
--output: Path for output CSV file
--config: Path to configuration YAML

Validation Logic
Altitude:
Rejects single digits after double digits unless natural descent
Allows small decreases during descent (configurable)
Strict increases validation (max 2 units)

Speed:
Range validation (0-40000 km/h)
Statistical validation using rolling window
Deviation checks from mean

Detailed logging of:
OCR readings and confidence
Validation decisions
Processing errors
Output in telemetry_extractor.log

## Known Limitations
OCR accuracy depends on frame brightness
Some readings may be missed during rapid changes
Requires specific SpaceX overlay format

## Future Improvements
Support for different overlay formats (Falcon 9)
Individual frames extraction from a webcast
Easy way to download webcasts from X
GUI development
Automating graph creation from the telemetry CSV

## Contributing
Feel free to open issues or submit pull requests for improvements.

