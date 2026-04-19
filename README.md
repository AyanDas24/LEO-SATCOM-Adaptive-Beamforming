# LEO SATCOM Adaptive Spot Beam System

Iridium-style adaptive beamforming system for satellite communications
over India & South East Asia.

## Results
- 92% CNN beam selection accuracy
- +19.3 dB SINR improvement over no beamforming
- 48 spot beams covering India, Bangladesh, SEA corridor

## Architecture
- 12×10 URA (120 elements), L-band 1621 MHz, 780 km LEO
- CNN classifier (Deep Learning Toolbox) replaces static LUT
- MVDR adaptive null steering for interference rejection
- RF Toolbox link budget per beam

## Toolboxes
MATLAB | Phased Array System Toolbox | RF Toolbox | Deep Learning Toolbox

## Reference
Based on: Iridium MMA architecture + Medina-Sanchez (2014) UMass PhD
