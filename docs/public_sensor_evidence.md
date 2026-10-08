# Public evidence and simulation boundary

## Publicly supported facts

1. CVRDE's 2016 paper *Health Monitoring for Armoured Fighting Vehicles* describes a diagnostic health-monitoring system for an AFV powerpack and says the monitored parameters include:
   - RTD temperature sensing for **engine coolant temperature** and **transmission oil temperature**.
   - Pressure sensing for **engine oil pressure** and **clutch pressure**.
   - Current sensing for **solenoids**, including over-current/short-circuit/ON-OFF monitoring.
   - Magnetic pickups for **engine and transmission RPM**; a phase-shifted pair is used to determine vehicle direction, and vehicle speed can be derived from the speed data.
   Source: https://pdfcoffee.com/innovative-design-and-development-practices-in-aerospace-and-automotive-engineering-i-dad-february-22-24-2016-pdf-pdf-free.html

2. Public DRDO/BEML material describes an Arjun powerpack anchor around **1400 HP at 2400 rpm** (the exact variant/source wording differs across public documents). We use this only as a simulation anchor, not as a calibration range.
   Source: https://indembassysuriname.gov.in/public_files/assets/pdf/BEML_EOI_for_appointment_of_Representative.pdf

3. DRDO's public product material for the Arjun family also documents a 1030 kW / 2400 rpm engine specification for a main battle tank configuration and cross-country speed around 40 km/h. The public material should be treated as variant-specific.
   Source: https://www.drdo.gov.in/drdo/sites/default/files/schemes_services/CompendiumProductforExport2025.pdf

4. Public reporting on the later indigenous DATRAN-1500 program describes testing across extreme conditions including high altitude and hot/cold environments. These profiles are used in this repository only as **stress-test environments**, not as claims about the exact certified operating envelope of every Arjun variant.
   Source: https://www.drdo.gov.in/drdo/sites/default/files/drdo-news-documents/NPC14Feb2024.pdf

5. The public NASA C-MAPSS dataset is a separate simulated turbofan-engine benchmark. It contains run-to-failure training trajectories and truncated test trajectories and should **not** be represented as Arjun data.
   Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

## What is simulated, not sourced as official Arjun telemetry

Public sources do not provide a complete Arjun telemetry corpus with sensor calibration curves, exact measurement distributions, run-to-failure labels, failure-mode labels, and sampling protocol. Therefore this repository does not claim that the generated values are official Arjun measurements.

The simulator creates physically coherent telemetry using the public channel list and platform anchors plus explicit engineering assumptions for:
- operating-mode persistence;
- thermal response to load and ambient temperature;
- lubrication-pressure behavior;
- clutch-pressure response;
- solenoid current behavior;
- wear/degradation effects;
- sensor noise, missing blocks and transient spikes;
- environmental stress profiles.

The hidden health/degradation state exists **only inside the data generator**. It is never written as a model feature. The only supervised target exported to the training table is the actual remaining time to the simulated run-to-failure point.
