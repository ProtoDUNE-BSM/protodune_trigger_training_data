# ProtoDUNE Trigger Training Data Generator

Python module to generate and preprocess ProtoDUNE trigger training data, including:

- converting ART ROOT-based training data into HDF5,
- binning ProtoDUNE training data

## Installation Requirements

Before installing, make sure you have: 
- Python ≥ 3.8
- A Python environment (e.g., virtualenv, conda)
- ART / root dependencies if you will be reading ROOT files
- Standard Python packages (numpy, h5py, etc.) — see requirements.txt

## Install From Source

Clone the repository:
```
git clone https://github.com/ProtoDUNE-BSM/protodune_trigger_training_data.git
cd protodune_trigger_training_data
```

After installing the source code, follow these steps:

### 1. Create the virtual environment
From the repository root:
```
python3 -m venv venv
```
This creates a virtual environment in a directory called venv/.

### 2. Activate the virtual environment
```
source venv/bin/activate
```

### 3. Installing Dependencies
After creating and activating your venv, you must install the required packages. Use the requirements.txt file:
```
pip install -r requirements.txt
```
This installs all Python dependencies needed by the module.

### 3. Install the package into your environment:
```
pip install .
```

If you are actively developing or editing the code, install in editable mode:
```
pip install -e .
```
This will make the package available without re-installation after every change.

## Using the Package

Once installed, the package is available under the generate_protodune_trigger_training_data namespace.

### Importing

In your Python scripts or notebooks 
```
import generate_protodune_trigger_training_data as gptd
```
The package exposes three main submodules:
```
convert_artroot_training_data_to_hdf5 — convert ROOT files to HDF5
bin_protodune_training_data — bin ProtoDUNE training data
bin_combined_plane_protodune_training_data — bin combined plane training data
```
You can import them like:
```
from generate_protodune_trigger_training_data import convert_artroot_training_data_to_hdf5
from generate_protodune_trigger_training_data import bin_protodune_training_data
from generate_protodune_trigger_training_data import bin_combined_plane_protodune_training_data
```

## Running the Training Data Scripts from the Terminal

If you have installed the package, you can run
```
python -m generate_protodune_trigger_training_data.convert_artroot_training_data_to_hdf5 \
  -i $INARTROOT -o $OUTHDF5NU -n
```
the `-n` option is for converting neutrino data to hdf5. Do cosmic data just remove the `-n` option.

The above loads a single art::ROOT file. However, often you would want to load a list of art::ROOT files. First create a '.txt' list using a bash script like below:

```
#!/bin/bash

COSMICTXTOUTFILENAME="pdhd_artroot_bsmw_trigger_training_cosmic_data_files.txt"                                         
WORKFLOW=10969

REGPATH="/pnfs/dune/scratch/users/chasnip/ProtoDUNEBSM/PDHDANA/fnal/${WORKFLOW}/1/001"
XROOTDPATH="root://fndca1.fnal.gov:1094//pnfs/fnal.gov/usr/dune/scratch/users/chasnip/ProtoDUNEBSM/PDHDANA/fnal/${WORKFLOW}/1/001"

for file in ${REGPATH}/*.root; do
    filename=$(basename "$file")
    echo "File name: $filename"
    XROOTDFILEPATH=${XROOTDPATH}/${filename}
    echo " xRootD path = ${XROOTDFILEPATH}"

    echo "${XROOTDFILEPATH}" >> ./${COSMICTXTOUTFILENAME}
done
```

Once you have the '.txt' list you can then run:
```
python -m generate_protodune_trigger_training_data.convert_artroot_training_data_to_hdf5 \
    -I $INARTROOTLIST -o $OUTHDF5NU -n
```

Finally you can bin the data and output the binned data to a '.npz' file:
```
python -m generate_protodune_trigger_training_data.bin_combined_plane_protodune_training_data.py \
    -i $OUTHDF5NU -o $OUTBINNEDNU -d "np04" --noapa1 -t 10 -c 10
```
This function will bin the hdf5 data for the NP04 detector, ignorinf APA 1 and with 10 time and channel bins.