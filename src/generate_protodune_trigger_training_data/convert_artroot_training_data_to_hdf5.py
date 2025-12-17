#!/usr/bin/env python3

import argparse
import uproot
import awkward as ak
import numpy as np
import pandas as pd
import h5py

# code to convert art::root file to hdf5  

def print_hdf5_structure(filename):
  with h5py.File(filename, "r") as f:
    def print_name(name):
      print(name)
    f.visit(print_name)

# Read an input list of art::ROOT files with uproot and output
# a hdf5 file.

def uproot_to_hdf5(filelist, output_hdf5, neutrino_only=False):
 
  branches = [
      "Window_apacrp",
      "Window_planeid",
      "Window_timepeak",
      "Window_channelid",
      "Window_adcintegral",
      "Window_tot",
      "Window_adcpeak",
      ]

  # --- Read ROOT file ---
  for filename in filelist:
    with uproot.open(filename) as f:
      print(f'Opening {filename}')
      if not neutrino_only:
        tree = f["GeneralProtoDUNETriggerTrainingDataMaker/TPWindowTree"]
      else:
        tree = f["GeneralProtoDUNETriggerTrainingDataMaker/TPNuWindowTree"]
      arrays = tree.arrays(branches, library="np")

    # --- Write HDF5 ---
    with h5py.File(output_hdf5, "a") as h5f:

      if "events" not in h5f:
        events_group = h5f.create_group("events")
      else:
        events_group = h5f["events"]

      existing = list(events_group.keys())
      if existing:
        next_event = max(int(e.split("_")[1]) for e in existing) + 1
      else:
        next_event = 0

      num_events = len(arrays[branches[0]])
      for i in range(num_events):

        event_group = events_group.create_group(f"event_{next_event + i}")
      
        for br in branches:
          subevents = arrays[br][i]
          br_group = event_group.create_group(br)
          for j, subvec in enumerate(subevents):
            arr = np.asarray(subvec, dtype=np.int32)
            br_group.create_dataset(f"subevent_{j}", data=arr)


def parse_args():
  parser = argparse.ArgumentParser(description="Script to convert a art::ROOT file to a hdf5 file.")
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("-i", "--inputfile", help="Single input art::ROOT file")
  group.add_argument("-I", "--inputlist", help="Text file containing a list of input files (one per line)")

  parser.add_argument("-o", "--outputfile", required=True, help="Output hdf5 file")
  parser.add_argument("-n", "--neutrinowindows", action="store_true", help="Only make the neutrino windows hdf5 file (default: False)")
  return parser.parse_args()

def main():
  args = parse_args()

  if args.inputfile:
    filelist = [args.inputfile]

  elif args.inputlist:
    with open(args.inputlist) as f:
      filelist = [line.strip() for line in f if line.strip()]

  outfile = args.outputfile
  neutrino_windows = args.neutrinowindows

  uproot_to_hdf5(filelist, outfile, neutrino_windows)

if __name__ == "__main__":
  main()
