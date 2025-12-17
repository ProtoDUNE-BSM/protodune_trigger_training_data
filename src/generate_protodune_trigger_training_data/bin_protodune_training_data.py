#!/usr/bin/env python3
import argparse
import h5py
import numpy as np
import pandas as pd
import os
import sys

def parse_args():
  parser = argparse.ArgumentParser(
      description="Create binned images from TPWindowTree HDF5 data."
      )

  parser.add_argument(
      "-i", "--inputfile",
      required=True,
      help="Input .hdf5 file produced by uproot_to_hdf5"
      )

  parser.add_argument(
      "-o", "--outputfile",
      required=True,
      help="File to save binned images (.npz)"
      )
  
  parser.add_argument(
      "-d", "--detector",
      required=True,
      help="Which protodune detector, np04 or np02"
      )
  
  parser.add_argument(
      "-t", "--ntimebins",
      required=True,
      help="Number of time bins"
      )
  
  parser.add_argument(
      "-c", "--nchannelbins",
      required=True,
      help="Number of channel id bins"
      )
  
  parser.add_argument(
      "-p", "--planechoice",
      type=int,
      default=2,
      required=False,
      help="Choose plane ID to make images for (default = 2)."
      )

  parser.add_argument(
      "--noapa1",
      action="store_true",
      help="Option to ignore broken apa 1 for np04"
      )

  return parser.parse_args()


def build_pdhd_plane_map():
  data = [
      {"tpc": 1, "plane": 0, "first_channel": 400,  "n_channels": 400},
      {"tpc": 1, "plane": 1, "first_channel": 1200, "n_channels": 400},
      {"tpc": 1, "plane": 2, "first_channel": 2080, "n_channels": 480},

      {"tpc": 2, "plane": 0, "first_channel": 2560, "n_channels": 400},
      {"tpc": 2, "plane": 1, "first_channel": 3360, "n_channels": 400},
      {"tpc": 2, "plane": 2, "first_channel": 4160, "n_channels": 480},

      {"tpc": 3, "plane": 0, "first_channel": 5520, "n_channels": 400},
      {"tpc": 3, "plane": 1, "first_channel": 6320, "n_channels": 400},
      {"tpc": 3, "plane": 2, "first_channel": 7200, "n_channels": 480},

      {"tpc": 4, "plane": 0, "first_channel": 7680, "n_channels": 400},
      {"tpc": 4, "plane": 1, "first_channel": 8480, "n_channels": 400},
      {"tpc": 4, "plane": 2, "first_channel": 9280, "n_channels": 480}
  ]

  df = pd.DataFrame(data)
  return df

def build_pdvd_plane_map():
  data = [
      {"tpc": 2, "plane": 0, "first_channel":  6144, "n_channels":  952},
      {"tpc": 2, "plane": 1, "first_channel":  7096, "n_channels":  952},
      {"tpc": 2, "plane": 2, "first_channel":  8048, "n_channels": 1168},

      {"tpc": 3, "plane": 0, "first_channel":  9216, "n_channels":  952},
      {"tpc": 3, "plane": 1, "first_channel": 10168, "n_channels":  952},
      {"tpc": 3, "plane": 2, "first_channel": 11120, "n_channels": 1168},

      {"tpc": 4, "plane": 0, "first_channel":  3072, "n_channels":  952},
      {"tpc": 4, "plane": 1, "first_channel":  4024, "n_channels":  952},
      {"tpc": 4, "plane": 2, "first_channel":  4976, "n_channels": 1168},

      {"tpc": 5, "plane": 0, "first_channel":     0, "n_channels":  952},
      {"tpc": 5, "plane": 1, "first_channel":   952, "n_channels":  952},
      {"tpc": 5, "plane": 2, "first_channel":  1904, "n_channels": 1168}
  ]

  df = pd.DataFrame(data)
  return df


def bin_subevent(
    timepeak, channelid, adcintegral,
    apa_mask, time_bins, channel_bins
    ):

  # bin so that channel bins are the x-axis and time bins are the y-axis
  # just like in an event display
  hist2d, _, _ = np.histogram2d(
      channelid[apa_mask],
      timepeak[apa_mask],
      bins=[channel_bins, time_bins],
      weights=adcintegral[apa_mask]
      )

  return hist2d

def main():
  args = parse_args()

  if args.detector == "np04":
    det_map = build_pdhd_plane_map()
  elif args.detector == "np02":
    det_map = build_pdvd_plane_map()
  else:
    sys.exit('ProtoDUNE detectors are either np04 or np02')

  if int(args.planechoice) not in {0, 1, 2}:
    sys.exit("Choice of plane ID must be 0, 1 or 2.")


  print('[START] BIN PROTODUNE TRAINING DATA')
  print(f'input file: {args.inputfile}')
  print(f'output file: {args.outputfile}')
  print(f'detector: {args.detector}')
  print(f'chosen plane: {args.planechoice}')

  all_images = []      # <- flat list of 2D histograms
  
  with h5py.File(args.inputfile, "r") as f:
    events = f["events"]
    print(f'There are {len(events)} events in the file')
    for event_name in events:
      event = events[event_name]
      
      timepeak_event    = event["Window_timepeak"]
      channelid_event   = event["Window_channelid"]
      adcintegral_event = event["Window_adcintegral"]
      apacrp_event      = event["Window_apacrp"]
      planeid_event     = event["Window_planeid"]

      # Loop over sub-events
      for sub in timepeak_event:
        timepeak = timepeak_event[sub][:]
        channelid = channelid_event[sub][:]
        adcintegral = adcintegral_event[sub][:]
        apacrp = apacrp_event[sub][:]
        planeid = planeid_event[sub][:]

        # Unique APAs/CRPs for this sub-event
        unique_apacrp = np.unique(apacrp)

        if args.noapa1 and args.detector == "np04":
          unique_apacrp = unique_apacrp[unique_apacrp != 1]

        for apacrp_id in unique_apacrp:
          mask = (apacrp == apacrp_id) & (planeid == int(args.planechoice))

          # get time and channel binning - only doing collection plane for now
          # todo: loop over planes as well 
          plane_chans = det_map[(det_map.tpc == apacrp_id) & (det_map.plane == int(args.planechoice))]

          first_channel = plane_chans.first_channel.item()
          last_channel = first_channel + plane_chans.n_channels.item()

          # Create channel binning for this sub-event in this apa/crp
          channel_bins = np.linspace(first_channel, last_channel, int(args.nchannelbins) + 1)

          if not timepeak[mask].size:
            continue

          earliest_time = np.min(timepeak[mask])

          # Create time binning based on the first TP time in the window plus 20k ticks
          time_bins = np.linspace(earliest_time, earliest_time + 20000, int(args.ntimebins) + 1)

          # create image
          img = bin_subevent(
              timepeak,
              channelid,
              adcintegral,
              mask,
              time_bins,
              channel_bins
              )

          # If you were to filter training data based on aggregate TP properties 
          # (e.g. total charge) then place those cuts here
          all_images.append(img)

  print('saving images in a .npz file')
  np.savez_compressed(args.outputfile,
                      images=np.array(all_images))

  print(f"Saved {len(all_images)} images to {args.outputfile}")

if __name__ == "__main__":
  main()
