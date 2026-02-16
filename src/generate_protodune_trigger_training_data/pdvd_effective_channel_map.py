#!/usr/bin/env python3

class PDVDEffectiveChannelMap:
  def __init__(self, first_channel, n_channels):
    self.first_channel = first_channel
    self.n_channels = n_channels
    self.n_channels_crp_block = self.n_channels / 4 # four crps in a plane, should = 292
    self.n_effective_channels = self.n_channels / 2

  def get_effective_channel_id(self, channel_id):
    # for pdvd, the effective channel id is the channel id within a crp block 
    # so that the same effective channel id corresponds to the same physical location 
    # on each crp in the plane 
    channel_id_in_plane = channel_id - self.first_channel
    crp_block = channel_id_in_plane // self.n_channels_crp_block
    channel_id_in_crp_block = channel_id_in_plane % self.n_channels_crp_block
    base_channel  = (crp_block >= 2) * self.n_channels_crp_block
    return channel_id_in_crp_block + base_channel + self.first_channel
  
  def get_n_effective_channels(self): return self.n_effective_channels