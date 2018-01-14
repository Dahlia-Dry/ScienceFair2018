# Author Dahlia Dry
# Last Modified 12/8/2017
# This program optimizes the performance of the neural net's data generator by making it threadsafe

import threading

class threadsafe_iter(object):
  def __init__(self, it):
      self.it = it
      self.lock = threading.Lock()

  def __iter__(self):
      return self

  def __next__(self):
      with self.lock:
          return self.it.__next__()

def threadsafe_generator(f):
  def g(*a, **kw):
      return threadsafe_iter(f(*a, **kw))
  return g

