"""Frozen species/camera/class/crop/view weights for the joint ridge objective."""
import collections
import numpy as np

def island_weights(rows):
 if not rows:raise ValueError('empty island practice')
 counts=collections.Counter(r['truth_species']for r in rows)
 return np.asarray([1/(len(counts)*counts[r['truth_species']])for r in rows],np.float64)

def camera_weights(rows):
 if not rows:raise ValueError('empty camera practice')
 key=lambda r:(str(r['dataset']).casefold(),str(r['location']).casefold())
 by=collections.defaultdict(collections.Counter)
 for r in rows:by[key(r)][r['label']]+=1
 return np.asarray([1/(len(by)*len(by[key(r)])*by[key(r)][r['label']])for r in rows],np.float64)

def view_weights(views,crop_weight):
 if views<1 or crop_weight<=0:raise ValueError('positive views and crop weight required')
 if views==1:return np.asarray([crop_weight],np.float64)
 return np.asarray([crop_weight/2]+[crop_weight/(2*(views-1))]*(views-1),np.float64)
