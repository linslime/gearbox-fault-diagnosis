import h5py

f = h5py.File('Model.h5', 'r')

print(f.attrs.get('keras_version'))