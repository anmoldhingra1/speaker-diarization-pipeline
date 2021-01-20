

## Pipeline

A pipeline takes an audio file as input and  returns agent and customer speech segments as output.

### Usage

```python
# load pipeline
import torch
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')

# apply diarization pipeline on your audio file
diarization = pipeline({'audio': '/path/to/your/audio.wav'})

# dump result to disk using RTTM format
with open('/path/to/your/audio.rttm', 'w') as f:
    diarization.write_rttm(f)
  
# iterate over speech turns
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.')
# Speaker "A" speaks between t=0.2s and t=1.4s.
# Speaker "B" speaks between t=2.3s and t=4.8s.
# Speaker "A" speaks between t=5.2s and t=8.3s.
# Speaker "C" speaks between t=8.3s and t=9.4s.
# ...
```

### Speaker diarization

|   | Pipeline               | Models used internally                       | Development set
|---|------------------------|----------------------------------------------|-----------------
| ✅ | `dia` or `dia_dihard`  | {`sad_dihard`, `scd_dihard`, `emb_voxceleb`} | `DIHARD.custom.dev`
| ✅ | `dia_ami`              | {`sad_ami`, `scd_ami`, `emb_ami`}            | `AMI.dev`


## Models

A model takes an audio file as input and returns a [`pyannote.core.SlidingWindowFeature` instance](http://pyannote.github.io/pyannote-core/reference.html#pyannote.core.SlidingWindowFeature) as output, that contains the raw output of the underlying neural network. 

### Usage

```python
# load model
import torch
model = torch.hub.load('pyannote/pyannote-audio', 'sad')

# apply model on your audio file
raw_scores = model({'audio': '/path/to/your/audio.wav'})
```

Most models can also be loaded as pipelines, using the `pipeline=True` option:

```python
# load model and wrap it in a pipeline
import torch
pipeline = torch.hub.load('pyannote/pyannote-audio', 'sad', pipeline=True)

# apply speech activity detection pipeline on your audio file
speech_activity_detection = pipeline({'audio': '/path/to/your/audio.wav'})

# dump result to disk using RTTM format
with open('/path/to/your/audio.sad.rttm', 'w') as f:
    speech_activity_detection.write_rttm(f)

for speech_region in speech_activity_detection.get_timeline():
    print(f'There is speech between t={speech_region.start:.1f}s and t={speech_region.end:.1f}s.')
# There is speech between t=0.2s and t=1.4s.
# There is speech between t=2.3s and t=4.8s.
# There is speech between t=5.2s and t=8.3s.
# There is speech between t=8.3s and t=9.4s.
# ...
```

### Speech activity detection

|   | Model                | Training set        | Development set
|---|----------------------|---------------------|-----------------
| ✅ | `sad` or `sad_dihard`| `DIHARD.custom.trn` | `DIHARD.custom.dev`
| ✅ |`sad_ami`             | `AMI.trn`           | `AMI.dev`




### Speaker change detection

|   | Model                 | Training set        | Development set
|---|-----------------------|---------------------|-----------------
| ✅ | `scd` or `scd_dihard` | `DIHARD.custom.trn` | `DIHARD.custom.dev`
| ✅ |`scd_ami`              | `AMI.trn`           | `AMI.dev`



### Overlapped speech detection

|   | Model                | Training set        | Development set
|---|----------------------|---------------------|-----------------
| ✅ |`ovl` or `ovl_dihard` | `DIHARD.custom.trn` | `DIHARD.custom.dev`
| ✅ |`ovl_ami`             | `AMI.trn`           | `AMI.dev`


### Speaker embedding

Speaker embedding models cannot be loaded as pipelines.

```python
# load model
import torch
model = torch.hub.load('pyannote/pyannote-audio', 'emb')

print(f'Embedding has dimension {model.dimension:d}.')
# Embedding has dimension 512.

# extract speaker embedding on the whole file using built-in sliding window
import numpy as np
from pyannote.core import Segment
embedding = model({'audio': '/path/to/your/audio.wav'})
for window, emb in embedding:
    assert isinstance(window, Segment)
    assert isinstance(emb, np.ndarray)    

# extract speaker embedding of an excerpt
excerpt1 = Segment(start=2.3, end=4.8)
emb1 = model.crop({'audio': '/path/to/your/audio.wav'}, excerpt1)
assert isinstance(emb1, np.ndarray)

# compare speaker embedding
from scipy.spatial.distance import cdist
excerpt2 = Segment(start=5.2, end=8.3)
emb2 = model.crop({'audio': '/path/to/your/audio.wav'}, excerpt2)
distance = cdist(np.mean(emb1, axis=0, keepdims=True), 
                 np.mean(emb2, axis=0, keepdims=True), 
                 metric='cosine')[0, 0]
```

|    | Model                 | Training set                             | Development set
|----|-----------------------|------------------------------------------|-----------------
| ✅ |`emb` or `emb_voxceleb` | `VoxCeleb1.custom.trn` ⋃ `VoxCeleb2.trn` | `VoxCeleb1.custom.dev`
| ✅ |`emb_ami`               | `AMI.trn`                                | `AMI.dev`





