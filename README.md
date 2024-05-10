# AIAS-Project-2
This method uses a population-based metaheuristic to denoise an OCT scan by overlaying Bezier curves that may 'camouflage' to add greater definition to the retinal layers

# Usage
```
usage: optimize.py [-h] [--image IMAGE] [--multiscale] [--recursive] [--recursive_depth RECURSIVE_DEPTH] [--iterative]

options:
  -h, --help            				        show this help message and exit
  --image IMAGE         				        Path to the image
  --multiscale          				        Use multiscale optimisation
  --recursive           				        Use recursive subdivision optimisation
  --recursive_depth RECURSIVE_DEPTH             Depth of the recursive subdivision (default depth = 4)
  --iterative           				        Use iterative optimisation
```

<b> ```python3 optimize.py``` will evaluate the image 'original.png' using both multiscale and recursive subdivision </b>
