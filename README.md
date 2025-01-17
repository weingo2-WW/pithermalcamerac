# A Raspberry Pi MLX9064 Thermal Camera C Interface
I decided to rewrite most of tomshaffner/PiThermalCam in C because the python dependencies were too bloated and slow for my pi zero W. 
However, the PiThermalCam project is still a great project if you feel differently about python and dependency management. 
The interface with the MLX9064 is a rewritten version of adafruit_mlx90640.py in C.

## Compiling
```
sudo apt-get install libopencv-dev
```
## Example
![Thermal Image of Me](/out_example.jpg)
