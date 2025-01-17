# A Raspberry Pi MLX9064 Thermal Camera C Interface
I decided to rewrite most of tomshaffner/PiThermalCam in C because the python dependencies were too bloated and slow for my pi zero W. 
However, the PiThermalCam project is still a great project if you feel differently about python and dependency management. 
The interface with the MLX9064 is a rewritten version of adafruit_mlx90640.py in C.

## Compiling
Dependencies:
```
sudo apt-get install libopencv-dev libbcm2835-dev
```
Compiling:
```
make
```

## Example
```
Usage: sudo ./thermal [-hgvn] [-b baud_rate]
        -g uses greyscale instead of the jet colorscale
        -b changes the default baud rate of 62500 to something else
        -v outputs to a video using ffmpeg
        -n disables python fileserver for web stream
        -h displays the usage message and exit
```
Running 
```
sudo ./thermal
```
will start a web server using python -m http.server and start grabbing images from the MLX9064.
![Thermal Image of Me](/out_example.jpg)
