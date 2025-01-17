# A Raspberry Pi MLX9064 Thermal Camera C Interface
I decided to rewrite most of tomshaffner/PiThermalCam in C because the python dependencies were too bloated and slow for my pi zero W. 
However, the PiThermalCam project is still a great project if you feel differently about python and dependency management. 
The interface with the MLX9064 is a rewritten version of adafruit_mlx90640.py in C.
The code is C++ because it is required for the opencv interface.

## Compiling
Dependencies:
```
sudo apt-get install libopencv-dev libbcm2835-dev g++
```
Compiling:
```
make
```
Ideally, you should not have to compile if you are using raspberian OS and the 32-bit version of the pi. However, the included binary could have other issues.

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
Here is an example of the out.jpg image generated by the thermal binary:

![Thermal Image of Me](/out_example.jpg)

### Recomendations
The program does not like to run above a baud rate of 200000 on my pi zero W. I found a note in adafruit_mlx90640.py saying the MLX9064 really does not like it when the host is slower than itself. So, eventhough the MLX9064 advertises a 1MHz baud rate I don't think its attainable with the pi zero W. This is why the default rate is 62.5 kHz to be safe.

## Install as a systemd service
Add the binary and service to the global path locations (aka installing).
```
sudo cp thermal /usr/bin/thermal
sudo cp thermal.service /usr/lib/systemd/system
sudo mkdir /usr/share/thermal
sudo cp index.html /usr/share/thermal
```
Enable and start the service
```
sudo systemctl enable --now thermal
```
Now a instance of thermal should be running with the default settings streaming on port 8000!
This service will automatically startup whenever the pi is booted up.

