# 3d_vol_balls

This is a small test program. It was inspired by the request of a friend who needed a program
to generate a couple of spheres (and "a couple" means many) in a 3D volume and get slice images of them. 
As I'm learning Rust at the moment I used the opportunity to test **multi-threading**, **OpenCL**, 
**memory mapping** and **gtk** integration in Rust. I recommend compiling the package with 
`cargo build --release` to get proper performance.

## Usage
In the GUI you have to select one or more OpenCL devices (an OpenCL capable machine is mandatory) and
an output folder. Only then the "Start Calculation" button becomes active and you
can start the computation. You may want to be careful when increasing the resolution values, the program is on the
memory hogging side, although with the memory mapping used it should half-way gracefully get along with 
dataset sizes beyond RAM capacity.
