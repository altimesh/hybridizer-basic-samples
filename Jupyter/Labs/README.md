This directory contains our jupyter labs.  
Their purpose is to demonstrate Hybridizer usage, from the very beginning to more advanced examples. 

## Prerequisites
The following software is required to run the labs:
#### Windows
[Visual Studio](https://visualstudio.microsoft.com/downloads/) (2012 and up) with C++ support. 
[CUDA 9.2](https://developer.nvidia.com/cuda-downloads) -- toolkit and driver
[Anaconda](https://www.anaconda.com/download/)
#### Linux
[CUDA 9.2](https://developer.nvidia.com/cuda-downloads) -- toolkit and driver
[Anaconda](https://www.anaconda.com/download/)
[Mono](https://www.mono-project.com/download/stable/#download-lin)
A default C/C++ compiler, such as gcc or clang (`sudo apt-get install gcc g++`)

## Installation
Clone this repository somewhere ([CLONE_DIR])
#### Windows
1. Add Anaconda scripts directory to your %PATH% : [AnacondaInstallDir]\Scripts
2. Install Hybridizer Essentials from [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=altimesh.AltimeshHybridizerExtensionEssentials)
3. Require a license : 
Open Visual Studio, go to *Hybridizer->License Settings* choose your option, and click *Subscribe*
You should receive a subscription by mail. Copy this subscription in the appropriate text box, and click *Refresh License*

#### Linux
1. Install Hybridizer Essentials from our release page [github](https://github.com/altimesh/hybridizer-basic-samples/releases)
2. Require a license (it's done automatically at the end of the deb installer, but you still can do it again using 
`/opt/altimesh/hybridizer-essentials/bin/RequestLicense.exe`
3. Permanently add Hybridizer install directory to your path, for example by putting that in your ~/.profile:
`export PATH=$PATH:/opt/altimesh/hybridizer-essentials/bin/`


## Run
Go to [CLONE_DIR]\Jupyter\Labs and run:
`jupyter notebook`