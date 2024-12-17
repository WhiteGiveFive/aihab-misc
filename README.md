# aihab-misc
This repo contains miscellaneous tools for the AIHAB project.
## Environment Setup
First create a conda environment with python. For instance, `conda create -n myenv python=3.10`.  
Then activate the created conda environment with `conda activate myenv` and install the dependencies with `pip install -r requirements.txt`

## Compiling the Image Annotator
Run `pyinstaller image_annotator.py` to compile the Image Annotator into an executable file. Then the `exe` file can be found in the generated `dist` folder.

## Running on Image Annotator
The first run of the compiled image annotator could take up to 1 minutes, please be patient. The next run will be quick. 
After running, a window will pop up to ask you to select the folder containing the images that you want to label.

![img.png](pop-up-window.png)

After selecting your folder, the main interface shows your images in the folder.

![img.png](main-window.png)