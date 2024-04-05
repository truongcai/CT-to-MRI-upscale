# How to Run this model

## Datasetup and Model Running

The first step is to download the kaggle dataset that we used for the model and download it into ur local orr instance folders using `kaggle datasets download -d darren2020/ct-to-mri-cgan`. Then, you would run the movedata.py script using `python movedata.py` and use the argparse options to give whatever the dataroot and batchsize you want to give to properly sort the images into respective folders. Following that, you would want to run `python full_model_run.py` with the  --dataroot option to give the baseroot that you would like to use for the training. The script runs the training first by default and saves the weights of the generator and discriminator within the directory or outside of the directory, depending on the filepath given. You can also change the model stucture in model.py and customize the layers and sizes for different testing. The losses will be stored in the output file that will be made in the directory. 

In order to test the model, you would run the full_model_run.py file with the `--mode test` in order to generate images on the testing dataset. 
