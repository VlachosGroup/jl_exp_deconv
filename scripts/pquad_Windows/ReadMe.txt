The file explains how to use the pquad files for Windows.

1a) Ensure that your pure component data is saved as .csv files in a format similar to the data in the pure_files directory.
1b) If you want to use the default pure-component data skip step 1a and enter "no" when prompted for entering your own data in step 6.
2) Ensure mixture data data with known concentrations is saved as .csv files in a manner similar to the the data in the mixture_files. The
labels in the first row should be contained in the names of the files from 1a. Spaces and underscores are interchangeable. This ensures that the predicted
pure-component concentration can be matched to the actual pure-component concentration.
3) Ensure that mixture data without known concentrations is saved as .csv files in a mannery similar to the data in the reaction_files directory.
4) Double click on Miniconda3-latest-Windows-x86_64.exe to install python on your machine (you can do this directly from python's website if you wish
https://docs.conda.io/en/latest/miniconda.html). Ensure that python is added to the windows PATH environment
*This means checking the box that says "Add Anaconda to the system PATH environment variable"
**Using miniconda or anaconda is important to get scipy to work correctly on windows.
5) Double click on install_pquad_bat.bat (only needs to be done once).
6) Double click on run_pquad_bat.bat to generate predicted concentrations and deconvoluted data files.
*when prompted directory paths should look like the following: C:\Users\Username\Downloads\pquad_Windows\mixture_files