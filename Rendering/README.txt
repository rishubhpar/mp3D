This document has steps to generate the copy-paste rendering given a set of label files. 

INSTALLATION
Install the required dependencies from requirements.txt

## COPY-PASTE RENDERING ##
Please follow the steps to generate copy-paste rendering for a given label file.

Step 1 : Create a database of copy-paste cars with metadata including
    a) Set the PATHS appropriately in 'copy_paste_database_create.py' (update the commented lines)
    b) Run "python copy_paste_database_create.py"
    c) Check the .csv database at the output location and use the same Path for Step 2 

Step 2 : Render cars on scene using the copy-paste database
    a) Set the PATHS for root directory, image folder, database path obtained from Step 1, and label folder path, appropriately in 'render_copy_paste_cars_on_scene.py' (update the commented lines)
    b) Run "python render_copy_paste_cars_on_scene.py"
