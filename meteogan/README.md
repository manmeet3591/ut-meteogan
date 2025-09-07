apptainer build meteogan.sif meteogan.def
apptainer exec --nv ../ai_land_model/apptainer_al_land.sif python train.py
