echo -e "spliting raw behave dataset"

# cd dataset

# mkdir behave-30fps-params

# tar -xvf behave-30fps-params-v1.tar -C ./behave-30fps-params/
# echo -e "Cleaning\n"
# rm behave-30fps-params-v1.tar

# echo -e "Cleaning Done!\n"

echo -e "spliting data now!\n"

# cd ..
python utils/behave_process.py

echo -e "Done!"