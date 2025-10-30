echo -e "Downloading T2M evaluators"
# gdown --fuzzy https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view
# gdown --fuzzy https://drive.google.com/file/d/1tX79xk0fflp07EZ660Xz1RAFE33iEyJR/view
gdown 1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8
tar -xvzf t2m.tar.gz
# rm -rf t2m
# rm -rf kit

# unzip t2m.zip
# unzip kit.zip
echo -e "Cleaning\n"
# rm t2m.zip
# rm kit.zip
rm t2m.tar.gz

echo -e "Downloading done!"