PREFIX=$1
echo ===============================
echo clip merging
echo "PREFIX=" $PREFIX
echo ===============================

rm mergelist-$PREFIX.txt
ls $PREFIX*.avi | while read each; do echo "file '$each'" >> mergelist-$PREFIX.txt; done
ffmpeg -f concat -i mergelist-$PREFIX.txt -safe 0 -c copy merged-$PREFIX.avi

echo ===============================
echo merged-$PREFIX.avi created
echo ===============================

