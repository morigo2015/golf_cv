SWING_DIR=video/swings
inotifywait -m -e close_write --format "%f" $SWING_DIR \
	| while read FILENAME
		do
			echo Detected $FILENAME
			vlc -R --rate 1.0 --verbose 0  --one-instance  $SWING_DIR/$FILENAME 2>nul &
		done
