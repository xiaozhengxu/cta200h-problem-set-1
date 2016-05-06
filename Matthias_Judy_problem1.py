import glob
import os
import sys

directory = os.getcwd() #string of current working directory
filenames = glob.glob(directory) #glob of all file names in directory
print filenames

find_word = sys.argv[1]
replace_word = sys.argv[2]

replacedirectory = directory+'/'+replace_word

try:
	os.mkdir(replacedirectory)
except OSError:
	pass
	
for i,fn in enumerate(filenames):
	f = open(fn,'r')
	text = f.read()
	f.close()

	if text.find(find_word) != -1:
		newtext = text.replace(find_word,replace_word)

		f = open(replacedirectory+'/'+fn,'w')
		f.write(newtext)
		f.close()




