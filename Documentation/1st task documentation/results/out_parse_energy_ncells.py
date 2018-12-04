#!/usr/bin/python
# a short script to extract from output CG iterations, Total Wall Time for solution and RHS assembly etc.
import argparse
import re


# define command line arguments
parser = argparse.ArgumentParser(description='a short script to extract from '
                                             'output relevant data. '
                                             'This is '
                                             'convinient for plotting')
parser.add_argument('file', metavar='file',
                    help='Output file.')
args = parser.parse_args()

print 'Parse', args.file, 'input file...'
# ready to start...
input_file = args.file
output_file = input_file+'_energy_ncells.parsed'

fin = open(input_file, 'r')
fout = open(output_file, 'w')

pattern = r'[+\-]?(?:[0-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?'

fout.write('NumberOfCells	TotalEnergy \n')
cycle = -1
err = -1
for line in fin:
# If blank line go to next line
    if not line.strip():
	#fout.write('\n')
        continue
    else:
	if 'Number of active cells' in line:
	    ncells = re.findall(pattern, line)[0]
	    line_striped = line.lstrip()
	    count = 0
            # line does not start with a number and we already parsed Cycle line:
            if not line_striped[0].isdigit():
	        # now add all numbers we can find in the line:
                for item in re.findall(pattern, line):
		    count = count + 1
		    #fout.write('{0}'.format(count))
		    if count == 1:
	                fout.write('{0}'.format(item))

        if 'Total energy' in line:
            err = re.findall(pattern, line)[0]
	    fout.write('\t{0}'.format(err)) 
            fout.write('\n')    
        
        if 'Starting epilogue' in line:
           fout.close()
           break

print "done!"
