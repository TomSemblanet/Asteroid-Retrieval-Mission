import numpy as np 

import matplotlib as mpl
import matplotlib.pyplot as plt 

file = open('/Users/semblanet/Desktop/Git/Asteroid-Retrieval-Mission/local/target_determination/deltaV.txt', 'r')

results = list()

for line in file.readlines():
	data = line.split(' ')
	ast_name = data[0] + ' ' + data[1]

	if ast_name == '2020 CD3':
		dV = 300
	else:
		dV = float(data[2])

	if dV < 850:
		results.append([ast_name, dV])

print(len(results))
input()


fig = plt.figure()
plt.bar([data[0] for data in results], [data[1] for data in results])
plt.xticks(rotation =-45, fontsize=9.5)

plt.ylabel('Delta-V [m/s]')
plt.title('Earth - NEA trajectory Delta-V')
plt.show()
