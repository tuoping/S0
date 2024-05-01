import numpy as np
from scipy.optimize import curve_fit
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
def func(x, a, b):
    return a / (1 + b * x)

eps = 0.001
plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
_xdata = np.loadtxt("Sk-kgrid1D.dat")[1:]
_kgriddata = np.loadtxt("Sk-kgrid.dat")[1:]
_ydata_input = np.loadtxt("Sk-"+sys.argv[1]+"-real.dat")
interval = 80
num_intervals = len(_ydata_input)//interval+1
i_start_list = list(range(0,len(_ydata_input),interval))[:num_intervals]
i_end_list = list(range(interval,len(_ydata_input)+interval,interval))[:num_intervals]
print("Num of fits = ", len(i_start_list))
n = len(i_start_list)+1
import matplotlib
line_colar = [matplotlib.colormaps["gnuplot"](float(i)/float(n)) for i in range(n)]

selk = np.array([i for i in range(len(_kgriddata))])
print(selk.shape)
plt.figure()
for idx_i_start in range(len(i_start_list)):
    _ydata = np.average(_ydata_input[i_start_list[idx_i_start]:i_end_list[idx_i_start]], axis=0)[1:]
    plt.scatter(_xdata[selk],_ydata[selk], s=3, color=line_colar[idx_i_start])
plt.savefig("Sk-"+sys.argv[1]+"-real.dat"[:-4]+".png")
plt.show()

plt.figure()
ax1 = plt.subplot(111)
_xdata = np.loadtxt("Sk-kgrid1D.dat")[1:]

s0 = []
s0_accumulate = []
s0_accumulate_start = []
cov_accumulate_start = []
ofit = open("fit-"+"Sk-"+sys.argv[1]+"-real.dat", "w")
for idx_i_start in range(len(i_start_list)):
    _ydata_accumulate_start = np.average(_ydata_input[:i_end_list[idx_i_start]], axis=0)[1:]
    print(min(_xdata), max(_xdata))
    xdata = _xdata[selk]
    print(min(xdata), max(xdata))
    ydata_accumulate_start = _ydata_accumulate_start[selk]
    popt_accumulate_start, pcov_accumulate_start = curve_fit(func, xdata, ydata_accumulate_start)
    s0_accumulate_start.append(popt_accumulate_start[0])
    cov_accumulate_start.append(pcov_accumulate_start[0][0])
    print("Fit parameters:: ", popt_accumulate_start)
    print("Covariance", pcov_accumulate_start)
    print("Relative err of estimate: ", np.sqrt(np.diag(pcov_accumulate_start))/popt_accumulate_start)
    
    ofit.write(f"Fit parameters:: {popt_accumulate_start[0]}  {popt_accumulate_start[1]}\n")
    ofit.write(f"Covariance:: [[{pcov_accumulate_start[0][0]}, {pcov_accumulate_start[0][1]}],  [{pcov_accumulate_start[1][0]}, {pcov_accumulate_start[1][1]}]\n")
    ofit.write(f"Relative err of estimate: {(np.sqrt(np.diag(pcov_accumulate_start))/popt_accumulate_start)[0]}  {(np.sqrt(np.diag(pcov_accumulate_start))/popt_accumulate_start)[1]}\n")
    
    ax1.plot(sorted(xdata),func(np.array(sorted(xdata)),popt_accumulate_start[0],popt_accumulate_start[1]), c=line_colar[idx_i_start])
    ax1.scatter(xdata,ydata_accumulate_start,s=3)
    
    # ax2.plot(sorted(xdata),func(np.array(sorted(xdata)),popt[0],popt[1]), c=line_colar[idx_i_start])
    # ax2.scatter(xdata,ydata,s=3)

plt.savefig("fit-"+"Sk-"+sys.argv[1]+"-real.dat"[:-4]+".png")
plt.show()


plt.figure()
# plt.plot(i_end_list, s0, marker="o")
# plt.plot(i_start_list, s0_accumulate, marker="x")
plt.plot(i_end_list, s0_accumulate_start, marker="s")
plt.savefig("s0-"+"Sk-"+sys.argv[1]+"-real.dat"[:-4]+".png")
with open("s0-"+"Sk-"+sys.argv[1]+"-real.dat"[:-4]+".dat", "w") as ofile:
    ofile.write("# i_start s0-CC cov-s0-CC\n")
    for i in range(len(i_start_list)):
        ofile.write(f"{i_start_list[i]} {s0_accumulate_start[i]}  {cov_accumulate_start[i]}\n")
