import sys
import re
import math

def frexp10(f):
    n = int(math.floor(math.log10(abs(f)))) if f != 0 else 0
    m =  f/10**n
    return (m,n)

temperature = 300
kT = 1.986e-3*temperature
beta = 1./kT
wgpt = re.compile("wg = +\[([^\]]+)\].*")
ubpt = re.compile("ub = +\[([^\]]+)\].*")
sbpt = re.compile("sb = +\[([^\]]+)\].*")
pbpt = re.compile("pb = +\[([^\]]+)\].*")
eljpt = re.compile("elj = +\[([^\]]+)\].*")
ucept = re.compile("uce = +\[([^\]]+)\].*")
nlpt = re.compile("nl = +\[([^\]]+)\].*")

nmodes = 0
for line in sys.stdin:
    m = wgpt.match(line)
    if m:
        wg = [ float(f) for f in m.group(1).split()]
        nmodes = len(wg)
    m = ubpt.match(line)
    if m:
        ub = [ float(f) for f in m.group(1).split()]
    m = sbpt.match(line)
    if m:
        sb = [ float(f) for f in m.group(1).split()]
    m = pbpt.match(line)
    if m:
        pb = [ float(f) for f in m.group(1).split()]
    m = eljpt.match(line)
    if m:
        elj = [ float(f) for f in m.group(1).split()]
    m = ucept.match(line)
    if m:
        uce = [ float(f) for f in m.group(1).split()]
    m = nlpt.match(line)
    if m:
        nl = [ float(f) for f in m.group(1).split()]
for i in range(nmodes):
    uc = elj[i]*uce[i]
    (wm,we) = frexp10(wg[i])
    (pm,pe) = frexp10(pb[i])
    #ub1 = ub[i] - beta*sb[i]*sb[i]
    print("mode %d & $%.2f \\times 10^{%d}$ & $%.2f \\times 10^{%d}$ & $%.2f$ & $%.2f$ & $%.1f$ & $%.1f$ & $%.1f$ \\tabularnewline" % (i+1,wm-0.01,we,pm,pe,ub[i],sb[i],elj[i],uc,nl[i]))
    #print("mode %d & $%.2e$ & $%.2e$ & $%.2f$ & $%.2f$ & $%.1f$ & $%.1f$ & $%.1f$ \\tabularnewline" % (i+1,wg[i],pb[i],ub[i],sb[i],elj[i],uc,nl[i]))
        
