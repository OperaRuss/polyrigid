import glob
import os
import numpy as np
import matplotlib.pyplot as plt

fpResults = "../images/results"

fig, ax = plt.subplots(1,1)
for file in sorted(glob.glob(fpResults+"/*")):
    if os.path.isdir(file):
        results = np.load(os.path.join(file, 'model_results.npz'))
        label = results['target']
        loss = np.load(os.path.join(file, 'model_loss.npz'))
        temp = {}
        for k,v in loss.items():
            temp[int(k)] = v

        x,y = zip(*sorted(temp.items()))
        ax.plot(x,y,color='b')
ax.set_title("MSE Loss for All Frames in Dynamic Sequence")
ax.set_ylabel("MSE Loss")
ax.set_xlabel("Iteration")
plt.xticks(np.arange(0,201,20))
plt.show()
plt.savefig("../images/results/summary_MSE.png",bbox_inches='tight')
plt.close()

finLoss = {}
netDICE = {}
percNegJDets = {}
meanRigid = {}

fig, axs = plt.subplots(2,2,figsize=(10,10)) # in clockwise from upper left: numNeg, netDICE, percentageNegJDets,meanRigidityScore
for file in sorted(glob.glob(fpResults+"/*")):
    if os.path.isdir(file):
        results = np.load(os.path.join(file, 'model_results.npz'))
        label = int(results['target'])
        finLoss[label] = results['finalLoss']
        netDICE[label] = results['netDICE']
        percNegJDets[label] = results['percentageNegJDets']
        meanRigid[label] = results['meanRigidityScore']

x,y = zip(*sorted(finLoss.items()))
axs[0,0].plot(x,y)
axs[0,0].set(title="Final Loss Score Achieved")
axs[0,0].set_xticks(np.arange(0,21,2))

x,y = zip(*sorted(netDICE.items()))
axs[0,1].plot(x,y)
axs[0,1].set(title="Net DICE Gain")
axs[0,1].set_xticks(np.arange(0,21,2))

x,y = zip(*sorted(percNegJDets.items()))
axs[1,0].plot(x,y)
axs[1,0].set(title="% Neg JDets")
axs[1,0].set_xticks(np.arange(0,21,2))

x,y = zip(*sorted(meanRigid.items()))
axs[1,1].plot(x,y)
axs[1,1].set(title="Mean Rigidity Score")
axs[1,1].set_xticks(np.arange(0,21,2))

fig.suptitle("Final Evaluation Scores After Registration on All Frames")
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig("../images/results/summary_metrics.png",bbox_inches='tight')
plt.show()
plt.close()



