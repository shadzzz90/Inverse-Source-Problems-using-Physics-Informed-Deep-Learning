import json
import matplotlib as mplt
from operator import add
import matplotlib.pyplot as plt
plt.rc('font', size=12)


folder = "Trial1"

def plot_losses():

    with open(f".//{folder}//total_loss_lst", 'r') as tfile, open(f".//{folder}//physics_loss_lst", 'r') as phyfile,\
            open(f".//{folder}//vel_loss_lst", 'r') as velfile, open(f".//{folder}//disp_loss_lst", 'r') as dispfile:

        tlist = json.load(tfile)
        phylst = json.load(phyfile)
        vel_lst = json.load(velfile)
        disp_lst = json.load(dispfile)

    data_loss = list( map(add, vel_lst, disp_lst ) )
    epoch_lst = [i for i in range(len(phylst))]

    plt.figure(1,figsize=(6,6))
    plt.semilogy(epoch_lst,tlist)
    plt.grid(True, which="both",linestyle ='--')
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', numticks=5)
    plt.xlabel('epochs')
    plt.gca().xaxis.set_major_formatter(mplt.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.ylabel('total loss')
    plt.savefig('total_loss.eps', format='eps')

    plt.figure(2,figsize=(6,6))
    plt.semilogy(epoch_lst, phylst)
    plt.grid(True, which="both",linestyle ='--')
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', numticks=5)
    plt.gca().xaxis.set_major_formatter(mplt.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.xlabel('epochs')
    plt.ylabel('physics loss')
    plt.savefig('phy_loss.eps', format='eps')

    plt.figure(3, figsize=(6, 6))
    plt.semilogy(epoch_lst, data_loss)
    plt.grid(True, which="both",linestyle ='--')
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', numticks=5)
    plt.gca().xaxis.set_major_formatter(mplt.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.xlabel('epochs')
    plt.ylabel('data loss')
    plt.savefig('data_loss.eps', format='eps')

plot_losses()