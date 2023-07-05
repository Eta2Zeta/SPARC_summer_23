import get_ascot_AT as myread
import numpy as np
import myprompts as my
import matplotlib as mpl
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages
import pdb

def pnp_losses():

    #  some stuff to get ready for the plots
    do_rasterized   = True
    do_rasterized_2 = True
    
    mpl.rcParams['image.composite_image']=False
    mpl.rcParams['pdf.fonttype']=42
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['xtick.direction']='in'
    mpl.rcParams['ytick.direction']='in'
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['xtick.minor.size'] = 3.5
    mpl.rcParams['ytick.minor.size'] = 3.5
    mpl.rcParams.update({'font.size':16})
    padsize=1.5
    plt.rcParams['xtick.minor.visible']=True
    plt.rcParams['ytick.minor.visible']=True

    
    fn       = my.prompt_string("\n   Enter name of ASCOT output file: ", "")
    pnp_time = my.prompt_float("   enter start time [sec] of nonprompt losses: ", 3.e-5)
   
    end = myread.get_ini_end(fn, 'endstate')
    ini = myread.get_ini_end(fn, 'inistate')
    
    ii_end      = end['ii_lost']
    ii_survived = end['ii_survived']
    
    time_end = end['time'][ii_end]

    timelost_max = np.max(time_end)
    
    ekev_end = end['ekev'][ii_end]
    ekev_ini = ini['ekev']

    weight_end = end['weight'][ii_end]
    weight_ini = ini['weight']

    jj_prompt    = (time_end <  pnp_time)
    jj_nonprompt = (time_end >= pnp_time)

    energy_ini            = np.sum( weight_ini*ekev_ini)
    energy_prompt_loss    = np.sum( weight_end[jj_prompt]    * ekev_end[jj_prompt])
    energy_nonprompt_loss = np.sum( weight_end[jj_nonprompt] * ekev_end[jj_nonprompt])

    floss_prompt    = energy_prompt_loss    / energy_ini
    floss_nonprompt = energy_nonprompt_loss / energy_ini
    floss_total     = (energy_prompt_loss + energy_nonprompt_loss) / energy_ini

    nlost_prompt    = ekev_end[jj_prompt].size
    nlost_nonprompt = ekev_end[jj_nonprompt].size

    print("")
    print(" percentage of power loss (prompt):     %6.3f "%(100*floss_prompt))
    print(" percentage of power loss (nonprompt):  %6.3f "%(100*floss_nonprompt))
    print(" percentage of power loss (total):      %6.3f "%(100*floss_total))
    print()
    print(" number of prompt-lost markers:         %8d   "%(nlost_prompt))
    print(" number of nonprompt-lost markers:      %8d   "%(nlost_nonprompt))
    print("")
    print("  maximum simulation time of lost markers  %10.3e"%(timelost_max))

    # ++++++++++++++++++++++++++++++++++++++++++
    #  some plots

    #pdb.set_trace()
    stub = fn.split('.')[0]
    filename_multi= stub + '_remy.pdf'

    print("\n ... there is graphical output in file: ",filename_multi, "\n")
    
    with PdfPages(filename_multi) as pdf:

        #  +++++++++++++++++++++++++++++++++++
        #    cumulative weighted energy loss
        
        qq_all = ekev_end * weight_end
        qq_np  = ekev_end[jj_nonprompt] * weight_end[jj_nonprompt]

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(time_end, bins=100, histtype='step', color='b', cumulative=True, density=True, weights=qq_all)
        plt.hist(time_end[jj_nonprompt], bins=100, histtype='step', color='r', cumulative=True, density=True, weights=qq_np)
        plt.ylim((0,1))
        plt.xlim(left=0.)
        plt.xlabel('time [sec]', fontsize=10)
        plt.grid('both')
        my_title = stub + ' cumulative weighted energy loss (b=total r=nonprompt)'
        plt.title(my_title, fontsize=10)
        plt.tight_layout(pad=1)
        pdf.savefig()

        #  CPU time versus simtime

        
if __name__ == '__main__':

    pnp_losses()
