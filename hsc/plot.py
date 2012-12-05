import numpy as np
import pylab as pl

def hsc_plot(x_lst,y_lst,x_name,y_name,plot_name):
    fig = pl.figure()
    sbplot = fig.add_subplot(211)
    sbplot.set_title(plot_name)
    sbplot.plot(x_lst,y_lst,color="red")
    sbplot.set_xlabel(x_name)    
    sbplot.set_ylabel(y_name)
    pl.show()

if __name__=="__main__":
    var = [1,2,3,4]
    accuracies = [3,4,8,6]
    hsc_plot(var,accuracies,"var","accuracies","Plot Title")
    
