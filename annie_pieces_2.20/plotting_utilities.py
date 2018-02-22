def rect(x,y,w,h,c,ax):
    polygon = plt.Rectangle((x,y),w,h,color=c)
    ax.add_patch(polygon)
def dist_fill(X,Y, cmap,ax):
        plt.plot(X,Y,lw=0)
        dx = X[1]-X[0]
        N  = float(X.size)
        for n, (x,y) in enumerate(zip(X,Y)):
            color = cmap[n,:]
            rect(x,0,dx,y,color,ax)
            
